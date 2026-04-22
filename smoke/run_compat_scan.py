import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import TruncatedSVD
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_smoke as rs


def load_base_model_and_tokenizer(config: Dict):
    model_name = config["model"]["name"]
    model_ref = config["model"].get("local_path", model_name)
    dtype = rs._dtype_from_cfg(config["model"].get("torch_dtype", "float16"))
    device = rs._device()
    local_files_only = bool(config["model"].get("local_files_only", False))

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=config["model"].get("trust_remote_code", False),
        local_files_only=local_files_only,
        use_fast=bool(config["model"].get("use_fast_tokenizer", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        trust_remote_code=config["model"].get("trust_remote_code", False),
        local_files_only=local_files_only,
        low_cpu_mem_usage=bool(config["model"].get("low_cpu_mem_usage", True)),
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def layer_spectrum_stats(model, tokenizer, device, texts_a: List[str], texts_b: List[str], train_cfg: Dict) -> Dict[str, float]:
    max_len = int(train_cfg["max_seq_len"])
    batch_size = max(1, int(train_cfg["batch_size"]))
    loader_a = rs.make_dataloader(texts_a, tokenizer, max_len=max_len, batch_size=batch_size, shuffle=False)
    loader_b = rs.make_dataloader(texts_b, tokenizer, max_len=max_len, batch_size=batch_size, shuffle=False)

    def collect(loader):
        mats = None
        seen = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            hs = out.hidden_states
            if mats is None:
                mats = {i: [] for i in range(1, len(hs))}
            bsz = int(batch["input_ids"].shape[0])
            seen += bsz
            for i in range(1, len(hs)):
                pooled = hs[i].mean(dim=1).float().cpu().numpy()
                mats[i].append(np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0))
            if seen >= 32:
                break
        if mats is None:
            return {}
        return {i: np.concatenate(chunks, axis=0) if chunks else np.zeros((1, 8), dtype=np.float32) for i, chunks in mats.items()}

    a_mats = collect(loader_a)
    b_mats = collect(loader_b)
    vals = []
    for i in sorted(set(a_mats.keys()) & set(b_mats.keys())):
        a = a_mats[i]
        b = b_mats[i]
        if a.shape[1] < 2 or b.shape[1] < 2:
            continue
        if a.shape[0] < 2 or b.shape[0] < 2:
            vals.append(rs.cosine(np.mean(a, axis=0), np.mean(b, axis=0)))
            continue
        svd_a = TruncatedSVD(n_components=1, random_state=42)
        svd_b = TruncatedSVD(n_components=1, random_state=42)
        va = svd_a.fit(a).components_[0]
        vb = svd_b.fit(b).components_[0]
        vals.append(rs.cosine(va, vb))

    if not vals:
        return {
            "spectrum_n_layers": 0.0,
            "spectrum_layers_mean": float("nan"),
            "spectrum_layers_std": float("nan"),
            "spectrum_layers_min": float("nan"),
            "spectrum_layers_max": float("nan"),
            "spectrum_layers_span": float("nan"),
        }

    arr = np.asarray(vals, dtype=np.float64)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    return {
        "spectrum_n_layers": float(len(arr)),
        "spectrum_layers_mean": float(np.mean(arr)),
        "spectrum_layers_std": float(np.std(arr)),
        "spectrum_layers_min": lo,
        "spectrum_layers_max": hi,
        "spectrum_layers_span": float(hi - lo),
    }


def compat_scan(config_path: str, output_dir: str) -> Dict:
    cfg = rs.read_config(config_path)
    rs.set_seed(int(cfg.get("seed", 42)))
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer, device = load_base_model_and_tokenizer(cfg)

    train_n = int(cfg["train"]["max_train_samples"])
    eval_n = int(cfg["train"]["max_eval_samples"])
    local_only = bool(cfg.get("data", {}).get("local_files_only", False))

    rows = []
    for pair in cfg["tasks"]:
        pair_name = pair["name"]
        ta = rs.parse_task(pair["task_a"])
        tb = rs.parse_task(pair["task_b"])
        a_train, _ = rs.sample_task_texts(ta, train_n, eval_n, local_files_only=local_only)
        b_train, _ = rs.sample_task_texts(tb, train_n, eval_n, local_files_only=local_only)

        act = rs.activation_overlap_real(model, tokenizer, device, a_train, b_train, cfg["train"])
        layer_geo = layer_spectrum_stats(model, tokenizer, device, a_train, b_train, cfg["train"])
        rows.append({"pair": pair_name, "activation_spectrum_overlap": float(act), **layer_geo})

    df = pd.DataFrame(rows)
    # 诊断风险分数（仅排序用途）：主项 + 两个辅项按经验符号加权
    for col in ["activation_spectrum_overlap", "spectrum_layers_std", "spectrum_layers_span"]:
        if col not in df.columns:
            df[col] = np.nan

    z_act = (df["activation_spectrum_overlap"] - df["activation_spectrum_overlap"].mean()) / (df["activation_spectrum_overlap"].std() + 1e-12)
    z_std = (df["spectrum_layers_std"] - df["spectrum_layers_std"].mean()) / (df["spectrum_layers_std"].std() + 1e-12)
    z_span = (df["spectrum_layers_span"] - df["spectrum_layers_span"].mean()) / (df["spectrum_layers_span"].std() + 1e-12)
    df["compat_risk_score"] = z_act - 0.5 * z_std - 0.5 * z_span
    df = df.sort_values("compat_risk_score", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(output_dir, "compat_scan.csv")
    json_path = os.path.join(output_dir, "compat_scan_summary.json")
    df.to_csv(csv_path, index=False)

    summary = {
        "config": config_path,
        "output_csv": csv_path,
        "num_pairs": int(len(df)),
        "top_risky_pairs": df[["pair", "compat_risk_score", "activation_spectrum_overlap"]].head(5).to_dict(orient="records"),
        "note": "compat_risk_score 仅用于排序，不是因果分数。",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {json_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="LoRA 任务兼容性扫描（仅前向，无训练）")
    parser.add_argument("--config", type=str, required=True, help="run_smoke 配置文件")
    parser.add_argument("--output-dir", type=str, default="outputs/compat_scan", help="扫描输出目录")
    args = parser.parse_args()

    compat_scan(config_path=args.config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
