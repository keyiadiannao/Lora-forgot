import argparse
import json
import math
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PREDICTORS = [
    "gradient_alignment",
    "fisher_overlap",
    "activation_spectrum_overlap",
    "svcca_overlap",
    "linear_cka_overlap",
    "c_couple",
    "spectrum_layers_mean",
    "spectrum_layers_std",
    "spectrum_layers_span",
    "spectrum_layers_max",
    "spectrum_layers_min",
]


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return obj


def safe_corr(df: pd.DataFrame, x: str, y: str) -> float:
    if x not in df.columns or y not in df.columns:
        return float("nan")
    a = pd.to_numeric(df[x], errors="coerce")
    b = pd.to_numeric(df[y], errors="coerce")
    m = (~a.isna()) & (~b.isna())
    if int(m.sum()) < 3:
        return float("nan")
    a = a[m]
    b = b[m]
    if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
        return float("nan")
    return float(a.corr(b))


def corr_block(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in PREDICTORS:
        out[p] = safe_corr(df, p, "forgetting")
    return out


def strategy_compare(pareto: pd.DataFrame, key_df: pd.DataFrame) -> List[Dict[str, Any]]:
    merged = pareto.merge(key_df, on=["seed", "pair"], how="inner")
    piv_f = merged.pivot_table(index=["seed", "pair"], columns="strategy", values="forgetting", aggfunc="mean")
    piv_g = merged.pivot_table(index=["seed", "pair"], columns="strategy", values="new_task_gain", aggfunc="mean")
    if "vanilla_lora" not in piv_f.columns or "vanilla_lora" not in piv_g.columns:
        return []

    rows = []
    for s in sorted(set(piv_f.columns) & set(piv_g.columns)):
        if s == "vanilla_lora":
            continue
        d_f = piv_f[s] - piv_f["vanilla_lora"]
        d_g = piv_g[s] - piv_g["vanilla_lora"]
        valid = (~d_f.isna()) & (~d_g.isna())
        if int(valid.sum()) == 0:
            continue
        vf = d_f[valid]
        vg = d_g[valid]
        noninferior = ((vf <= 0) & (vg >= 0)).mean()
        rows.append(
            {
                "strategy": s,
                "n": int(valid.sum()),
                "mean_delta_forgetting": float(vf.mean()),
                "mean_delta_gain": float(vg.mean()),
                "noninferior_rate": float(noninferior),
            }
        )
    rows.sort(key=lambda r: (r["mean_delta_forgetting"], -r["mean_delta_gain"]))
    return rows


def per_seed_corr(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in PREDICTORS:
        vals = []
        for _, g in df.groupby("seed"):
            r = safe_corr(g, p, "forgetting")
            vals.append(r)
        s = pd.Series(vals, dtype=float)
        valid = s.dropna()
        out[p] = {
            "values": [None if pd.isna(x) else float(x) for x in s.tolist()],
            "mean": None if len(valid) == 0 else float(valid.mean()),
            "std": None if len(valid) <= 1 else float(valid.std()),
            "sign_consistency": None
            if len(valid) == 0
            else float(max((valid > 0).mean(), (valid < 0).mean())),
        }
    return out


def make_report(input_dir: str, output_path: str) -> Dict[str, Any]:
    pair_csv = os.path.join(input_dir, "multiseed_pair_metrics.csv")
    pareto_csv = os.path.join(input_dir, "multiseed_pareto.csv")
    pair_df = pd.read_csv(pair_csv)
    pareto_df = pd.read_csv(pareto_csv)

    if "seed" not in pair_df.columns:
        pair_df["seed"] = 0
    if "seed" not in pareto_df.columns:
        pareto_df["seed"] = 0

    all_corr = corr_block(pair_df)
    pos_keys = pair_df[pair_df["forgetting"] > 0][["seed", "pair"]].drop_duplicates()
    neg_keys = pair_df[pair_df["forgetting"] <= 0][["seed", "pair"]].drop_duplicates()

    subset_pos = pair_df.merge(pos_keys, on=["seed", "pair"], how="inner")
    subset_neg = pair_df.merge(neg_keys, on=["seed", "pair"], how="inner")

    result = {
        "input_dir": os.path.abspath(input_dir),
        "n_all": int(len(pair_df)),
        "n_pos": int(len(subset_pos)),
        "n_neg": int(len(subset_neg)),
        "corr_all_vs_forgetting": all_corr,
        "corr_pos_vs_forgetting": corr_block(subset_pos),
        "corr_neg_vs_forgetting": corr_block(subset_neg),
        "per_seed_corr": per_seed_corr(pair_df),
        "strategy_compare_all": strategy_compare(pareto_df, pair_df[["seed", "pair"]].drop_duplicates()),
        "strategy_compare_pos": strategy_compare(pareto_df, pos_keys),
        "strategy_compare_neg": strategy_compare(pareto_df, neg_keys),
    }

    tips = []
    r_std = result["corr_all_vs_forgetting"].get("spectrum_layers_std")
    r_span = result["corr_all_vs_forgetting"].get("spectrum_layers_span")
    if pd.notna(r_std) and abs(float(r_std)) >= 0.4:
        tips.append(f"spectrum_layers_std 与遗忘相关较强 (r={float(r_std):.3f})，值得继续按层谱方向。")
    if pd.notna(r_span) and abs(float(r_span)) >= 0.35:
        tips.append(f"spectrum_layers_span 与遗忘有中等相关 (r={float(r_span):.3f})，可做跨层异质性机制验证。")

    pos_cmp = result["strategy_compare_pos"]
    winners = [
        r["strategy"]
        for r in pos_cmp
        if r["mean_delta_forgetting"] < 0 and r["mean_delta_gain"] >= 0 and r["noninferior_rate"] >= 0.4
    ]
    if winners:
        tips.append("正遗忘子集上较稳的候选策略: " + ", ".join(winners))
    else:
        tips.append("正遗忘子集上暂无稳定优于 vanilla 的策略，建议继续增加 seed 验证。")

    result["decision_tips_zh"] = tips

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(result), f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="聚焦稳定性验证报告")
    parser.add_argument("--input-dir", type=str, required=True, help="包含 multiseed_pair_metrics.csv 的目录")
    parser.add_argument("--output", type=str, default="", help="默认写入 input-dir/focus_report.json")
    args = parser.parse_args()

    out = args.output or os.path.join(args.input_dir, "focus_report.json")
    make_report(args.input_dir, out)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
