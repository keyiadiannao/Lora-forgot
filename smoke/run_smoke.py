import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import DownloadConfig, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TaskSpec:
    dataset: str
    subset: Optional[str]
    train_split: str
    eval_split: str
    input_key: str
    target_key: str


class TextCausalDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_len: int):
        self.samples = []
        for t in texts:
            toks = tokenizer(
                t,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = toks["input_ids"][0]
            attention_mask = toks["attention_mask"][0]
            labels = input_ids.clone()
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [x["input_ids"] for x in batch]
        attention_mask = [x["attention_mask"] for x in batch]
        labels = [x["labels"] for x in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fallback_text_bank(task: TaskSpec, n: int) -> List[str]:
    # 网络不可用或证书失败时的本地保底样本，保证烟雾测试可继续
    lower_name = f"{task.dataset}:{task.subset}".lower()
    if "gsm8k" in lower_name or "math" in lower_name:
        base = [
            "If Alice has 12 apples and gives away 5, how many remain?",
            "A train moves 60 km in 1.5 hours, what is the average speed?",
            "Compute 17 * 23 and explain each step.",
            "Solve for x: 3x + 5 = 20.",
        ]
    elif "mmlu" in lower_name or "physics" in lower_name:
        base = [
            "What is Newton's second law and how is it applied?",
            "Describe conservation of momentum in collisions.",
            "What does Ohm's law state for electrical circuits?",
            "Why does increasing mass increase inertia?",
        ]
    elif "winograd" in lower_name:
        base = [
            "Tom thanked Jim because he helped fix the issue. Who helped?",
            "The trophy does not fit in the suitcase because it is too big. What is big?",
            "Sarah called Emma while she was driving. Who was driving?",
            "The city council denied the permit because they feared violence. Who feared violence?",
        ]
    else:
        base = [
            "Explain the main idea of the paragraph.",
            "Summarize the argument in one sentence.",
            "Identify cause and effect in this statement.",
            "Rewrite the sentence in a formal tone.",
        ]

    out = []
    for i in range(n):
        sample = base[i % len(base)]
        out.append(f"Q: {sample}\nA: fallback_answer_{i % 7}")
    return out


def sample_task_texts(
    task: TaskSpec,
    train_n: int,
    eval_n: int,
    local_files_only: bool = False,
    mirror_endpoint: Optional[str] = None,
    allow_fallback: bool = True,
) -> Tuple[List[str], List[str]]:
    def _apply_mirror_endpoint(endpoint: str) -> None:
        # 既设置环境变量，也尽量覆盖运行时常量，避免库初始化后仍指向 huggingface.co
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HF_HUB_ENDPOINT"] = endpoint
        os.environ["HUGGINGFACE_HUB_BASE_URL"] = endpoint
        try:
            import datasets.config as ds_cfg  # type: ignore

            ds_cfg.HF_ENDPOINT = endpoint
        except Exception:
            pass
        try:
            from huggingface_hub import constants as hf_const  # type: ignore

            hf_const.ENDPOINT = endpoint
        except Exception:
            pass

    def _load_once() -> Tuple[List[str], List[str]]:
        dl_cfg = DownloadConfig(local_files_only=local_files_only)
        if task.subset in (None, "", "null", "None"):
            ds = load_dataset(task.dataset, download_config=dl_cfg)
        else:
            ds = load_dataset(task.dataset, task.subset, download_config=dl_cfg)
        train_split = ds[task.train_split]
        eval_split = ds[task.eval_split]

        tn = min(train_n, len(train_split))
        en = min(eval_n, len(eval_split))
        train_rows = train_split.shuffle(seed=42).select(range(tn))
        eval_rows = eval_split.shuffle(seed=42).select(range(en))

        train_texts = [f"Q: {str(row.get(task.input_key, ''))}\nA: {str(row.get(task.target_key, ''))}" for row in train_rows]
        eval_texts = [f"Q: {str(row.get(task.input_key, ''))}\nA: {str(row.get(task.target_key, ''))}" for row in eval_rows]
        return train_texts, eval_texts

    try:
        return _load_once()
    except Exception as e:
        # 在线模式先尝试镜像重试一次；离线模式直接按 allow_fallback 走
        if (not local_files_only) and mirror_endpoint:
            print(f"[WARN] load_dataset 首次失败，尝试镜像重试: {mirror_endpoint} ({type(e).__name__})")
            _apply_mirror_endpoint(mirror_endpoint)
            try:
                return _load_once()
            except Exception as e2:
                e = e2

        if not allow_fallback:
            raise RuntimeError(
                f"load_dataset 失败且已禁用 fallback: {task.dataset}/{task.subset} ({type(e).__name__})"
            ) from e

        print(
            f"[WARN] load_dataset 失败 ({task.dataset}/{task.subset})，"
            f"已切换 fallback 样本。错误: {type(e).__name__}"
        )
        return _fallback_text_bank(task, train_n), _fallback_text_bank(task, eval_n)


def tfidf_matrix(texts_a: List[str], texts_b: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    vec = TfidfVectorizer(max_features=8192, ngram_range=(1, 2))
    mat = vec.fit_transform(texts_a + texts_b)
    a = mat[: len(texts_a)].astype(np.float32)
    b = mat[len(texts_a) :].astype(np.float32)
    return a, b


def cosine(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    denom = np.linalg.norm(x) * np.linalg.norm(y) + eps
    return float(np.dot(x, y) / denom)


def metric_gradient_alignment_proxy(texts_a: List[str], texts_b: List[str]) -> float:
    # 代理梯度：用 TF-IDF 平均方向近似任务更新方向
    a, b = tfidf_matrix(texts_a, texts_b)
    grad_a = np.asarray(a.mean(axis=0)).ravel()
    grad_b = np.asarray(b.mean(axis=0)).ravel()
    return cosine(grad_a, grad_b)


def metric_fisher_overlap_proxy(texts_a: List[str], texts_b: List[str]) -> float:
    # 代理 Fisher：用二阶强度近似（特征平方均值）
    a, b = tfidf_matrix(texts_a, texts_b)
    fisher_a = np.asarray(a.power(2).mean(axis=0)).ravel()
    fisher_b = np.asarray(b.power(2).mean(axis=0)).ravel()
    return cosine(fisher_a, fisher_b)


def metric_activation_spectrum_overlap(texts_a: List[str], texts_b: List[str]) -> float:
    a, b = tfidf_matrix(texts_a, texts_b)
    # 使用首主方向（特征空间向量）比较，避免样本数不一致造成维度冲突
    svd_a = TruncatedSVD(n_components=1, random_state=42)
    svd_b = TruncatedSVD(n_components=1, random_state=42)
    va = svd_a.fit(a).components_[0]
    vb = svd_b.fit(b).components_[0]
    return cosine(va, vb)


def proxy_forgetting_from_couple(c_couple: float, noise_scale: float = 0.05) -> float:
    # 仅用于烟雾测试链路，正式实验应替换为真实 sequential SFT 结果
    noise = np.random.normal(loc=0.0, scale=noise_scale)
    forgetting = -0.35 * c_couple + noise
    return float(np.clip(forgetting, -1.0, 0.2))


def strategy_points(forgetting: float) -> Dict[str, Tuple[float, float]]:
    # 输出 (old_task_retention, new_task_gain)，用于 Pareto 可视化
    base_retention = 1.0 + forgetting
    points = {
        "vanilla_lora": (max(0.0, base_retention), 0.10),
        "random_freeze_30": (max(0.0, base_retention + 0.03), 0.07),
        "lower_layers_freeze_30": (max(0.0, base_retention + 0.04), 0.06),
        "c_couple_freeze_30": (max(0.0, base_retention + 0.07), 0.09),
        "spectrum_freeze_30": (max(0.0, base_retention + 0.065), 0.085),
        "spectrum_extreme_freeze_30": (max(0.0, base_retention + 0.068), 0.086),
    }
    return points


def _model_target_modules(model, configured_targets: List[str]) -> List[str]:
    names = [name for name, _ in model.named_modules()]
    if configured_targets:
        hit = []
        for t in configured_targets:
            if any(t in n for n in names):
                hit.append(t)
        if hit:
            return sorted(list(set(hit)))

    fallback_candidates = ["q_proj", "v_proj", "k_proj", "o_proj", "c_attn", "c_proj", "query_key_value"]
    hit = []
    for c in fallback_candidates:
        if any(c in n for n in names):
            hit.append(c)
    if not hit:
        raise ValueError("无法自动推断 LoRA target_modules，请在配置中手动指定。")
    return sorted(list(set(hit)))


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype_from_cfg(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(str(dtype_name).lower(), torch.float16)


def load_lora_model_and_tokenizer(config: Dict):
    model_name = config["model"]["name"]
    model_ref = config["model"].get("local_path", model_name)
    dtype = _dtype_from_cfg(config["model"].get("torch_dtype", "float16"))
    device = _device()
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

    targets = _model_target_modules(model, config["lora"].get("target_modules", []))
    lora_cfg = LoraConfig(
        r=int(config["lora"]["rank"]),
        lora_alpha=int(config["lora"]["alpha"]),
        lora_dropout=float(config["lora"]["dropout"]),
        target_modules=targets,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer, device


def make_dataloader(texts: List[str], tokenizer, max_len: int, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TextCausalDataset(texts, tokenizer, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=Collator(tokenizer))


def train_texts(model, tokenizer, device, texts: List[str], train_cfg: Dict) -> Dict[str, float]:
    if len(texts) == 0:
        return {"avg_loss": float("nan"), "steps": 0}
    model.train()
    lr = float(train_cfg["learning_rate"])
    epochs = int(train_cfg["epochs"])
    batch_size = int(train_cfg["batch_size"])
    grad_acc = int(train_cfg["grad_accum_steps"])
    max_len = int(train_cfg["max_seq_len"])
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))

    loader = make_dataloader(texts, tokenizer, max_len=max_len, batch_size=batch_size, shuffle=True)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    optimizer.zero_grad(set_to_none=True)
    step = 0
    loss_sum = 0.0
    loss_count = 0
    skipped_nonfinite = 0
    skipped_oom = 0
    for _ in range(epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            try:
                out = model(**batch)
                raw_loss = out.loss
                if not torch.isfinite(raw_loss):
                    skipped_nonfinite += 1
                    model.zero_grad(set_to_none=True)
                    continue
                loss = raw_loss / grad_acc
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    skipped_oom += 1
                    model.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise
            step += 1
            loss_sum += float(raw_loss.detach().cpu().item())
            loss_count += 1
            if step % grad_acc == 0:
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                        max_norm=max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    if step % grad_acc != 0:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                max_norm=max_grad_norm,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    avg_loss = float(loss_sum / max(1, loss_count))
    return {
        "avg_loss": avg_loss,
        "steps": int(step),
        "skipped_nonfinite": int(skipped_nonfinite),
        "skipped_oom": int(skipped_oom),
    }


@torch.no_grad()
def eval_loss(model, tokenizer, device, texts: List[str], train_cfg: Dict) -> float:
    if len(texts) == 0:
        return float("nan")
    model.eval()
    loader = make_dataloader(
        texts,
        tokenizer,
        max_len=int(train_cfg["max_seq_len"]),
        batch_size=max(1, int(train_cfg["batch_size"])),
        shuffle=False,
    )
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        lv = out.loss.detach().cpu()
        if torch.isfinite(lv):
            losses.append(float(lv.item()))
    return float(np.mean(losses)) if losses else float("nan")


def _flatten_grad_vector(model) -> np.ndarray:
    chunks = []
    for _, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            chunks.append(p.grad.detach().float().cpu().reshape(-1))
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    return torch.cat(chunks).numpy()


def lora_param_l2_norm(model) -> float:
    chunks = []
    for name, p in model.named_parameters():
        if "lora_" in name:
            chunks.append(p.detach().float().reshape(-1).cpu())
    if not chunks:
        return 0.0
    v = torch.cat(chunks)
    return float(torch.norm(v, p=2).item())


def _module_trainable_param_keys(model) -> List[str]:
    keys = []
    for name, p in model.named_parameters():
        if p.requires_grad and ("lora_" in name):
            keys.append(name)
    return keys


def _module_prefix_from_param_name(param_name: str) -> str:
    if ".lora_" in param_name:
        return param_name.split(".lora_")[0]
    return param_name


def gradient_metrics_real(
    model,
    tokenizer,
    device,
    texts_a: List[str],
    texts_b: List[str],
    train_cfg: Dict,
) -> Tuple[float, float, Dict[str, float]]:
    max_len = int(train_cfg["max_seq_len"])
    batch_size = max(1, int(train_cfg["batch_size"]))
    loader_a = make_dataloader(texts_a, tokenizer, max_len=max_len, batch_size=batch_size, shuffle=False)
    loader_b = make_dataloader(texts_b, tokenizer, max_len=max_len, batch_size=batch_size, shuffle=False)

    batch_a = next(iter(loader_a))
    batch_b = next(iter(loader_b))
    batch_a = {k: v.to(device) for k, v in batch_a.items()}
    batch_b = {k: v.to(device) for k, v in batch_b.items()}

    model.train()
    model.zero_grad(set_to_none=True)
    out_a = model(**batch_a)
    out_a.loss.backward()
    grad_a = _flatten_grad_vector(model)

    grads_per_param_a = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and "lora_" in name:
            grads_per_param_a[name] = p.grad.detach().float().cpu().reshape(-1).numpy()

    fisher_a = grad_a**2

    model.zero_grad(set_to_none=True)
    out_b = model(**batch_b)
    out_b.loss.backward()
    grad_b = _flatten_grad_vector(model)

    grads_per_param_b = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None and "lora_" in name:
            grads_per_param_b[name] = p.grad.detach().float().cpu().reshape(-1).numpy()

    fisher_b = grad_b**2
    model.zero_grad(set_to_none=True)

    layer_scores = {}
    all_keys = sorted(set(grads_per_param_a.keys()) & set(grads_per_param_b.keys()))
    for k in all_keys:
        prefix = _module_prefix_from_param_name(k)
        layer_scores.setdefault(prefix, [])
        layer_scores[prefix].append(cosine(grads_per_param_a[k], grads_per_param_b[k]))
    layer_scores = {k: float(np.mean(v)) for k, v in layer_scores.items()}

    grad_align = cosine(grad_a, grad_b)
    fisher_overlap = cosine(fisher_a, fisher_b)
    return grad_align, fisher_overlap, layer_scores


@torch.no_grad()
def _collect_last_hidden_repr(model, tokenizer, device, texts: List[str], train_cfg: Dict) -> np.ndarray:
    max_len = int(train_cfg["max_seq_len"])
    batch_size = max(1, int(train_cfg["batch_size"]))
    loader = make_dataloader(texts, tokenizer, max_len=max_len, batch_size=batch_size, shuffle=False)
    mats = []
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        hs = out.hidden_states[-1]  # [B, T, H]
        pooled = hs.mean(dim=1).float().cpu().numpy()  # [B, H]
        mats.append(pooled)
        if sum(x.shape[0] for x in mats) >= 32:
            break
    if not mats:
        return np.zeros((1, 8), dtype=np.float32)
    arr = np.concatenate(mats, axis=0)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _align_rows(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(int(a.shape[0]), int(b.shape[0]))
    if n <= 0:
        return np.zeros((1, a.shape[1] if a.ndim == 2 else 8), dtype=np.float32), np.zeros(
            (1, b.shape[1] if b.ndim == 2 else 8), dtype=np.float32
        )
    return a[:n], b[:n]


def _invsqrtm_psd(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, eps, None)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return inv_sqrt


def activation_overlap_real(model, tokenizer, device, texts_a: List[str], texts_b: List[str], train_cfg: Dict) -> float:
    a = _collect_last_hidden_repr(model, tokenizer, device, texts_a, train_cfg)
    b = _collect_last_hidden_repr(model, tokenizer, device, texts_b, train_cfg)
    if a.shape[1] < 2 or b.shape[1] < 2:
        return float("nan")
    svd_a = TruncatedSVD(n_components=1, random_state=42)
    svd_b = TruncatedSVD(n_components=1, random_state=42)
    va = svd_a.fit(a).components_[0]
    vb = svd_b.fit(b).components_[0]
    return cosine(va, vb)


def linear_cka_overlap_real(model, tokenizer, device, texts_a: List[str], texts_b: List[str], train_cfg: Dict) -> float:
    a = _collect_last_hidden_repr(model, tokenizer, device, texts_a, train_cfg)
    b = _collect_last_hidden_repr(model, tokenizer, device, texts_b, train_cfg)
    a, b = _align_rows(a, b)
    if a.shape[0] < 2 or b.shape[0] < 2:
        return float("nan")
    x = a - np.mean(a, axis=0, keepdims=True)
    y = b - np.mean(b, axis=0, keepdims=True)
    xty = x.T @ y
    hsic = float(np.sum(xty * xty))
    n_x = float(np.linalg.norm(x.T @ x, ord="fro"))
    n_y = float(np.linalg.norm(y.T @ y, ord="fro"))
    if n_x <= 1e-12 or n_y <= 1e-12:
        return float("nan")
    return float(hsic / (n_x * n_y + 1e-12))


def svcca_overlap_real(
    model,
    tokenizer,
    device,
    texts_a: List[str],
    texts_b: List[str],
    train_cfg: Dict,
    max_components: int = 20,
) -> float:
    a = _collect_last_hidden_repr(model, tokenizer, device, texts_a, train_cfg)
    b = _collect_last_hidden_repr(model, tokenizer, device, texts_b, train_cfg)
    a, b = _align_rows(a, b)
    if a.shape[0] < 3 or b.shape[0] < 3 or a.shape[1] < 2 or b.shape[1] < 2:
        return float("nan")

    a0 = a - np.mean(a, axis=0, keepdims=True)
    b0 = b - np.mean(b, axis=0, keepdims=True)
    k = int(min(max_components, a0.shape[0] - 1, b0.shape[0] - 1, a0.shape[1], b0.shape[1]))
    if k < 1:
        return float("nan")

    z_a = TruncatedSVD(n_components=k, random_state=42).fit_transform(a0)
    z_b = TruncatedSVD(n_components=k, random_state=42).fit_transform(b0)
    z_a = z_a - np.mean(z_a, axis=0, keepdims=True)
    z_b = z_b - np.mean(z_b, axis=0, keepdims=True)

    n = float(max(z_a.shape[0] - 1, 1))
    c_aa = (z_a.T @ z_a) / n
    c_bb = (z_b.T @ z_b) / n
    c_ab = (z_a.T @ z_b) / n
    try:
        inv_aa = _invsqrtm_psd(c_aa)
        inv_bb = _invsqrtm_psd(c_bb)
        m = inv_aa @ c_ab @ inv_bb
        svals = np.linalg.svd(m, compute_uv=False)
        svals = np.clip(svals, 0.0, 1.0)
        return float(np.mean(svals))
    except Exception:
        return float("nan")


def _layer_idx_from_lora_prefix(prefix: str) -> Optional[int]:
    # PEFT 下常见前缀含 base_model.model...，不能用 ^model.layers 锚死
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
    if m:
        return int(m.group(1))
    m = re.search(r"\.h\.(\d+)\.", prefix)
    if m:
        return int(m.group(1))
    return None


def spectrum_layer_geometry_stats(per_layer: Dict[int, float]) -> Dict[str, float]:
    """按层 activation 谱余弦（与 activation_spectrum_layer_scores_real 中 per_layer 一致）的汇总，用于跨层几何分析。"""
    vals = [float(v) for v in per_layer.values() if np.isfinite(v)]
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


@torch.no_grad()
def activation_spectrum_layer_scores_real(
    model,
    tokenizer,
    device,
    texts_a: List[str],
    texts_b: List[str],
    train_cfg: Dict,
    layer_score_prefixes: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[int, float]]:
    """
    各 decoder 层：对 A/B 文本在该层 hidden 上做 token-mean 池化后，分别 TruncatedSVD 取首主方向，
    再算两方向余弦（与全局 activation_overlap_real 同构，但是按层）。
    同一物理层的所有 LoRA prefix 共享该分数，供 spectrum_freeze_30 与 c_couple 一样做 top-ratio 冻结。
    """
    if not layer_score_prefixes:
        return {}, {}
    layer_ids = sorted({_layer_idx_from_lora_prefix(p) for p in layer_score_prefixes.keys() if _layer_idx_from_lora_prefix(p) is not None})
    if not layer_ids:
        return {p: float("-inf") for p in layer_score_prefixes}, {}

    max_len = int(train_cfg["max_seq_len"])
    batch_size = max(1, int(train_cfg["batch_size"]))
    loader_a = make_dataloader(texts_a, tokenizer, max_len=max_len, batch_size=batch_size, shuffle=False)
    loader_b = make_dataloader(texts_b, tokenizer, max_len=max_len, batch_size=batch_size, shuffle=False)

    def collect_for_texts(loader) -> Dict[int, np.ndarray]:
        chunks: Dict[int, List[np.ndarray]] = {lid: [] for lid in layer_ids}
        rows = 0
        model.eval()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
            )
            hs = out.hidden_states
            bsz = int(batch["input_ids"].shape[0])
            rows += bsz
            for lid in layer_ids:
                idx = lid + 1
                if idx >= len(hs):
                    idx = len(hs) - 1
                h = hs[idx]
                pooled = h.mean(dim=1).float().cpu().numpy()
                pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
                chunks[lid].append(pooled)
            if rows >= 32:
                break
        out_mats = {}
        for lid in layer_ids:
            if not chunks[lid]:
                out_mats[lid] = np.zeros((1, 8), dtype=np.float32)
            else:
                out_mats[lid] = np.concatenate(chunks[lid], axis=0)
        return out_mats

    mats_a = collect_for_texts(loader_a)
    mats_b = collect_for_texts(loader_b)

    per_layer: Dict[int, float] = {}
    for lid in layer_ids:
        a = mats_a[lid]
        b = mats_b[lid]
        if a.shape[1] < 2 or b.shape[1] < 2 or a.shape[0] < 1 or b.shape[0] < 1:
            per_layer[lid] = float("-inf")
            continue
        # 小样本时 SVD 不稳定：至少 2 行用首主方向；否则用 batch 均值向量的 cosine
        if a.shape[0] < 2 or b.shape[0] < 2:
            ma = np.mean(a, axis=0)
            mb = np.mean(b, axis=0)
            per_layer[lid] = cosine(ma, mb)
            continue
        svd_a = TruncatedSVD(n_components=1, random_state=42)
        svd_b = TruncatedSVD(n_components=1, random_state=42)
        va = svd_a.fit(a).components_[0]
        vb = svd_b.fit(b).components_[0]
        per_layer[lid] = cosine(va, vb)

    scores: Dict[str, float] = {}
    for prefix in layer_score_prefixes:
        lid = _layer_idx_from_lora_prefix(prefix)
        if lid is None:
            scores[prefix] = float("-inf")
        else:
            scores[prefix] = float(per_layer.get(lid, float("-inf")))
    return scores, per_layer


def top_k_modules(layer_scores: Dict[str, float], ratio: float, largest: bool) -> List[str]:
    if not layer_scores:
        return []
    k = max(1, int(len(layer_scores) * ratio))
    items = sorted(layer_scores.items(), key=lambda x: x[1], reverse=largest)
    return [x[0] for x in items[:k]]


def spectrum_extreme_scores(spectrum_scores: Dict[str, float]) -> Dict[str, float]:
    """模块谱分数到均值的偏离程度，数值越大表示越“极端”。"""
    finite_vals = [float(v) for v in spectrum_scores.values() if np.isfinite(v)]
    if not finite_vals:
        return {k: float("-inf") for k in spectrum_scores}
    m = float(np.mean(finite_vals))
    out: Dict[str, float] = {}
    for k, v in spectrum_scores.items():
        if not np.isfinite(v):
            out[k] = float("-inf")
        else:
            out[k] = float(abs(float(v) - m))
    return out


def freeze_modules_for_strategy(
    model,
    strategy: str,
    layer_scores: Dict[str, float],
    ratio: float = 0.3,
    seed: int = 42,
    spectrum_scores: Optional[Dict[str, float]] = None,
    spectrum_extreme: Optional[Dict[str, float]] = None,
) -> None:
    prefixes = sorted(layer_scores.keys())
    if not prefixes:
        return

    if strategy == "vanilla_lora":
        to_freeze = []
    elif strategy == "random_freeze_30":
        rng = random.Random(seed)
        k = max(1, int(len(prefixes) * ratio))
        to_freeze = rng.sample(prefixes, k)
    elif strategy == "lower_layers_freeze_30":
        k = max(1, int(len(prefixes) * ratio))
        to_freeze = prefixes[:k]
    elif strategy == "c_couple_freeze_30":
        to_freeze = top_k_modules(layer_scores, ratio=ratio, largest=True)
    elif strategy == "spectrum_freeze_30":
        if spectrum_scores is None:
            raise ValueError("spectrum_freeze_30 需要 spectrum_scores（按层 activation 谱重叠）")
        to_freeze = top_k_modules(spectrum_scores, ratio=ratio, largest=True)
    elif strategy == "spectrum_extreme_freeze_30":
        if spectrum_extreme is None:
            raise ValueError("spectrum_extreme_freeze_30 需要 spectrum_extreme 分数")
        to_freeze = top_k_modules(spectrum_extreme, ratio=ratio, largest=True)
    else:
        raise ValueError(f"未知策略: {strategy}")

    freeze_set = set(to_freeze)
    for name, p in model.named_parameters():
        if "lora_" not in name:
            continue
        prefix = _module_prefix_from_param_name(name)
        p.requires_grad = prefix not in freeze_set


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a / (b + eps))


def pearson_safe(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    if x_col not in df.columns or y_col not in df.columns:
        return float("nan")
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    valid = (~x.isna()) & (~y.isna())
    if int(valid.sum()) < 3:
        return float("nan")
    x = x[valid]
    y = y[valid]
    if float(x.std()) < 1e-12 or float(y.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def weighted_c_couple(metric_values: Dict[str, float], weights: Dict[str, float]) -> float:
    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("c_couple_weights 的总和必须大于 0")
    score = 0.0
    for key, weight in weights.items():
        score += weight * metric_values[key]
    return float(score / total_w)


def parse_task(task_cfg: Dict) -> TaskSpec:
    return TaskSpec(
        dataset=task_cfg["dataset"],
        subset=task_cfg.get("subset", None),
        train_split=task_cfg["train_split"],
        eval_split=task_cfg["eval_split"],
        input_key=task_cfg["input_key"],
        target_key=task_cfg["target_key"],
    )


def run_proxy_smoke(config: Dict, output_dir: str) -> None:
    train_n = int(config["train"]["max_train_samples"])
    eval_n = int(config["train"]["max_eval_samples"])
    weights = config["metrics"]["c_couple_weights"]
    data_local_only = bool(config.get("data", {}).get("local_files_only", False))
    data_mirror = config.get("data", {}).get("hf_endpoint", None)
    allow_fallback = bool(config.get("data", {}).get("allow_fallback", True))

    rows = []
    pareto_rows = []
    for pair in config["tasks"]:
        pair_name = pair["name"]
        task_a = parse_task(pair["task_a"])
        task_b = parse_task(pair["task_b"])

        a_train, _ = sample_task_texts(
            task_a,
            train_n,
            eval_n,
            local_files_only=data_local_only,
            mirror_endpoint=data_mirror,
            allow_fallback=allow_fallback,
        )
        b_train, _ = sample_task_texts(
            task_b,
            train_n,
            eval_n,
            local_files_only=data_local_only,
            mirror_endpoint=data_mirror,
            allow_fallback=allow_fallback,
        )

        metric_values = {
            "gradient_alignment": metric_gradient_alignment_proxy(a_train, b_train),
            "fisher_overlap": metric_fisher_overlap_proxy(a_train, b_train),
            "activation_spectrum_overlap": metric_activation_spectrum_overlap(a_train, b_train),
        }
        c_couple = weighted_c_couple(metric_values, weights)
        forgetting = proxy_forgetting_from_couple(c_couple)
        points = strategy_points(forgetting)

        row = {
            "pair": pair_name,
            **metric_values,
            "c_couple": c_couple,
            "forgetting": forgetting,
        }
        rows.append(row)

        for strategy, (retention, gain) in points.items():
            pareto_rows.append(
                {
                    "pair": pair_name,
                    "strategy": strategy,
                    "old_task_retention": retention,
                    "new_task_gain": gain,
                }
            )

    metrics_df = pd.DataFrame(rows)
    pareto_df = pd.DataFrame(pareto_rows)

    metrics_path = os.path.join(output_dir, "pair_metrics.csv")
    pareto_path = os.path.join(output_dir, "pareto_points.csv")
    metrics_df.to_csv(metrics_path, index=False)
    pareto_df.to_csv(pareto_path, index=False)

    corr = metrics_df[["gradient_alignment", "fisher_overlap", "activation_spectrum_overlap", "c_couple", "forgetting"]].corr(
        numeric_only=True
    )
    summary = {
        "rows": len(metrics_df),
        "pearson_r": {
            "gradient_alignment_vs_forgetting": float(corr.loc["gradient_alignment", "forgetting"]),
            "fisher_overlap_vs_forgetting": float(corr.loc["fisher_overlap", "forgetting"]),
            "activation_spectrum_overlap_vs_forgetting": float(corr.loc["activation_spectrum_overlap", "forgetting"]),
            "c_couple_vs_forgetting": float(corr.loc["c_couple", "forgetting"]),
        },
        "note": "当前为 proxy smoke run：用于验证工程流程和统计链路，不代表真实 SFT 效果。",
    }

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {metrics_path}")
    print(f"[OK] Saved: {pareto_path}")
    print(f"[OK] Saved: {os.path.join(output_dir, 'summary.json')}")


def run_real_smoke(config: Dict, output_dir: str) -> None:
    train_n = int(config["train"]["max_train_samples"])
    eval_n = int(config["train"]["max_eval_samples"])
    weights = config["metrics"]["c_couple_weights"]
    strategies = config["strategies"]
    data_local_only = bool(config.get("data", {}).get("local_files_only", False))
    data_mirror = config.get("data", {}).get("hf_endpoint", None)
    allow_fallback = bool(config.get("data", {}).get("allow_fallback", True))

    model, tokenizer, device = load_lora_model_and_tokenizer(config)
    print(f"[INFO] device={device}")

    pair_rows = []
    pareto_rows = []
    for pair in config["tasks"]:
        pair_name = pair["name"]
        print(f"[INFO] running pair={pair_name}")
        task_a = parse_task(pair["task_a"])
        task_b = parse_task(pair["task_b"])

        a_train, a_eval = sample_task_texts(
            task_a,
            train_n,
            eval_n,
            local_files_only=data_local_only,
            mirror_endpoint=data_mirror,
            allow_fallback=allow_fallback,
        )
        b_train, b_eval = sample_task_texts(
            task_b,
            train_n,
            eval_n,
            local_files_only=data_local_only,
            mirror_endpoint=data_mirror,
            allow_fallback=allow_fallback,
        )

        # A 阶段微调（含前后诊断）
        a_loss_before_a = eval_loss(model, tokenizer, device, a_eval, config["train"])
        lora_norm_before_a = lora_param_l2_norm(model)
        a_train_stat = train_texts(model, tokenizer, device, a_train, config["train"])
        a_loss_after_a = eval_loss(model, tokenizer, device, a_eval, config["train"])
        lora_norm_after_a = lora_param_l2_norm(model)
        b_loss_before = eval_loss(model, tokenizer, device, b_eval, config["train"])
        print(
            f"[DIAG] {pair_name} A-train steps={a_train_stat['steps']} "
            f"avg_train_loss={a_train_stat['avg_loss']:.6f} "
            f"skipped_nonfinite={a_train_stat.get('skipped_nonfinite', 0)} "
            f"skipped_oom={a_train_stat.get('skipped_oom', 0)} "
            f"A_eval_before={a_loss_before_a:.6f} A_eval_after={a_loss_after_a:.6f} "
            f"delta_A={a_loss_after_a - a_loss_before_a:.6f} "
            f"lora_l2_delta={lora_norm_after_a - lora_norm_before_a:.6f}"
        )

        grad_align, fisher_overlap, layer_scores = gradient_metrics_real(
            model, tokenizer, device, a_train, b_train, config["train"]
        )
        act_overlap = activation_overlap_real(model, tokenizer, device, a_train, b_train, config["train"])
        svcca_overlap = svcca_overlap_real(model, tokenizer, device, a_train, b_train, config["train"])
        cka_overlap = linear_cka_overlap_real(model, tokenizer, device, a_train, b_train, config["train"])
        spectrum_scores, spectrum_per_layer = activation_spectrum_layer_scores_real(
            model, tokenizer, device, a_train, b_train, config["train"], layer_scores
        )
        spectrum_extreme = spectrum_extreme_scores(spectrum_scores)
        spectrum_geo = spectrum_layer_geometry_stats(spectrum_per_layer)
        # 少数架构/样本下按层前缀为空：用最后一层标量谱重叠回填均值，避免 CSV 全空
        if float(spectrum_geo.get("spectrum_n_layers", 0) or 0) == 0 and np.isfinite(act_overlap):
            ao = float(act_overlap)
            spectrum_geo["spectrum_n_layers"] = 1.0
            spectrum_geo["spectrum_layers_mean"] = ao
            spectrum_geo["spectrum_layers_std"] = 0.0
            spectrum_geo["spectrum_layers_min"] = ao
            spectrum_geo["spectrum_layers_max"] = ao
            spectrum_geo["spectrum_layers_span"] = 0.0
        metric_values = {
            "gradient_alignment": grad_align,
            "fisher_overlap": fisher_overlap,
            "activation_spectrum_overlap": act_overlap,
            "svcca_overlap": svcca_overlap,
            "linear_cka_overlap": cka_overlap,
        }
        c_couple = weighted_c_couple(metric_values, weights)

        # 保存 A 阶段状态，供不同策略从同一起点出发
        state_a = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # 按策略在 B 上微调并统计遗忘/增益
        for s in strategies:
            model.load_state_dict(state_a, strict=True)
            for name, p in model.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True
            freeze_modules_for_strategy(
                model,
                s,
                layer_scores,
                ratio=0.3,
                seed=int(config["seed"]),
                spectrum_scores=spectrum_scores,
                spectrum_extreme=spectrum_extreme,
            )

            lora_norm_before_b = lora_param_l2_norm(model)
            b_train_stat = train_texts(model, tokenizer, device, b_train, config["train"])
            lora_norm_after_b = lora_param_l2_norm(model)
            a_loss_after_b = eval_loss(model, tokenizer, device, a_eval, config["train"])
            b_loss_after_b = eval_loss(model, tokenizer, device, b_eval, config["train"])

            forgetting = a_loss_after_b - a_loss_after_a
            # 允许 retention > 1 表示反向迁移，避免把信息压扁成 1.0
            retention = safe_div(a_loss_after_a, a_loss_after_b)
            new_task_gain = safe_div((b_loss_before - b_loss_after_b), abs(b_loss_before))

            pareto_rows.append(
                {
                    "pair": pair_name,
                    "strategy": s,
                    "a_loss_before_a": a_loss_before_a,
                    "a_loss_after_a": a_loss_after_a,
                    "a_loss_after_b": a_loss_after_b,
                    "b_loss_before": b_loss_before,
                    "b_loss_after_b": b_loss_after_b,
                    "forgetting": forgetting,
                    "old_task_retention": retention,
                    "new_task_gain": new_task_gain,
                    "b_train_avg_loss": float(b_train_stat["avg_loss"]),
                    "b_train_steps": int(b_train_stat["steps"]),
                    "b_train_skipped_nonfinite": int(b_train_stat.get("skipped_nonfinite", 0)),
                    "b_train_skipped_oom": int(b_train_stat.get("skipped_oom", 0)),
                    "lora_l2_before_b": lora_norm_before_b,
                    "lora_l2_after_b": lora_norm_after_b,
                    "lora_l2_delta_b": lora_norm_after_b - lora_norm_before_b,
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 记录 pair 级指标（以 vanilla 的遗忘作为主标）
        vanilla_rows = [r for r in pareto_rows if r["pair"] == pair_name and r["strategy"] == "vanilla_lora"]
        vanilla_forgetting = float(vanilla_rows[-1]["forgetting"]) if vanilla_rows else float("nan")
        pair_rows.append(
            {
                "pair": pair_name,
                **metric_values,
                "c_couple": c_couple,
                "forgetting": vanilla_forgetting,
                **spectrum_geo,
            }
        )

    metrics_df = pd.DataFrame(pair_rows)
    pareto_df = pd.DataFrame(pareto_rows)
    metrics_path = os.path.join(output_dir, "pair_metrics.csv")
    pareto_path = os.path.join(output_dir, "pareto_points.csv")
    metrics_df.to_csv(metrics_path, index=False)
    pareto_df.to_csv(pareto_path, index=False)

    pearson = {
        "gradient_alignment_vs_forgetting": pearson_safe(metrics_df, "gradient_alignment", "forgetting"),
        "fisher_overlap_vs_forgetting": pearson_safe(metrics_df, "fisher_overlap", "forgetting"),
        "activation_spectrum_overlap_vs_forgetting": pearson_safe(metrics_df, "activation_spectrum_overlap", "forgetting"),
        "svcca_overlap_vs_forgetting": pearson_safe(metrics_df, "svcca_overlap", "forgetting"),
        "linear_cka_overlap_vs_forgetting": pearson_safe(metrics_df, "linear_cka_overlap", "forgetting"),
        "c_couple_vs_forgetting": pearson_safe(metrics_df, "c_couple", "forgetting"),
    }
    for col in ("spectrum_layers_mean", "spectrum_layers_std", "spectrum_layers_span", "spectrum_layers_max", "spectrum_layers_min"):
        if col in metrics_df.columns:
            pearson[f"{col}_vs_forgetting"] = pearson_safe(metrics_df, col, "forgetting")

    summary = {
        "rows": int(len(metrics_df)),
        "pearson_r": pearson,
        "note": "real 模式：指标来自真实顺序微调（A->B）与真实 eval loss。单 seed 样本过少时相关性会返回 NaN。",
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {metrics_path}")
    print(f"[OK] Saved: {pareto_path}")
    print(f"[OK] Saved: {os.path.join(output_dir, 'summary.json')}")


def print_dry_run(config: Dict) -> None:
    print("===== DRY RUN =====")
    print(f"seed: {config['seed']}")
    print(f"output_dir: {config['output_dir']}")
    print(f"model: {config['model']['name']}")
    print("tasks:")
    for t in config["tasks"]:
        print(f"  - {t['name']}")
    print("metrics:", list(config["metrics"]["c_couple_weights"].keys()))
    print("strategies:", config["strategies"])
    print("===================")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA 跨任务干涉烟雾测试最小框架")
    parser.add_argument("--config", type=str, required=True, help="yaml 配置路径")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置中的随机种子")
    parser.add_argument("--output-dir", type=str, default=None, help="覆盖配置中的输出目录")
    parser.add_argument("--dry-run", action="store_true", help="仅检查配置，不跑实验")
    parser.add_argument(
        "--mode",
        type=str,
        default="proxy",
        choices=["proxy", "real"],
        help="proxy: 流程烟雾测试；real: 真实顺序微调实验",
    )
    args = parser.parse_args()

    config = read_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    set_seed(int(config["seed"]))
    output_dir = config["output_dir"]
    ensure_dir(output_dir)

    if args.dry_run:
        print_dry_run(config)
        return

    if args.mode == "proxy":
        run_proxy_smoke(config, output_dir)
        return
    if args.mode == "real":
        run_real_smoke(config, output_dir)
        return

    raise ValueError(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()








