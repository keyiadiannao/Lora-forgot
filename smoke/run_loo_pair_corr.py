"""
留一 pair（LOPO）：每次去掉一个 pair 名称对应的全部行后，在剩余 seed×pair 点上
重算各预测列与 target 的 Pearson / Spearman。

可选在进入循环前丢掉 pair 名包含某子串的所有行（例如先去掉所有含 gsm8k 的 pair，
再在剩余 pair 上做 LOPO）。

示例：
  python smoke/run_loo_pair_corr.py \\
    --csv outputs/real_smoke_qwen7b_8pairs_smoke/multiseed_pair_metrics.csv \\
    --target forgetting \\
    --output-csv outputs/.../loo_pair_corr.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

DEFAULT_PREDICTORS = [
    "activation_spectrum_overlap",
    "activation_principal_cos_k2",
    "activation_principal_cos_k3",
    "activation_principal_cos_k5",
    "gradient_alignment",
    "fisher_overlap",
    "svcca_overlap",
    "linear_cka_overlap",
    "c_couple",
]


def _safe_corr(df: pd.DataFrame, x: str, y: str, method: str) -> float:
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
    return float(a.corr(b, method=method))


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


def _ensure_parent(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def run_lopo(
    df: pd.DataFrame,
    target: str,
    predictors: List[str],
    pair_col: str,
) -> pd.DataFrame:
    pairs = sorted(df[pair_col].dropna().unique().tolist())
    rows: List[Dict[str, Any]] = []
    for held in pairs:
        sub = df[df[pair_col] != held]
        for p in predictors:
            if p not in sub.columns:
                continue
            rows.append(
                {
                    "held_pair": held,
                    "predictor": p,
                    "n_rows": int(len(sub)),
                    "pearson": _safe_corr(sub, p, target, "pearson"),
                    "spearman": _safe_corr(sub, p, target, "spearman"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="LOPO：去掉单个 pair 后重算 Pearson/Spearman")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--target", type=str, default="forgetting")
    ap.add_argument(
        "--predictors",
        type=str,
        nargs="*",
        default=None,
        help="默认为一组常用列；CSV 中不存在的列会跳过",
    )
    ap.add_argument("--pair-col", type=str, default="pair")
    ap.add_argument(
        "--base-exclude-substr",
        type=str,
        default=None,
        help="先删除 pair 名包含该子串的所有行（大小写不敏感），再做 LOPO",
    )
    ap.add_argument("--output-csv", type=str, default=None)
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df.columns or args.pair_col not in df.columns:
        raise ValueError(f"CSV 需含列: {args.target}, {args.pair_col}")

    predictors = list(args.predictors) if args.predictors is not None else list(DEFAULT_PREDICTORS)
    predictors = [p for p in predictors if p in df.columns]
    if not predictors:
        raise ValueError("无可用预测列：请检查 --predictors 或 CSV 列名")

    work = df.dropna(subset=[args.target, args.pair_col]).copy()
    note = ""
    if args.base_exclude_substr:
        pat = str(args.base_exclude_substr)
        mask = work[args.pair_col].astype(str).str.contains(pat, case=False, regex=False)
        work = work[~mask].copy()
        note = f"已按 --base-exclude-substr 删除含 {pat!r} 的 pair 行后做 LOPO。"

    if work.empty:
        raise ValueError("过滤后无数据行")

    table = run_lopo(work, args.target, predictors, args.pair_col)
    pooled = {
        p: {
            "pearson": _safe_corr(work, p, args.target, "pearson"),
            "spearman": _safe_corr(work, p, args.target, "spearman"),
        }
        for p in predictors
    }
    summary: Dict[str, Any] = {
        "csv": args.csv,
        "target": args.target,
        "predictors": predictors,
        "n_rows_input": int(len(df)),
        "n_rows_used": int(len(work)),
        "n_pairs_held": int(work[args.pair_col].nunique()),
        "pearson_spearman_pooled_on_used_rows": pooled,
        "note": (
            "每行 held_pair：去掉该 pair 全部点后，在剩余点上计算相关。"
            + (" " + note if note else "")
        ),
    }

    if args.output_csv:
        _ensure_parent(args.output_csv)
        table.to_csv(args.output_csv, index=False)
        print(f"[OK] {args.output_csv}")
    if args.output_json:
        _ensure_parent(args.output_json)
        payload = {"summary": summary, "lopo_rows": table.to_dict(orient="records")}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(_sanitize(payload), f, ensure_ascii=False, indent=2)
        print(f"[OK] {args.output_json}")
    if not args.output_csv and not args.output_json:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
