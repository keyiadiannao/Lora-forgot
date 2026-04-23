"""
对 seed×pair 级 CSV 做留一法 Pearson（LOSO：留一 seed；LOPO：留一 pair），用于多主方向等指标的稳健性检查。

示例（在仓库根目录）：
  python smoke/run_holdout_corr.py \\
    --csv outputs/real_smoke_qwen7b_8pairs_smoke/multiseed_pair_metrics.csv \\
    --target forgetting \\
    --predictors activation_spectrum_overlap activation_principal_cos_k3 activation_principal_cos_k5 \\
    --output-json outputs/real_smoke_qwen7b_8pairs_smoke/holdout_corr.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    x, y = x[m], y[m]
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _loso_rows(df: pd.DataFrame, target: str, predictors: List[str], seed_col: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seeds = sorted(df[seed_col].dropna().unique().tolist())
    for held in seeds:
        train = df[df[seed_col] != held]
        test = df[df[seed_col] == held]
        for p in predictors:
            out.append(
                {
                    "scheme": "loso",
                    "held": int(held) if not isinstance(held, str) else held,
                    "predictor": p,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "r_train": _pearson(train[p].to_numpy(), train[target].to_numpy()),
                    "r_test": _pearson(test[p].to_numpy(), test[target].to_numpy()),
                }
            )
    return out


def _lopo_rows(df: pd.DataFrame, target: str, predictors: List[str], pair_col: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pairs = sorted(df[pair_col].dropna().unique().tolist())
    for held in pairs:
        train = df[df[pair_col] != held]
        test = df[df[pair_col] == held]
        for p in predictors:
            out.append(
                {
                    "scheme": "lopo",
                    "held": held,
                    "predictor": p,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "r_train": _pearson(train[p].to_numpy(), train[target].to_numpy()),
                    "r_test": _pearson(test[p].to_numpy(), test[target].to_numpy()),
                }
            )
    return out


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


def run_holdout(
    csv_path: str,
    target: str,
    predictors: List[str],
    seed_col: str = "seed",
    pair_col: str = "pair",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    for c in [target, seed_col, pair_col] + predictors:
        if c not in df.columns:
            raise ValueError(f"缺少列: {c}")
    df = df.dropna(subset=[target, seed_col, pair_col])
    loso = _loso_rows(df, target, predictors, seed_col)
    lopo = _lopo_rows(df, target, predictors, pair_col)
    table = pd.DataFrame(loso + lopo)
    pooled = {p: _pearson(df[p].to_numpy(), df[target].to_numpy()) for p in predictors}
    summary = {
        "csv": csv_path,
        "target": target,
        "predictors": predictors,
        "n_rows": int(len(df)),
        "pearson_pooled_all_rows": pooled,
        "note": "LOSO 测试集为单 seed 的 8 个 pair；LOPO 测试集为单 pair 的多个 seed。r_test 样本少时波动大，仅作探索性报告。",
    }
    return table, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="seed×pair CSV 的 LOSO / LOPO Pearson")
    ap.add_argument("--csv", type=str, required=True, help="含 seed, pair, forgetting 与各预测列的 CSV")
    ap.add_argument("--target", type=str, default="forgetting")
    ap.add_argument("--predictors", type=str, nargs="+", required=True)
    ap.add_argument("--seed-col", type=str, default="seed")
    ap.add_argument("--pair-col", type=str, default="pair")
    ap.add_argument("--output-csv", type=str, default=None, help="逐折相关系数表")
    ap.add_argument("--output-json", type=str, default=None, help="摘要 JSON")
    args = ap.parse_args()

    table, summary = run_holdout(
        csv_path=args.csv,
        target=args.target,
        predictors=list(args.predictors),
        seed_col=args.seed_col,
        pair_col=args.pair_col,
    )
    out_csv = args.output_csv
    if out_csv:
        _ensure_parent(out_csv)
        table.to_csv(out_csv, index=False)
        print(f"[OK] {out_csv}")
    out_json = args.output_json
    if out_json:
        _ensure_parent(out_json)
        payload = {"summary": summary, "rows": table.to_dict(orient="records")}
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(_sanitize(payload), f, ensure_ascii=False, indent=2)
        print(f"[OK] {out_json}")
    if not out_csv and not out_json:
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
