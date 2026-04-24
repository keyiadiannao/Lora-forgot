import argparse
import json
import math
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "activation_spectrum_overlap",
    "activation_principal_cos_k2",
    "activation_principal_cos_k3",
    "activation_principal_cos_k5",
    "gradient_alignment",
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


def _canon_pair(pair: str) -> str:
    if "_vs_" not in pair:
        return pair
    a, b = pair.split("_vs_", 1)
    return "__".join(sorted([a, b]))


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


def _simple_ols_with_interaction(df: pd.DataFrame, metric: str, y_col: str = "forgetting") -> Dict[str, Any]:
    base = df.copy()
    base = base.dropna(subset=[y_col, metric, "is_reverse"])
    if len(base) < 8:
        return {"ok": False, "reason": "insufficient_rows"}
    y = base[y_col].to_numpy(dtype=float)
    m = base[metric].to_numpy(dtype=float)
    d = base["is_reverse"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(base)), m, d, m * d])
    names = ["intercept", metric, "is_reverse", f"{metric}:is_reverse"]
    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ beta
    resid = y - y_hat
    n, p = x.shape
    dof = max(n - p, 1)
    s2 = float((resid @ resid) / dof)
    cov = s2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.clip(np.diag(cov), 1e-15, None))
    out = {}
    for i, name in enumerate(names):
        out[name] = {"coef": float(beta[i]), "se": float(se[i])}
    return {"ok": True, "n": int(n), "dof": int(dof), "coefficients": out}


def build_report(forward_csv: str, reverse_csv: str, metrics: List[str]) -> Dict[str, Any]:
    fw = pd.read_csv(forward_csv)
    rv = pd.read_csv(reverse_csv)
    fw["direction"] = "forward"
    rv["direction"] = "reverse"
    fw["is_reverse"] = 0
    rv["is_reverse"] = 1
    both = pd.concat([fw, rv], ignore_index=True)
    if "pair" in both.columns:
        both["pair_canon"] = both["pair"].astype(str).map(_canon_pair)

    corr_rows = []
    for direction, part in [("forward", fw), ("reverse", rv), ("all", both)]:
        for metric in metrics:
            corr_rows.append(
                {
                    "direction": direction,
                    "metric": metric,
                    "pearson": _safe_corr(part, metric, "forgetting", "pearson"),
                    "spearman": _safe_corr(part, metric, "forgetting", "spearman"),
                    "n_rows": int(len(part)),
                }
            )
    corr_df = pd.DataFrame(corr_rows)

    ols = {m: _simple_ols_with_interaction(both, m) for m in metrics}
    return {
        "inputs": {"forward_csv": forward_csv, "reverse_csv": reverse_csv},
        "rows": int(len(both)),
        "corr_rows": corr_rows,
        "ols_interaction": ols,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Directional report: compare forward and reverse forgetting correlations")
    ap.add_argument("--forward-csv", type=str, required=True)
    ap.add_argument("--reverse-csv", type=str, required=True)
    ap.add_argument("--metrics", type=str, nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="optional seed filter applied to both inputs")
    ap.add_argument("--output-json", type=str, default=None)
    ap.add_argument("--output-csv", type=str, default=None)
    args = ap.parse_args()

    if args.seeds:
        fw = pd.read_csv(args.forward_csv)
        rv = pd.read_csv(args.reverse_csv)
        fw = fw[fw["seed"].isin(args.seeds)] if "seed" in fw.columns else fw
        rv = rv[rv["seed"].isin(args.seeds)] if "seed" in rv.columns else rv
        tmp_dir = os.path.join(os.getcwd(), "outputs", "_tmp_directional")
        os.makedirs(tmp_dir, exist_ok=True)
        fw_tmp = os.path.join(tmp_dir, "forward_filtered.csv")
        rv_tmp = os.path.join(tmp_dir, "reverse_filtered.csv")
        fw.to_csv(fw_tmp, index=False)
        rv.to_csv(rv_tmp, index=False)
        report = build_report(fw_tmp, rv_tmp, metrics=list(args.metrics))
        report["inputs"]["seed_filter"] = list(args.seeds)
    else:
        report = build_report(args.forward_csv, args.reverse_csv, metrics=list(args.metrics))
    corr_df = pd.DataFrame(report["corr_rows"])
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
        corr_df.to_csv(args.output_csv, index=False)
        print(f"[OK] {args.output_csv}")
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(_sanitize(report), f, ensure_ascii=False, indent=2)
        print(f"[OK] {args.output_json}")
    if not args.output_csv and not args.output_json:
        print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
