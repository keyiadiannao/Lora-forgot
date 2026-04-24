import argparse
import json
import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd


METRICS = [
    ("k1", "activation_spectrum_overlap"),
    ("k2", "activation_principal_cos_k2"),
    ("k3", "activation_principal_cos_k3"),
    ("k5", "activation_principal_cos_k5"),
    ("grad", "gradient_alignment"),
]


def safe_corr(df: pd.DataFrame, x: str, y: str, method: str) -> float:
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


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return obj


def run_one(label: str, csv_path: str, target: str) -> Dict:
    df = pd.read_csv(csv_path)
    rows: List[Dict] = []
    for alias, col in METRICS:
        rows.append(
            {
                "model": label,
                "metric_alias": alias,
                "metric_column": col,
                "pearson": safe_corr(df, col, target, "pearson"),
                "spearman": safe_corr(df, col, target, "spearman"),
                "n_rows": int(len(df)),
            }
        )
    return {"label": label, "csv": csv_path, "target": target, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate k-vs-r (Pearson+Spearman) table from pair metrics CSVs")
    ap.add_argument("--csv", type=str, nargs="+", required=True, help="CSV specs: label=path")
    ap.add_argument("--target", type=str, default="forgetting")
    ap.add_argument("--output-json", type=str, default=None)
    ap.add_argument("--output-csv", type=str, default=None)
    args = ap.parse_args()

    reports: List[Dict] = []
    flat_rows: List[Dict] = []
    for spec in args.csv:
        if "=" not in spec:
            raise ValueError(f"invalid --csv spec: {spec}, expected label=path")
        label, path = spec.split("=", 1)
        rep = run_one(label=label, csv_path=path, target=args.target)
        reports.append(rep)
        flat_rows.extend(rep["rows"])

    out_df = pd.DataFrame(flat_rows)
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
        out_df.to_csv(args.output_csv, index=False)
        print(f"[OK] {args.output_csv}")
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(sanitize({"target": args.target, "reports": reports}), f, ensure_ascii=False, indent=2)
        print(f"[OK] {args.output_json}")
    if not args.output_csv and not args.output_json:
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
