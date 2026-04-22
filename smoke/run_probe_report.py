"""
对多 seed 烟雾输出做「方向探测」汇总：分层相关、按遗忘符号子集的策略均值、简易推荐。

用法:
  python smoke/run_probe_report.py --input-dir outputs/real_smoke_qwen15b_8pairs_quick
  python smoke/run_probe_report.py --pair-csv path/to/multiseed_pair_metrics.csv --pareto-csv path/to/multiseed_pareto.csv --output path/to/probe_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _json_float(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, (np.floating,)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return x


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return _json_float(obj)
    return obj


def _safe_corr(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    if x_col not in df.columns or y_col not in df.columns:
        return float("nan")
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    valid = (~x.isna()) & (~y.isna())
    if int(valid.sum()) < 3:
        return float("nan")
    x = x[valid].to_numpy(dtype=np.float64)
    y = y[valid].to_numpy(dtype=np.float64)
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _predictor_columns(df: pd.DataFrame) -> List[str]:
    base = [
        "gradient_alignment",
        "fisher_overlap",
        "activation_spectrum_overlap",
        "svcca_overlap",
        "linear_cka_overlap",
        "c_couple",
    ]
    extra = [
        "spectrum_layers_mean",
        "spectrum_layers_std",
        "spectrum_layers_span",
        "spectrum_layers_max",
        "spectrum_layers_min",
    ]
    cols = [c for c in base if c in df.columns]
    cols.extend([c for c in extra if c in df.columns])
    return cols


def _subset_corr(df: pd.DataFrame, name: str, predictor_cols: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": name, "n": int(len(df)), "pearson_vs_forgetting": {}}
    for c in predictor_cols:
        out["pearson_vs_forgetting"][c] = _safe_corr(df, c, "forgetting")
    return out


def _strategy_means_on_subset(
    pareto: pd.DataFrame,
    pair_keys: pd.DataFrame,
    strategies: List[str],
) -> Dict[str, Any]:
    """仅在 (seed,pair) 属于 pair_keys 时，对各 strategy 求 forgetting / new_task_gain 均值。"""
    key_cols = ["seed", "pair"]
    sub = pareto.merge(pair_keys, on=key_cols, how="inner")
    rows = []
    for s in strategies:
        d = sub[sub["strategy"] == s]
        if len(d) == 0:
            continue
        rows.append(
            {
                "strategy": s,
                "mean_forgetting": float(d["forgetting"].mean()),
                "mean_new_task_gain": float(d["new_task_gain"].mean()),
                "n": int(len(d)),
            }
        )
    return {"rows": rows}


def _rank_directions(strata: List[Dict[str, Any]], predictor_cols: List[str]) -> List[Dict[str, Any]]:
    """按 |r| 粗排预测变量。"""
    ranked = []
    for st in strata:
        n = st.get("n", 0)
        corrs = st.get("pearson_vs_forgetting", {})
        items = []
        for p in predictor_cols:
            r = corrs.get(p, float("nan"))
            if pd.isna(r):
                continue
            items.append((p, float(r), abs(float(r))))
        items.sort(key=lambda x: -x[2])
        ranked.append(
            {
                "subset": st.get("name"),
                "n": n,
                "by_abs_corr": [{"predictor": a[0], "r": a[1], "abs_r": a[2]} for a in items],
            }
        )
    return ranked


def _predictor_followup_zh(name: str) -> str:
    if name == "activation_spectrum_overlap":
        return "「activation 谱 / 层几何」方向（与全局 16 点结论对照看是否子集特异）"
    if name == "spectrum_layers_span":
        return "「跨层谱差异（层间异质性）」方向，可与按层干预、CKA 等结合"
    if name in ("spectrum_layers_mean", "spectrum_layers_max", "spectrum_layers_min", "spectrum_layers_std"):
        return "「按层 activation 谱统计」方向（与最后一层标量谱互补）"
    if name == "c_couple":
        return "「拆解或重加权 c_couple」方向（注意 n 小时相关易抖，需更多点验证）"
    if name == "fisher_overlap":
        return "「Fisher/二阶近似」类基线或指标"
    if name == "gradient_alignment":
        return "「梯度对齐/冲突」类机制或冻结启发"
    return "该预测变量"


def _recommend(strata: List[Dict[str, Any]], strat_means: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """启发式：正遗忘子集上谁 |r| 最大；该子集上哪策略 forgetting 最低且 gain 不低于 vanilla 均值。"""
    tips: List[str] = []
    pos = next((s for s in strata if s.get("name") == "forgetting_gt_0"), None)
    if pos and pos.get("n", 0) >= 3:
        corrs = pos["pearson_vs_forgetting"]
        items = [(k, float(v)) for k, v in corrs.items() if pd.notna(v)]
        items.sort(key=lambda t: -abs(t[1]))
        if items:
            best_name, best_r = items[0]
            tips.append(
                f"在 vanilla 正遗忘子集 (n={pos['n']}) 上，|r| 最大的是 {best_name} (r≈{best_r:.3f})，可优先推进 {_predictor_followup_zh(best_name)}。"
            )
    else:
        tips.append("正遗忘子集点数 < 3，分层相关不稳定；需更多 seed/pair 或更长训练再判。")

    sm_pos = strat_means.get("forgetting_gt_0", {}).get("rows", [])
    if sm_pos:
        van = next((r for r in sm_pos if r["strategy"] == "vanilla_lora"), None)
        if van:
            vg = van["mean_new_task_gain"]
            candidates = []
            for r in sm_pos:
                if r["strategy"] == "vanilla_lora":
                    continue
                if r["mean_forgetting"] < van["mean_forgetting"] and r["mean_new_task_gain"] >= vg - 1e-6:
                    candidates.append(r["strategy"])
            if candidates:
                tips.append(f"正遗忘子集上相对 vanilla（遗忘更低且 gain 不降）的策略: {', '.join(candidates)}")
            else:
                tips.append("正遗忘子集上暂无策略在「更低遗忘 + gain 不降」上同时优于 vanilla 均值。")

    full = next((s for s in strata if s.get("name") == "all"), None)
    if full and full.get("n", 0) >= 5:
        cor = full["pearson_vs_forgetting"].get("spectrum_layers_span")
        if cor is not None and pd.notna(cor) and abs(float(cor)) >= 0.25:
            tips.append(
                f"全量样本上 spectrum_layers_span 与 forgetting 的 r≈{float(cor):.3f}，可关注「跨层谱差异是否刻画干扰难度」。"
            )
    return {"heuristic_tips_zh": tips}


def run_report(pair_csv: str, pareto_csv: str, output_json: str) -> Dict[str, Any]:
    df = pd.read_csv(pair_csv)
    pareto = pd.read_csv(pareto_csv)
    if "forgetting" not in df.columns:
        raise ValueError(f"{pair_csv} 缺少 forgetting 列")
    if "seed" not in df.columns:
        df = df.copy()
        df["seed"] = 0
    if "seed" not in pareto.columns:
        pareto = pareto.copy()
        pareto["seed"] = 0

    df_pos = df[df["forgetting"] > 0].copy()
    df_neg = df[df["forgetting"] <= 0].copy()
    pred_cols = _predictor_columns(df)
    strata = [
        _subset_corr(df, "all", pred_cols),
        _subset_corr(df_pos, "forgetting_gt_0", pred_cols),
        _subset_corr(df_neg, "forgetting_le_0", pred_cols),
    ]

    strategies = sorted(pareto["strategy"].dropna().unique().tolist())
    keys_all = df[["seed", "pair"]].drop_duplicates()
    keys_pos = df_pos[["seed", "pair"]].drop_duplicates()
    keys_neg = df_neg[["seed", "pair"]].drop_duplicates()

    strat_means = {
        "all": _strategy_means_on_subset(pareto, keys_all, strategies),
        "forgetting_gt_0": _strategy_means_on_subset(pareto, keys_pos, strategies),
        "forgetting_le_0": _strategy_means_on_subset(pareto, keys_neg, strategies),
    }

    ranked = _rank_directions(strata, pred_cols)
    rec = _recommend(strata, strat_means)

    result: Dict[str, Any] = {
        "source_pair_metrics": os.path.abspath(pair_csv),
        "source_pareto": os.path.abspath(pareto_csv),
        "counts": {
            "all_pairs": int(len(df)),
            "forgetting_gt_0": int(len(df_pos)),
            "forgetting_le_0": int(len(df_neg)),
        },
        "stratified_pearson_vs_forgetting": strata,
        "ranked_predictors_by_abs_r": ranked,
        "strategy_mean_by_subset": strat_means,
        "recommendation": rec,
    }

    out_dir = os.path.dirname(os.path.abspath(output_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(result), f, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="烟雾结果方向探测报告")
    p.add_argument(
        "--input-dir",
        type=str,
        default="",
        help="含 multiseed_pair_metrics.csv 与 multiseed_pareto.csv 的目录",
    )
    p.add_argument("--pair-csv", type=str, default="")
    p.add_argument("--pareto-csv", type=str, default="")
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="probe_report.json 路径；默认写入 input-dir",
    )
    args = p.parse_args()

    if args.input_dir:
        pair_csv = os.path.join(args.input_dir, "multiseed_pair_metrics.csv")
        pareto_csv = os.path.join(args.input_dir, "multiseed_pareto.csv")
        out_json = args.output or os.path.join(args.input_dir, "probe_report.json")
    else:
        if not args.pair_csv or not args.pareto_csv:
            raise SystemExit("请指定 --input-dir 或同时指定 --pair-csv 与 --pareto-csv")
        pair_csv = args.pair_csv
        pareto_csv = args.pareto_csv
        out_json = args.output or "probe_report.json"

    if not os.path.isfile(pair_csv):
        raise SystemExit(f"找不到: {pair_csv}")
    if not os.path.isfile(pareto_csv):
        raise SystemExit(f"找不到: {pareto_csv}")

    run_report(pair_csv, pareto_csv, out_json)
    print(f"[OK] {out_json}")


if __name__ == "__main__":
    main()
