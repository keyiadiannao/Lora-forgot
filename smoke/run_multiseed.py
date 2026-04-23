import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List

import pandas as pd
import yaml


def read_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_single_seed(
    python_bin: str,
    run_smoke_path: str,
    config_path: str,
    mode: str,
    seed: int,
    out_dir: str,
) -> int:
    cmd = [
        python_bin,
        run_smoke_path,
        "--config",
        config_path,
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--output-dir",
        out_dir,
    ]
    print(f"[RUN] seed={seed} -> {out_dir}")
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def load_seed_outputs(out_dir: str):
    summary_path = os.path.join(out_dir, "summary.json")
    pair_path = os.path.join(out_dir, "pair_metrics.csv")
    pareto_path = os.path.join(out_dir, "pareto_points.csv")
    if not (os.path.exists(summary_path) and os.path.exists(pair_path) and os.path.exists(pareto_path)):
        raise FileNotFoundError(f"缺少输出文件: {out_dir}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    pair_df = pd.read_csv(pair_path)
    pareto_df = pd.read_csv(pareto_path)
    return summary, pair_df, pareto_df


def safe_corr(df: pd.DataFrame, x_col: str, y_col: str) -> float:
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
    return float(x.corr(y))


def aggregate_results(
    base_out_dir: str,
    seeds: List[int],
    h2_compare_strategy: str = "c_couple_freeze_30",
    run_probe: bool = True,
) -> Dict:
    summary_rows = []
    pair_rows = []
    pareto_rows = []

    for seed in seeds:
        out_dir = os.path.join(base_out_dir, f"seed_{seed}")
        summary, pair_df, pareto_df = load_seed_outputs(out_dir)
        summary_rows.append(
            {
                "seed": seed,
                **summary.get("pearson_r", {}),
            }
        )
        pair_df["seed"] = seed
        pareto_df["seed"] = seed
        pair_rows.append(pair_df)
        pareto_rows.append(pareto_df)

    summary_df = pd.DataFrame(summary_rows)
    all_pair_df = pd.concat(pair_rows, ignore_index=True)
    all_pareto_df = pd.concat(pareto_rows, ignore_index=True)
    n_points = int(len(all_pair_df))

    corr_mean = summary_df.drop(columns=["seed"], errors="ignore").mean(numeric_only=True).to_dict()
    corr_std = summary_df.drop(columns=["seed"], errors="ignore").std(numeric_only=True).fillna(0.0).to_dict()
    corr_all = {
        "gradient_alignment_vs_forgetting": safe_corr(all_pair_df, "gradient_alignment", "forgetting"),
        "fisher_overlap_vs_forgetting": safe_corr(all_pair_df, "fisher_overlap", "forgetting"),
        "activation_spectrum_overlap_vs_forgetting": safe_corr(all_pair_df, "activation_spectrum_overlap", "forgetting"),
        "c_couple_vs_forgetting": safe_corr(all_pair_df, "c_couple", "forgetting"),
    }
    for col in ("svcca_overlap", "linear_cka_overlap"):
        if col in all_pair_df.columns:
            corr_all[f"{col}_vs_forgetting"] = safe_corr(all_pair_df, col, "forgetting")
    for col in ("activation_principal_cos_k3", "activation_principal_cos_k5"):
        if col in all_pair_df.columns:
            corr_all[f"{col}_vs_forgetting"] = safe_corr(all_pair_df, col, "forgetting")
    for col in (
        "spectrum_layers_mean",
        "spectrum_layers_std",
        "spectrum_layers_span",
        "spectrum_layers_max",
        "spectrum_layers_min",
    ):
        if col in all_pair_df.columns:
            corr_all[f"{col}_vs_forgetting"] = safe_corr(all_pair_df, col, "forgetting")

    strategy_stat = (
        all_pareto_df.groupby("strategy")[["old_task_retention", "new_task_gain", "forgetting"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )

    h2_pairs = 0
    h2_total = 0
    fg_pivot = all_pareto_df.pivot_table(
        index=["seed", "pair"],
        columns="strategy",
        values=["forgetting", "new_task_gain"],
        aggfunc="mean",
    )
    col_f = ("forgetting", h2_compare_strategy)
    col_g = ("new_task_gain", h2_compare_strategy)
    if col_f in fg_pivot.columns and ("forgetting", "vanilla_lora") in fg_pivot.columns:
        for idx in fg_pivot.index:
            h2_total += 1
            forget_ok = fg_pivot.loc[idx, col_f] <= fg_pivot.loc[idx, ("forgetting", "vanilla_lora")]
            gain_ok = fg_pivot.loc[idx, col_g] >= fg_pivot.loc[idx, ("new_task_gain", "vanilla_lora")]
            if bool(forget_ok and gain_ok):
                h2_pairs += 1

    h1_corr = float(corr_all.get("c_couple_vs_forgetting", float("nan")))
    single_corrs = [
        abs(float(corr_all.get("gradient_alignment_vs_forgetting", float("nan")))),
        abs(float(corr_all.get("fisher_overlap_vs_forgetting", float("nan")))),
        abs(float(corr_all.get("activation_spectrum_overlap_vs_forgetting", float("nan")))),
    ]
    single_corrs = [x for x in single_corrs if pd.notna(x)]
    h1_pass = False
    if n_points >= 15 and pd.notna(h1_corr) and single_corrs:
        h1_pass = abs(h1_corr) >= (max(single_corrs) + 0.05)
    h2_pass = (h2_total > 0) and (h2_pairs / h2_total >= 0.5)

    result = {
        "num_seeds": len(seeds),
        "seeds": seeds,
        "n_points_seed_pair": n_points,
        "pearson_mean": corr_mean,
        "pearson_std": corr_std,
        "pearson_all_seed_pair": corr_all,
        "strategy_mean": strategy_stat,
        "h1_pass_rule": "使用所有 seed×pair 点，且 n_points>=15：abs(c_couple_corr) >= max(abs(single_corrs)) + 0.05",
        "h1_pass": bool(h1_pass),
        "h2_compare_strategy": h2_compare_strategy,
        "h2_pass_rule": f"{h2_compare_strategy} 在 >=50% (seed,pair) 上同时不劣于 vanilla 的 forgetting(更小) 和 gain(更大)",
        "h2_pass": bool(h2_pass),
        "h2_support_count": int(h2_pairs),
        "h2_total_count": int(h2_total),
    }

    summary_csv = os.path.join(base_out_dir, "multiseed_corr.csv")
    all_pair_csv = os.path.join(base_out_dir, "multiseed_pair_metrics.csv")
    all_pareto_csv = os.path.join(base_out_dir, "multiseed_pareto.csv")
    json_path = os.path.join(base_out_dir, "multiseed_summary.json")

    summary_df.to_csv(summary_csv, index=False)
    all_pair_df.to_csv(all_pair_csv, index=False)
    all_pareto_df.to_csv(all_pareto_csv, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved: {summary_csv}")
    print(f"[OK] Saved: {all_pair_csv}")
    print(f"[OK] Saved: {all_pareto_csv}")
    print(f"[OK] Saved: {json_path}")

    if run_probe:
        probe_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_probe_report.py")
        if os.path.isfile(probe_script):
            pr = subprocess.run(
                [sys.executable, probe_script, "--input-dir", base_out_dir],
                check=False,
            )
            if pr.returncode == 0:
                print(f"[OK] probe_report.json -> {base_out_dir}")
            else:
                print(f"[WARN] probe_report 退出码 {pr.returncode}，可手动运行: python smoke/run_probe_report.py --input-dir {base_out_dir}")
        else:
            print(f"[WARN] 未找到 {probe_script}，跳过 probe 报告")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="多seed烟雾测试批量运行与汇总")
    parser.add_argument("--config", type=str, required=True, help="run_smoke 的配置文件")
    parser.add_argument("--mode", type=str, default="real", choices=["proxy", "real"], help="运行模式")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="随机种子列表，例如 42 43 44")
    parser.add_argument("--python-bin", type=str, default="python", help="Python 解释器路径")
    parser.add_argument("--run-smoke-path", type=str, default="smoke\\run_smoke.py", help="run_smoke.py 路径")
    parser.add_argument("--aggregate-only", action="store_true", help="只读取已有 seed 输出做聚合，不重新运行")
    parser.add_argument(
        "--h2-strategy",
        type=str,
        default="c_couple_freeze_30",
        help="H2 比较用的冻结策略名（需出现在 pareto 的 strategy 列），例如 c_couple_freeze_30 或 spectrum_freeze_30",
    )
    parser.add_argument("--no-probe", action="store_true", help="聚合完成后不自动生成 probe_report.json")
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    base_out_dir = cfg["output_dir"]
    ensure_dir(base_out_dir)

    if not args.aggregate_only:
        rc_list = []
        for seed in args.seeds:
            seed_out = os.path.join(base_out_dir, f"seed_{seed}")
            ensure_dir(seed_out)
            rc = run_single_seed(
                python_bin=args.python_bin,
                run_smoke_path=args.run_smoke_path,
                config_path=args.config,
                mode=args.mode,
                seed=seed,
                out_dir=seed_out,
            )
            rc_list.append((seed, rc))

        failed = [seed for seed, rc in rc_list if rc != 0]
        if failed:
            raise RuntimeError(f"以下 seed 运行失败: {failed}")

    aggregate_results(
        base_out_dir=base_out_dir,
        seeds=args.seeds,
        h2_compare_strategy=args.h2_strategy,
        run_probe=not args.no_probe,
    )


if __name__ == "__main__":
    main()
