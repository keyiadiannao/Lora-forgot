import argparse
import json
import math
import os
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


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


def _fit_fixed_effect_ols(df: pd.DataFrame, formula_cols: list[str]) -> Dict[str, Any]:
    """
    退化方案：两因素固定效应 OLS（seed + pair 哑变量）.
    formula_cols: 形如 ["activation_spectrum_overlap"] 或 ["activation_spectrum_overlap", "spectrum_layers_std", "spectrum_layers_span"].
    """
    base = df.copy()
    base = base.dropna(subset=["forgetting"] + formula_cols + ["seed", "pair"])
    if len(base) < 8:
        return {"ok": False, "reason": "样本太少，无法做固定效应回归"}

    y = base["forgetting"].to_numpy(dtype=float)
    x_num = base[formula_cols].to_numpy(dtype=float)
    seed_d = pd.get_dummies(base["seed"].astype(str), prefix="seed", drop_first=True)
    pair_d = pd.get_dummies(base["pair"].astype(str), prefix="pair", drop_first=True)

    X = np.concatenate(
        [
            np.ones((len(base), 1), dtype=float),
            x_num,
            seed_d.to_numpy(dtype=float),
            pair_d.to_numpy(dtype=float),
        ],
        axis=1,
    )
    col_names = ["Intercept"] + formula_cols + list(seed_d.columns) + list(pair_d.columns)

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat

    n = X.shape[0]
    p = X.shape[1]
    dof = max(n - p, 1)
    s2 = float((resid @ resid) / dof)
    xtx_inv = np.linalg.pinv(X.T @ X)
    cov = s2 * xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), a_min=1e-15, a_max=None))
    tvals = beta / se
    pvals = 2.0 * (1.0 - stats.t.cdf(np.abs(tvals), df=dof))

    key_rows = {}
    for name in formula_cols:
        idx = col_names.index(name)
        key_rows[name] = {
            "coef": float(beta[idx]),
            "se": float(se[idx]),
            "t": float(tvals[idx]),
            "p_value": float(pvals[idx]),
        }

    return {
        "ok": True,
        "type": "fixed_effect_ols_fallback",
        "n": int(n),
        "dof": int(dof),
        "predictor_stats": key_rows,
        "note": "未检测到 statsmodels，使用 seed+pair 两因素固定效应 OLS 近似。",
    }


def _fit_mixedlm_if_available(df: pd.DataFrame, formula: str) -> Tuple[bool, Dict[str, Any]]:
    try:
        import statsmodels.formula.api as smf  # type: ignore
    except Exception:
        return False, {"ok": False, "reason": "statsmodels 不可用"}

    data = df.copy().dropna(subset=["forgetting", "activation_spectrum_overlap", "seed", "pair"])
    if "spectrum_layers_std" in formula:
        data = data.dropna(subset=["spectrum_layers_std", "spectrum_layers_span"])
    if len(data) < 8:
        return True, {"ok": False, "reason": "样本太少，无法拟合 mixedlm"}

    try:
        with warnings.catch_warnings(record=True) as warns:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                formula=formula,
                data=data,
                groups=data["seed"].astype(str),
                vc_formula={"pair": "0 + C(pair)"},
            )
            fit = model.fit(reml=False, method="lbfgs", maxiter=200, disp=False)

        warning_msgs = [str(w.message) for w in warns]
        params = {}
        for k, v in fit.params.items():
            if k in fit.pvalues.index:
                params[k] = {
                    "coef": float(v),
                    "p_value": float(fit.pvalues[k]),
                }
        return True, {
            "ok": True,
            "type": "mixedlm",
            "n": int(len(data)),
            "formula": formula,
            "aic": float(fit.aic) if np.isfinite(fit.aic) else None,
            "converged": bool(getattr(fit, "converged", False)),
            "warnings": warning_msgs,
            "params": params,
        }
    except Exception as e:
        return True, {"ok": False, "reason": f"mixedlm 拟合失败: {e}"}


def run_regressions(pair_df: pd.DataFrame) -> Dict[str, Any]:
    # 模型1：主指标
    ok_m1, m1 = _fit_mixedlm_if_available(pair_df, "forgetting ~ activation_spectrum_overlap")
    if not (ok_m1 and m1.get("ok")):
        m1 = _fit_fixed_effect_ols(pair_df, ["activation_spectrum_overlap"])

    # 模型2：主+辅
    ok_m2, m2 = _fit_mixedlm_if_available(
        pair_df, "forgetting ~ activation_spectrum_overlap + spectrum_layers_std + spectrum_layers_span"
    )
    if not (ok_m2 and m2.get("ok")):
        m2 = _fit_fixed_effect_ols(pair_df, ["activation_spectrum_overlap", "spectrum_layers_std", "spectrum_layers_span"])

    return {"model_main": m1, "model_main_plus_aux": m2}


def _safe_corr(df: pd.DataFrame, x: str, y: str) -> float:
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


def predictor_corr_table(pair_df: pd.DataFrame) -> Dict[str, Any]:
    rows = []
    for p in PREDICTORS:
        if p not in pair_df.columns:
            continue
        r = _safe_corr(pair_df, p, "forgetting")
        rows.append({"predictor": p, "pearson_r": r, "abs_r": None if not np.isfinite(r) else abs(float(r))})
    rows = sorted(rows, key=lambda x: -1.0 if x["abs_r"] is None else -x["abs_r"])
    return {"rows": rows}


def run_paired_tests(pareto_df: pd.DataFrame) -> Dict[str, Any]:
    piv = pareto_df.pivot_table(
        index=["seed", "pair"],
        columns="strategy",
        values=["forgetting", "new_task_gain"],
        aggfunc="mean",
    )

    need = [
        ("forgetting", "spectrum_freeze_30"),
        ("forgetting", "random_freeze_30"),
        ("new_task_gain", "spectrum_freeze_30"),
        ("new_task_gain", "random_freeze_30"),
    ]
    if any(c not in piv.columns for c in need):
        return {"ok": False, "reason": "缺少 spectrum_freeze_30 或 random_freeze_30 列"}

    d_forget = (piv[("forgetting", "spectrum_freeze_30")] - piv[("forgetting", "random_freeze_30")]).dropna()
    d_gain = (piv[("new_task_gain", "spectrum_freeze_30")] - piv[("new_task_gain", "random_freeze_30")]).dropna()

    n = int(min(len(d_forget), len(d_gain)))
    if n < 3:
        return {"ok": False, "reason": f"有效配对样本过少: {n}"}

    d_forget = d_forget.iloc[:n]
    d_gain = d_gain.iloc[:n]

    t_forget = stats.ttest_1samp(d_forget, popmean=0.0, nan_policy="omit")
    t_gain = stats.ttest_1samp(d_gain, popmean=0.0, nan_policy="omit")
    try:
        w_forget = stats.wilcoxon(d_forget)
        w_gain = stats.wilcoxon(d_gain)
        w_forget_p = float(w_forget.pvalue)
        w_gain_p = float(w_gain.pvalue)
    except Exception:
        w_forget_p = None
        w_gain_p = None

    return {
        "ok": True,
        "n_pairs": n,
        "delta_definition": {
            "delta_forgetting": "spectrum_freeze_30 - random_freeze_30（负值更好）",
            "delta_gain": "spectrum_freeze_30 - random_freeze_30（正值更好）",
        },
        "forgetting": {
            "mean_delta": float(np.mean(d_forget)),
            "ttest_p_value": float(t_forget.pvalue),
            "wilcoxon_p_value": w_forget_p,
        },
        "new_task_gain": {
            "mean_delta": float(np.mean(d_gain)),
            "ttest_p_value": float(t_gain.pvalue),
            "wilcoxon_p_value": w_gain_p,
        },
    }


def make_p0_report(input_dir: str, output_json: str) -> Dict[str, Any]:
    pair_csv = os.path.join(input_dir, "multiseed_pair_metrics.csv")
    pareto_csv = os.path.join(input_dir, "multiseed_pareto.csv")
    pair_df = pd.read_csv(pair_csv)
    pareto_df = pd.read_csv(pareto_csv)

    out = {
        "input_dir": os.path.abspath(input_dir),
        "n_pair_rows": int(len(pair_df)),
        "n_pareto_rows": int(len(pareto_df)),
        "predictor_corr_vs_forgetting": predictor_corr_table(pair_df),
        "regression_tests": run_regressions(pair_df),
        "paired_test_spectrum_vs_random": run_paired_tests(pareto_df),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(_sanitize(out), f, ensure_ascii=False, indent=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 统计检验：回归 + 配对检验")
    parser.add_argument("--input-dir", type=str, required=True, help="包含 multiseed_pair_metrics.csv 的目录")
    parser.add_argument("--output", type=str, default="", help="默认写入 input-dir/p0_stats_report.json")
    args = parser.parse_args()

    output = args.output or os.path.join(args.input_dir, "p0_stats_report.json")
    make_p0_report(args.input_dir, output)
    print(f"[OK] {output}")


if __name__ == "__main__":
    main()
