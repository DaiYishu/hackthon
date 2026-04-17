"""Regime-based mean reversion baseline for Stochastic (per asset_class).

Dataset
-------
Input parquet must contain:
- asset_id, asset_class, date, stoch_k

Goal
----
Predict:

    stoch_k(t + horizon)

via a regime-dependent mean reversion model:

    delta = stoch_k(t+h) - stoch_k(t)
    delta ≈ α_regime(t) * (50 - stoch_k(t))

Regimes
-------
- LOW:  stoch_k < 20
- MID:  20 <= stoch_k <= 80
- HIGH: stoch_k > 80

Features
--------
- X_LOW  = (50 - stoch_k) * I_LOW
- X_MID  = (50 - stoch_k) * I_MID
- X_HIGH = (50 - stoch_k) * I_HIGH

Modeling
--------
Fit ONE linear regression per asset_class:

    delta ~ X_LOW + X_MID + X_HIGH

with no intercept (so when stoch_k==50, predicted delta==0).

No leakage
----------
We do a time-based split per asset_class after combining assets.
To avoid label leakage near the split boundary, we exclude rows whose
target timestamp (date_target) crosses from train into test.

Outputs
-------
- stoch_mean_reversion_results.csv  (per asset_id)
- stoch_mean_reversion_summary.csv  (per asset_class)

"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


REQUIRED_COLS = ["asset_id", "asset_class", "date", "stoch_k"]

MEAN_LEVEL = 50.0
LOW_TH = 20.0
HIGH_TH = 80.0


def load_dataset(input_path: str | Path) -> pd.DataFrame:
    """STEP 1 — Load parquet file and sort by (asset_id, date)."""

    input_path = Path(input_path)
    df = pd.read_parquet(input_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stoch_k"] = pd.to_numeric(df["stoch_k"], errors="coerce")

    df = df.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return df


def build_target_and_delta(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """STEP 2 — Build stoch_target and delta per asset_id."""

    parts: list[pd.DataFrame] = []

    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values(["date"], kind="mergesort").copy()

        g["stoch_target"] = g["stoch_k"].shift(-int(horizon))
        g["date_target"] = g["date"].shift(-int(horizon))
        g["delta"] = g["stoch_target"] - g["stoch_k"]
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)

    out = out.dropna(subset=["stoch_k", "stoch_target", "date_target", "delta"]).copy()
    return out


def add_regime_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """STEP 3 — Define regimes and create I_LOW / I_MID / I_HIGH."""

    df = df.copy()
    k = pd.to_numeric(df["stoch_k"], errors="coerce").to_numpy(dtype=float)

    i_low = k < LOW_TH
    i_mid = (k >= LOW_TH) & (k <= HIGH_TH)
    i_high = k > HIGH_TH

    df["I_LOW"] = i_low.astype(int)
    df["I_MID"] = i_mid.astype(int)
    df["I_HIGH"] = i_high.astype(int)

    return df


def add_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """STEP 4 — Build X_LOW / X_MID / X_HIGH."""

    df = df.copy()
    k = pd.to_numeric(df["stoch_k"], errors="coerce").to_numpy(dtype=float)

    mean_term = MEAN_LEVEL - k
    df["X_LOW"] = mean_term * df["I_LOW"].to_numpy(dtype=float)
    df["X_MID"] = mean_term * df["I_MID"].to_numpy(dtype=float)
    df["X_HIGH"] = mean_term * df["I_HIGH"].to_numpy(dtype=float)

    return df


def assign_time_split_no_leakage_per_class(
    df: pd.DataFrame,
    train_frac: float,
) -> pd.DataFrame:
    """STEP 5 — Combine assets then do a time-based split per asset_class.

    Split is based on feature timestamp `date` within each asset_class.

    Leakage guard:
    - Train rows require BOTH date <= cutoff_date AND date_target <= cutoff_date.
      (so targets are not taken from the test period)

    - Test rows require date > cutoff_date.

    Rows that are neither train nor test are marked as 'gap' and excluded.
    """

    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must be in (0, 1)")

    parts: list[pd.DataFrame] = []

    for asset_class, g in df.groupby("asset_class", sort=False):
        g = g.sort_values(["date"], kind="mergesort").copy()

        n = int(len(g))
        if n == 0:
            continue

        cut_idx = int(math.floor(float(train_frac) * n)) - 1
        cut_idx = max(0, min(cut_idx, n - 1))
        cutoff_date = g["date"].iloc[cut_idx]

        train_mask = (g["date"] <= cutoff_date) & (g["date_target"] <= cutoff_date)
        test_mask = g["date"] > cutoff_date

        split = np.full(n, "gap", dtype=object)
        split[train_mask.to_numpy(dtype=bool)] = "train"
        split[test_mask.to_numpy(dtype=bool)] = "test"

        g["split"] = split
        g["_cutoff_date"] = cutoff_date
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return out


@dataclass(frozen=True)
class ClassModel:
    asset_class: str
    alpha_low: float
    alpha_mid: float
    alpha_high: float
    model: LinearRegression
    n_train_rows: int


def fit_models_per_class(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, ClassModel], dict[str, str]]:
    """Fit one linear regression per asset_class on training rows."""

    models: dict[str, ClassModel] = {}
    failures: dict[str, str] = {}

    train_df = df[df["split"] == "train"].copy()

    for asset_class, g in train_df.groupby("asset_class", sort=False):
        g = g.dropna(subset=feature_cols + ["delta"]).copy()

        if len(g) < 10:
            failures[str(asset_class)] = "too_few_train_rows"
            continue

        X = g[feature_cols].to_numpy(dtype=float)
        y = g["delta"].to_numpy(dtype=float)

        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            failures[str(asset_class)] = "non_finite_values_in_train"
            continue

        lr = LinearRegression(fit_intercept=False)
        lr.fit(X, y)

        coef = np.asarray(lr.coef_, dtype=float).ravel()
        if coef.shape[0] != len(feature_cols):
            failures[str(asset_class)] = "unexpected_coef_shape"
            continue

        models[str(asset_class)] = ClassModel(
            asset_class=str(asset_class),
            alpha_low=float(coef[0]),
            alpha_mid=float(coef[1]),
            alpha_high=float(coef[2]),
            model=lr,
            n_train_rows=int(len(g)),
        )

    return models, failures


def predict_on_test(
    df: pd.DataFrame,
    models: dict[str, ClassModel],
    feature_cols: list[str],
) -> pd.DataFrame:
    """STEP 6/7 — Predict delta and clip stoch_pred into [0, 100]."""

    out = df.copy()
    out["delta_pred"] = np.nan
    out["stoch_pred"] = np.nan

    test_mask = out["split"] == "test"
    if not bool(test_mask.any()):
        return out

    for asset_class, g_idx in out.loc[test_mask].groupby("asset_class", sort=False).groups.items():
        asset_class = str(asset_class)
        if asset_class not in models:
            continue

        idx = np.asarray(list(g_idx), dtype=int)
        X = out.loc[idx, feature_cols].to_numpy(dtype=float)
        delta_pred = models[asset_class].model.predict(X)

        out.loc[idx, "delta_pred"] = delta_pred
        out.loc[idx, "stoch_pred"] = out.loc[idx, "stoch_k"].to_numpy(dtype=float) + delta_pred

    # Clip predictions
    out.loc[test_mask, "stoch_pred"] = out.loc[test_mask, "stoch_pred"].clip(lower=0.0, upper=100.0)
    return out


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return float("nan")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def evaluate_per_asset(
    df_pred: pd.DataFrame,
    models: dict[str, ClassModel],
    model_failures: dict[str, str],
    n_rows_total_by_asset: dict[str, int],
) -> pd.DataFrame:
    """STEP 8 — Evaluate per asset_id on test rows only."""

    results_rows: list[dict[str, object]] = []

    for asset_id, g in df_pred.groupby("asset_id", sort=False):
        g = g.sort_values(["date"], kind="mergesort")
        asset_class = str(g["asset_class"].iloc[0])
        asset_id_str = str(asset_id)

        row: dict[str, object] = {
            "asset_id": asset_id_str,
            "asset_class": asset_class,
            "status": "ok",
            "reason": "",
            "n_rows_total": int(n_rows_total_by_asset.get(asset_id_str, len(g))),
            "cutoff_date": g.get("_cutoff_date", pd.Series([pd.NaT])).iloc[0],
            "n_train_rows": int((g["split"] == "train").sum()),
            "n_gap_rows": int((g["split"] == "gap").sum()),
            "n_test_rows": int((g["split"] == "test").sum()),
        }

        if asset_class not in models:
            row["status"] = "skipped"
            row["reason"] = f"no_class_model: {model_failures.get(asset_class, 'unknown')}"
            results_rows.append(row)
            continue

        m = models[asset_class]
        row.update({"alpha_low": m.alpha_low, "alpha_mid": m.alpha_mid, "alpha_high": m.alpha_high})

        gt = g[g["split"] == "test"].copy()
        if len(gt) < 3:
            row["status"] = "skipped"
            row["reason"] = "too_few_test_rows"
            results_rows.append(row)
            continue

        pred = pd.to_numeric(gt["stoch_pred"], errors="coerce").to_numpy(dtype=float)
        actual = pd.to_numeric(gt["stoch_target"], errors="coerce").to_numpy(dtype=float)
        current = pd.to_numeric(gt["stoch_k"], errors="coerce").to_numpy(dtype=float)

        ok = np.isfinite(pred) & np.isfinite(actual) & np.isfinite(current)
        n_ok = int(ok.sum())
        n_eval = int(len(gt))

        if n_ok < 3:
            row["status"] = "skipped"
            row["reason"] = "too_few_finite_predictions"
            results_rows.append(row)
            continue

        pred_ok = pred[ok]
        actual_ok = actual[ok]
        current_ok = current[ok]

        corr = _safe_corr(pred_ok, actual_ok)

        pred_dir = np.sign(pred_ok - current_ok)
        actual_dir = np.sign(actual_ok - current_ok)
        trend_acc = float(np.mean(pred_dir == actual_dir))

        row.update(
            {
                "n_eval": n_eval,
                "n_pred_ok": n_ok,
                "pred_ok_rate": float(n_ok / n_eval) if n_eval else float("nan"),
                "corr_pred_vs_actual": float(corr),
                "trend_accuracy": float(trend_acc),
            }
        )

        results_rows.append(row)

    return pd.DataFrame(results_rows)


def build_class_summary(results: pd.DataFrame, models: dict[str, ClassModel]) -> pd.DataFrame:
    """STEP 9 — Per-class summary over successfully modeled assets."""

    if results.empty:
        return pd.DataFrame(
            columns=[
                "asset_class",
                "n_assets_modeled",
                "avg_corr",
                "avg_trend_accuracy",
                "avg_pred_ok_rate",
                "alpha_low",
                "alpha_mid",
                "alpha_high",
            ]
        )

    ok = results[results["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(
            columns=[
                "asset_class",
                "n_assets_modeled",
                "avg_corr",
                "avg_trend_accuracy",
                "avg_pred_ok_rate",
                "alpha_low",
                "alpha_mid",
                "alpha_high",
            ]
        )

    summary = (
        ok.groupby("asset_class", dropna=False)
        .agg(
            n_assets_modeled=("asset_id", "nunique"),
            avg_corr=("corr_pred_vs_actual", "mean"),
            avg_trend_accuracy=("trend_accuracy", "mean"),
            avg_pred_ok_rate=("pred_ok_rate", "mean"),
            alpha_low=("alpha_low", "first"),
            alpha_mid=("alpha_mid", "first"),
            alpha_high=("alpha_high", "first"),
        )
        .reset_index()
        .sort_values(["asset_class"], kind="mergesort")
    )

    # Ensure classes with fitted model but zero ok assets still appear (optional, minimal).
    # Keep the UX simple: only include modeled assets.
    return summary


def main() -> int:
    default_input = Path("data") / "data_15mins" / "stoch_features_15min.parquet"
    default_output_dir = Path("results_stoch") / "model2_mean_reversion"

    parser = argparse.ArgumentParser(description="Regime-based mean reversion model for stoch_k(t+h)")
    parser.add_argument("-i", "--input", default=str(default_input), help="Input parquet path")
    parser.add_argument("-o", "--output-dir", default=str(default_output_dir), help="Output directory")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon in steps")
    parser.add_argument("--train-frac", type=float, default=0.80, help="Train fraction (time-based per class)")

    args = parser.parse_args()

    horizon = int(args.horizon)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = ["X_LOW", "X_MID", "X_HIGH"]

    print("STEP 1 — Load data")
    print("=" * 80)
    df = load_dataset(args.input)
    print(f"Input shape: {df.shape}")
    n_rows_total_by_asset = df.groupby("asset_id").size().astype(int).astype(int).to_dict()

    print("STEP 2 — Build target and delta")
    print("=" * 80)
    df_sup = build_target_and_delta(df, horizon=horizon)
    print(f"Supervised rows (after dropna target): {df_sup.shape}")

    print("STEP 3 — Define regimes")
    print("=" * 80)
    df_reg = add_regime_indicators(df_sup)

    print("STEP 4 — Build features")
    print("=" * 80)
    df_feat = add_mean_reversion_features(df_reg)

    print("STEP 5 — Per-class split + fit")
    print("=" * 80)
    df_split = assign_time_split_no_leakage_per_class(df_feat, train_frac=float(args.train_frac))

    n_train = int((df_split["split"] == "train").sum())
    n_gap = int((df_split["split"] == "gap").sum())
    n_test = int((df_split["split"] == "test").sum())
    print(f"Rows: train={n_train} gap={n_gap} test={n_test}")

    models, failures = fit_models_per_class(df_split, feature_cols=feature_cols)
    print(f"Class models fitted: {len(models)}")
    if failures:
        print(f"Class fit failures: {len(failures)}")

    print("STEP 6/7 — Predict and clip")
    print("=" * 80)
    df_pred = predict_on_test(df_split, models=models, feature_cols=feature_cols)

    print("STEP 8 — Evaluation")
    print("=" * 80)
    results = evaluate_per_asset(
        df_pred,
        models=models,
        model_failures=failures,
        n_rows_total_by_asset={str(k): int(v) for k, v in n_rows_total_by_asset.items()},
    )
    summary = build_class_summary(results, models=models)

    print("STEP 9 — Save results")
    print("=" * 80)
    results_path = out_dir / "stoch_mean_reversion_results.csv"
    summary_path = out_dir / "stoch_mean_reversion_summary.csv"

    results.to_csv(results_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
