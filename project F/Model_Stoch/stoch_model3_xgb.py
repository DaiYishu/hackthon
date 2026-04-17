"""XGBoost baseline for Stochastic (per asset_class) to predict stoch_k(t+15).

Dataset
-------
Input parquet must contain:
- asset_id, asset_class, date, stoch_k

Goal
----
Predict:

    y(t) = stoch_k(t + horizon)

Modeling strategy
-----------------
- Per-class modeling (fit one model per asset_class)
- Time-based split (80/20) after combining assets
- No leakage: training rows must have both feature timestamp (date) and target
  timestamp (date_target) within the training period.

Feature engineering (per asset_id)
----------------------------------
Lag features:
- lag1, lag2, lag3, lag5, lag10

Momentum:
- delta1 = stoch_k - lag1
- delta2 = lag1 - lag2

Mean reversion:
- mr = 50 - stoch_k

Regimes:
- is_low  = (stoch_k < 20)
- is_high = (stoch_k > 80)

Interactions:
- mr_low  = mr * is_low
- mr_high = mr * is_high

Outputs
-------
- stoch_xgb_results.csv  (per asset_id)
- stoch_xgb_summary.csv  (per asset_class)

"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


REQUIRED_COLS = ["asset_id", "asset_class", "date", "stoch_k"]

MEAN_LEVEL = 50.0
LOW_TH = 20.0
HIGH_TH = 80.0

LAGS = (1, 2, 3, 5, 10)


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


def add_features_per_asset(df: pd.DataFrame) -> pd.DataFrame:
    """STEP 2 — Feature engineering per asset_id."""

    parts: list[pd.DataFrame] = []

    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values(["date"], kind="mergesort").copy()

        for lag in LAGS:
            g[f"lag{lag}"] = g["stoch_k"].shift(int(lag))

        g["delta1"] = g["stoch_k"] - g["lag1"]
        g["delta2"] = g["lag1"] - g["lag2"]

        g["mr"] = MEAN_LEVEL - g["stoch_k"]

        g["is_low"] = (g["stoch_k"] < LOW_TH).astype(int)
        g["is_high"] = (g["stoch_k"] > HIGH_TH).astype(int)

        g["mr_low"] = g["mr"] * g["is_low"].astype(float)
        g["mr_high"] = g["mr"] * g["is_high"].astype(float)

        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return out


def add_target_per_asset(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """STEP 3 — Build target y=stoch_k(t+horizon) per asset_id."""

    parts: list[pd.DataFrame] = []

    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values(["date"], kind="mergesort").copy()

        g["y"] = g["stoch_k"].shift(-int(horizon))
        g["date_target"] = g["date"].shift(-int(horizon))
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return out


def build_supervised(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Drop NaN rows for features/target (as required)."""

    needed = ["stoch_k", "y", "date", "date_target"] + list(feature_cols)
    out = df.dropna(subset=needed).copy()

    # Ensure numeric dtype for modeling.
    out[feature_cols] = out[feature_cols].apply(pd.to_numeric, errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out["stoch_k"] = pd.to_numeric(out["stoch_k"], errors="coerce")
    out = out.dropna(subset=["y", "stoch_k"] + list(feature_cols)).copy()
    return out


def assign_split_per_class(df: pd.DataFrame, train_frac: float) -> pd.DataFrame:
    """STEP 4 — Per-class time split with leakage guard.

    - Combine assets per class
    - Split 80/20 by time order (date)

    Leakage guard:
    - Train rows require both date <= cutoff_date AND date_target <= cutoff_date.
    - Test rows require date > cutoff_date.
    """

    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must be in (0, 1)")

    parts: list[pd.DataFrame] = []

    for asset_class, g in df.groupby("asset_class", sort=False):
        g = g.sort_values(["date", "asset_id"], kind="mergesort").copy()

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
        g["cutoff_date"] = cutoff_date
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return out


@dataclass(frozen=True)
class ClassModel:
    asset_class: str
    model: XGBRegressor
    n_train_rows: int
    n_test_rows: int
    cutoff_date: pd.Timestamp


def fit_predict_per_class(
    df: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, ClassModel], dict[str, str]]:
    """STEP 4/5 — Fit XGBRegressor per class; predict on test."""

    out = df.copy()
    out["y_pred"] = np.nan

    models: dict[str, ClassModel] = {}
    failures: dict[str, str] = {}

    for asset_class, g in out.groupby("asset_class", sort=False):
        g = g.sort_values(["date", "asset_id"], kind="mergesort").copy()

        train_df = g[g["split"] == "train"].copy()
        test_df = g[g["split"] == "test"].copy()

        if len(train_df) < 50:
            failures[str(asset_class)] = "too_few_train_rows"
            continue
        if len(test_df) < 10:
            failures[str(asset_class)] = "too_few_test_rows"
            continue

        X_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df["y"].to_numpy(dtype=float)

        if not (np.isfinite(X_train).all() and np.isfinite(y_train).all()):
            failures[str(asset_class)] = "non_finite_values_in_train"
            continue

        model = XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)

        X_test = test_df[feature_cols].to_numpy(dtype=float)
        y_pred = model.predict(X_test)

        # Write predictions back by index.
        out.loc[test_df.index, "y_pred"] = y_pred

        models[str(asset_class)] = ClassModel(
            asset_class=str(asset_class),
            model=model,
            n_train_rows=int(len(train_df)),
            n_test_rows=int(len(test_df)),
            cutoff_date=pd.to_datetime(test_df["cutoff_date"].iloc[0]),
        )

    return out, models, failures


def clip_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """STEP 6 — Clip predictions into [0, 100]."""

    out = df.copy()
    out.loc[out["y_pred"].notna(), "y_pred"] = out.loc[out["y_pred"].notna(), "y_pred"].clip(0.0, 100.0)
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
    n_rows_total_by_asset: dict[str, int],
    class_failures: dict[str, str],
) -> pd.DataFrame:
    """STEP 7 — Evaluate per asset_id using predicted test rows only."""

    rows: list[dict[str, object]] = []

    for asset_id, g in df_pred.groupby("asset_id", sort=False):
        g = g.sort_values(["date"], kind="mergesort")
        asset_id_str = str(asset_id)
        asset_class = str(g["asset_class"].iloc[0])

        row: dict[str, object] = {
            "asset_id": asset_id_str,
            "asset_class": asset_class,
            "status": "ok",
            "reason": "",
            "n_rows_total": int(n_rows_total_by_asset.get(asset_id_str, len(g))),
            "n_train_rows": int((g["split"] == "train").sum()),
            "n_gap_rows": int((g["split"] == "gap").sum()),
            "n_test_rows": int((g["split"] == "test").sum()),
        }

        gt = g[(g["split"] == "test") & (g["y_pred"].notna())].copy()
        if len(gt) < 3:
            row["status"] = "skipped"
            row["reason"] = class_failures.get(asset_class, "no_predictions")
            rows.append(row)
            continue

        pred = pd.to_numeric(gt["y_pred"], errors="coerce").to_numpy(dtype=float)
        actual = pd.to_numeric(gt["y"], errors="coerce").to_numpy(dtype=float)
        current = pd.to_numeric(gt["stoch_k"], errors="coerce").to_numpy(dtype=float)

        ok = np.isfinite(pred) & np.isfinite(actual) & np.isfinite(current)
        n_ok = int(ok.sum())
        n_eval = int(len(gt))

        if n_ok < 3:
            row["status"] = "skipped"
            row["reason"] = "too_few_finite_predictions"
            rows.append(row)
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

        rows.append(row)

    return pd.DataFrame(rows)


def build_class_summary(results: pd.DataFrame) -> pd.DataFrame:
    """STEP 8 — Per-class summary over successfully modeled assets."""

    if results.empty:
        return pd.DataFrame(
            columns=[
                "asset_class",
                "n_assets_modeled",
                "avg_corr",
                "avg_trend_accuracy",
                "avg_pred_ok_rate",
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
            ]
        )

    summary = (
        ok.groupby("asset_class", dropna=False)
        .agg(
            n_assets_modeled=("asset_id", "nunique"),
            avg_corr=("corr_pred_vs_actual", "mean"),
            avg_trend_accuracy=("trend_accuracy", "mean"),
            avg_pred_ok_rate=("pred_ok_rate", "mean"),
        )
        .reset_index()
        .sort_values(["asset_class"], kind="mergesort")
    )

    return summary


def main() -> int:
    default_input = Path("data") / "data_15mins" / "stoch_features_15min.parquet"
    default_output_dir = Path("results_stoch") / "model3_xgb"

    parser = argparse.ArgumentParser(description="Per-class XGBoost to predict stoch_k(t+h)")
    parser.add_argument("-i", "--input", default=str(default_input), help="Input parquet path")
    parser.add_argument("-o", "--output-dir", default=str(default_output_dir), help="Output directory")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon in steps")
    parser.add_argument("--train-frac", type=float, default=0.80, help="Train fraction (time-based per class)")

    args = parser.parse_args()

    horizon = int(args.horizon)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        "lag1",
        "lag2",
        "lag3",
        "lag5",
        "lag10",
        "delta1",
        "delta2",
        "mr",
        "is_low",
        "is_high",
        "mr_low",
        "mr_high",
    ]

    xgb_params: dict[str, object] = {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1,
    }

    print("STEP 1 — Load data")
    print("=" * 80)
    df = load_dataset(args.input)
    print(f"Input shape: {df.shape}")
    n_rows_total_by_asset = df.groupby("asset_id").size().astype(int).to_dict()

    print("STEP 2 — Feature engineering (per asset_id)")
    print("=" * 80)
    df_feat = add_features_per_asset(df)

    print("STEP 3 — Build target y=stoch_k(t+h)")
    print("=" * 80)
    df_tgt = add_target_per_asset(df_feat, horizon=horizon)

    print("STEP 3 — Drop NaN rows")
    print("=" * 80)
    df_sup = build_supervised(df_tgt, feature_cols=feature_cols)
    print(f"Supervised rows: {df_sup.shape}")

    print("STEP 4 — Split train/test per class (no leakage)")
    print("=" * 80)
    df_split = assign_split_per_class(df_sup, train_frac=float(args.train_frac))
    n_train = int((df_split["split"] == "train").sum())
    n_gap = int((df_split["split"] == "gap").sum())
    n_test = int((df_split["split"] == "test").sum())
    print(f"Rows: train={n_train} gap={n_gap} test={n_test}")

    print("STEP 4 — Train per-class XGBoost")
    print("=" * 80)
    df_pred, models, failures = fit_predict_per_class(df_split, feature_cols=feature_cols, xgb_params=xgb_params)
    print(f"Class models fitted: {len(models)}")
    if failures:
        print(f"Class fit failures: {len(failures)}")

    print("STEP 5/6 — Predict and clip")
    print("=" * 80)
    df_pred = clip_predictions(df_pred)

    print("STEP 7 — Evaluation per asset")
    print("=" * 80)
    results = evaluate_per_asset(
        df_pred,
        n_rows_total_by_asset={str(k): int(v) for k, v in n_rows_total_by_asset.items()},
        class_failures=failures,
    )
    summary = build_class_summary(results)

    print("STEP 8 — Save results")
    print("=" * 80)
    results_path = out_dir / "stoch_xgb_results.csv"
    summary_path = out_dir / "stoch_xgb_summary.csv"

    results.to_csv(results_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
