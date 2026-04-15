"""Model 4 — Per-class supervised forecasting for macd(t+15).

Goal
----
Build ONE supervised regression model per asset_class to predict:

    macd_target = macd(t+15)

Features combine:
- structural information: ema12, ema26
- temporal information: macd lags
- jump/regime information: gap_flag derived from time gaps

Dataset
-------
Input: macd_features_15min.parquet
Expected columns:
- asset_id, asset_class, date
- ema12, ema26, macd
- (optional) macd_signal, macd_hist

Constraints
-----------
- Create lags/targets per asset_id (no mixing across assets).
- Respect time order.
- Avoid leakage: training examples must not use targets from the test period.

Models
------
Two versions per class:
1) Ridge regression
2) XGBoost regressor if available; otherwise GradientBoostingRegressor fallback

Outputs
-------
Writes 4 CSVs to the output directory:
- model4_ridge_results.csv
- model4_ridge_summary.csv
- model4_xgb_results.csv
- model4_xgb_summary.csv

"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "ema12",
    "ema26",
    "macd_lag1",
    "macd_lag2",
    "macd_lag3",
    "gap_flag",
]


@dataclass(frozen=True)
class SplitInfo:
    cutoff_date: pd.Timestamp
    n_train: int
    n_test: int


def load_dataset(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    required = {"asset_id", "asset_class", "date", "ema12", "ema26", "macd"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return df


def feature_engineering(
    df: pd.DataFrame,
    horizon: int = 15,
    lags: Iterable[int] = (1, 2, 3),
    gap_minutes: int = 30,
) -> pd.DataFrame:
    """Build per-asset lags, regime flag, and target.

    Returns a row-wise modeling dataset where each row corresponds to an input time t
    and the target is macd(t+horizon).
    """

    parts: list[pd.DataFrame] = []
    gap_td = pd.Timedelta(minutes=int(gap_minutes))

    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values(["date"], kind="mergesort").copy()

        # Regime flag from time gaps.
        time_gap = g["date"].diff()
        g["gap_flag"] = (time_gap > gap_td).fillna(False).astype(int)

        # Lags.
        for lag in lags:
            g[f"macd_lag{lag}"] = g["macd"].shift(int(lag))

        # Target and its timestamp (for leakage-safe splitting).
        g["target_date"] = g["date"].shift(-int(horizon))
        g["macd_target"] = g["macd"].shift(-int(horizon))

        required_cols = [
            "asset_id",
            "asset_class",
            "date",
            "ema12",
            "ema26",
            "macd",
            "gap_flag",
            "target_date",
            "macd_target",
        ] + [f"macd_lag{lag}" for lag in lags]

        g = g.dropna(subset=required_cols)
        g = g[required_cols]
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["asset_class", "date", "asset_id"], kind="mergesort").reset_index(drop=True)
    return out


def time_split_per_class(df_class: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
    """Split by time (first 80% train, last 20% test) and avoid leakage.

    Leakage rule:
    - Training rows must have target_date <= cutoff_date.
    - Test rows are strictly after cutoff_date (by date).
    """

    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0,1)")

    df_class = df_class.sort_values(["date", "asset_id"], kind="mergesort").reset_index(drop=True)
    n = len(df_class)
    if n < 20:
        raise ValueError("Too few rows to split")

    split_idx = int(math.floor(n * float(train_frac)))
    split_idx = max(1, min(split_idx, n - 1))

    cutoff_date = pd.Timestamp(df_class["date"].iloc[split_idx - 1])

    train = df_class[(df_class["date"] <= cutoff_date) & (df_class["target_date"] <= cutoff_date)].copy()
    test = df_class[df_class["date"] > cutoff_date].copy()

    info = SplitInfo(cutoff_date=cutoff_date, n_train=int(len(train)), n_test=int(len(test)))
    return train, test, info


def _corr(pred: np.ndarray, actual: np.ndarray) -> float:
    if len(pred) < 3:
        return float("nan")
    if np.nanstd(pred) < 1e-12 or np.nanstd(actual) < 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, actual)[0, 1])


def evaluate_per_asset(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Compute per-asset correlation and trend accuracy on the test split."""

    if len(df_test) != len(y_pred):
        raise ValueError("Length mismatch between df_test and y_pred")

    tmp = df_test[["asset_id", "asset_class", "macd", "macd_target"]].copy()
    tmp["y_pred"] = y_pred

    rows: list[dict[str, object]] = []
    for asset_id, g in tmp.groupby("asset_id", sort=False):
        pred = g["y_pred"].to_numpy(dtype=float)
        actual = g["macd_target"].to_numpy(dtype=float)
        current = g["macd"].to_numpy(dtype=float)

        ok = np.isfinite(pred) & np.isfinite(actual) & np.isfinite(current)
        n_total = int(len(g))
        n_ok = int(ok.sum())

        row: dict[str, object] = {
            "asset_id": asset_id,
            "asset_class": str(g["asset_class"].iloc[0]),
            "status": "ok",
            "reason": "",
            "n_test": n_total,
            "n_pred_ok": n_ok,
            "pred_ok_rate": float(n_ok / n_total) if n_total else float("nan"),
            "corr_pred_vs_actual": float("nan"),
            "trend_accuracy": float("nan"),
        }

        if n_ok < 3:
            row["status"] = "skipped"
            row["reason"] = "too_few_test_points"
            rows.append(row)
            continue

        pred_ok = pred[ok]
        actual_ok = actual[ok]
        current_ok = current[ok]

        row["corr_pred_vs_actual"] = _corr(pred_ok, actual_ok)

        pred_dir = np.sign(pred_ok - current_ok)
        actual_dir = np.sign(actual_ok - current_ok)
        row["trend_accuracy"] = float(np.mean(pred_dir == actual_dir))

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_by_class(per_asset: pd.DataFrame) -> pd.DataFrame:
    ok = per_asset[per_asset["status"] == "ok"].copy()
    if len(ok) == 0:
        return pd.DataFrame(
            [
                {
                    "asset_class": "__all__",
                    "n_assets_evaluated": 0,
                    "avg_corr": float("nan"),
                    "avg_trend_accuracy": float("nan"),
                    "avg_pred_ok_rate": float("nan"),
                }
            ]
        )

    grp = (
        ok.groupby("asset_class", dropna=False)
        .agg(
            n_assets_evaluated=("asset_id", "nunique"),
            avg_corr=("corr_pred_vs_actual", "mean"),
            avg_trend_accuracy=("trend_accuracy", "mean"),
            avg_pred_ok_rate=("pred_ok_rate", "mean"),
        )
        .reset_index()
    )

    overall = pd.DataFrame(
        [
            {
                "asset_class": "__all__",
                "n_assets_evaluated": int(ok["asset_id"].nunique()),
                "avg_corr": float(ok["corr_pred_vs_actual"].mean()),
                "avg_trend_accuracy": float(ok["trend_accuracy"].mean()),
                "avg_pred_ok_rate": float(ok["pred_ok_rate"].mean()),
            }
        ]
    )

    return pd.concat([grp, overall], ignore_index=True)


def train_ridge(train_df: pd.DataFrame) -> Pipeline:
    X = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y = train_df["macd_target"].to_numpy(dtype=float)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(X, y)
    return model


def train_xgb_or_fallback(train_df: pd.DataFrame):
    """Train XGBoost if available; else fallback to GradientBoostingRegressor."""

    X = train_df[FEATURE_COLS].to_numpy(dtype=float)
    y = train_df["macd_target"].to_numpy(dtype=float)

    try:
        from xgboost import XGBRegressor  # type: ignore

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=0,
        )
        backend = "xgboost"
    except Exception:
        model = GradientBoostingRegressor(random_state=42)
        backend = "gbrt"

    model.fit(X, y)
    return model, backend


def run_per_class_models(df_model: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ridge_asset_rows: list[pd.DataFrame] = []
    xgb_asset_rows: list[pd.DataFrame] = []

    for asset_class, df_class in df_model.groupby("asset_class", sort=False):
        try:
            train_df, test_df, info = time_split_per_class(df_class, train_frac=train_frac)
        except Exception as e:  # noqa: BLE001
            # If split fails, mark all assets in this class as skipped.
            assets = df_class[["asset_id"]].drop_duplicates()
            base = assets.assign(
                asset_class=asset_class,
                status="skipped",
                reason=f"class_split_failed: {e}",
                n_test=0,
                n_pred_ok=0,
                pred_ok_rate=np.nan,
                corr_pred_vs_actual=np.nan,
                trend_accuracy=np.nan,
            )
            ridge_asset_rows.append(base.copy())
            xgb_asset_rows.append(base.copy())
            continue

        if len(train_df) < 50 or len(test_df) < 10:
            assets = df_class[["asset_id"]].drop_duplicates()
            base = assets.assign(
                asset_class=asset_class,
                status="skipped",
                reason="too_few_rows_after_split",
                n_test=int(len(test_df)),
                n_pred_ok=0,
                pred_ok_rate=np.nan,
                corr_pred_vs_actual=np.nan,
                trend_accuracy=np.nan,
            )
            ridge_asset_rows.append(base.copy())
            xgb_asset_rows.append(base.copy())
            continue

        # Train Ridge
        ridge_model = train_ridge(train_df)
        X_test = test_df[FEATURE_COLS].to_numpy(dtype=float)
        ridge_pred = ridge_model.predict(X_test)
        ridge_asset = evaluate_per_asset(test_df, ridge_pred)
        ridge_asset["model"] = "ridge"
        ridge_asset["class_cutoff_date"] = info.cutoff_date
        ridge_asset["n_train_rows"] = info.n_train
        ridge_asset["n_test_rows"] = info.n_test
        ridge_asset_rows.append(ridge_asset)

        # Train XGB or fallback
        xgb_model, backend = train_xgb_or_fallback(train_df)
        xgb_pred = xgb_model.predict(X_test)
        xgb_asset = evaluate_per_asset(test_df, xgb_pred)
        xgb_asset["model"] = "xgb"
        xgb_asset["model_backend"] = backend
        xgb_asset["class_cutoff_date"] = info.cutoff_date
        xgb_asset["n_train_rows"] = info.n_train
        xgb_asset["n_test_rows"] = info.n_test
        xgb_asset_rows.append(xgb_asset)

    ridge_results = pd.concat(ridge_asset_rows, ignore_index=True) if ridge_asset_rows else pd.DataFrame()
    xgb_results = pd.concat(xgb_asset_rows, ignore_index=True) if xgb_asset_rows else pd.DataFrame()

    ridge_summary = summarize_by_class(ridge_results) if len(ridge_results) else summarize_by_class(pd.DataFrame(columns=["status"]))
    xgb_summary = summarize_by_class(xgb_results) if len(xgb_results) else summarize_by_class(pd.DataFrame(columns=["status"]))

    # Add backend note to xgb summary.
    if len(xgb_results) and "model_backend" in xgb_results.columns:
        backend = str(xgb_results["model_backend"].dropna().unique().tolist()[:1][0]) if xgb_results["model_backend"].notna().any() else "unknown"
        xgb_summary.insert(len(xgb_summary.columns), "model_backend", backend)

    return ridge_results, ridge_summary, xgb_results, xgb_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Model 4: per-class supervised regression for macd(t+15)")
    parser.add_argument("-i", "--input", default="macd_features_15min.parquet", help="Input parquet path")
    parser.add_argument("-o", "--output-dir", default="data/model4_supervised", help="Output directory")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon in steps")
    parser.add_argument("--gap-minutes", type=int, default=30, help="Gap threshold for gap_flag")
    parser.add_argument("--train-frac", type=float, default=0.80, help="Train fraction (by time) per class")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("STEP 1 — Load dataset")
    print("=" * 80)
    df = load_dataset(args.input)
    print(f"Input shape: {df.shape}")

    print("STEP 2/3 — Feature engineering")
    print("=" * 80)
    df_model = feature_engineering(
        df,
        horizon=int(args.horizon),
        lags=(1, 2, 3),
        gap_minutes=int(args.gap_minutes),
    )
    print(f"Modeling rows: {len(df_model)}")

    print("STEP 4/5/6 — Train per-class models, predict, evaluate")
    print("=" * 80)
    ridge_results, ridge_summary, xgb_results, xgb_summary = run_per_class_models(
        df_model,
        train_frac=float(args.train_frac),
    )

    print("STEP 7 — Save outputs")
    print("=" * 80)
    ridge_results_path = out_dir / "model4_ridge_results.csv"
    ridge_summary_path = out_dir / "model4_ridge_summary.csv"
    xgb_results_path = out_dir / "model4_xgb_results.csv"
    xgb_summary_path = out_dir / "model4_xgb_summary.csv"

    ridge_results.to_csv(ridge_results_path, index=False, encoding="utf-8")
    ridge_summary.to_csv(ridge_summary_path, index=False, encoding="utf-8")
    xgb_results.to_csv(xgb_results_path, index=False, encoding="utf-8")
    xgb_summary.to_csv(xgb_summary_path, index=False, encoding="utf-8")

    print(f"Saved: {ridge_results_path}")
    print(f"Saved: {ridge_summary_path}")
    print(f"Saved: {xgb_results_path}")
    print(f"Saved: {xgb_summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
