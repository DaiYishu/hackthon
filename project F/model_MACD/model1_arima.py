"""Exploratory ARIMA at the asset-class level.

Goal
----
Fit ONE ARIMA model per asset_class (not per asset) using standardized MACD.
Then apply the class-level model to each asset in that class to predict:

    target = macd(t+15)

Key ideas
---------
- Normalize MACD within each asset to remove scale differences.
- Fit ARIMA parameters on training windows only (avoid leakage).
- Forecast per asset using ONLY that asset's past observations (no mixing sequences).

Outputs
-------
- per_class_arima_results.csv   (per asset)
- per_class_summary.csv         (per asset_class)
- per_class_arima_skipped_assets.csv

"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---------------------------
# Data loading / preparation
# ---------------------------

REQUIRED_COLS = ["asset_id", "asset_class", "date", "macd"]


def load_dataset(input_path: str | Path) -> pd.DataFrame:
    """STEP 1 — Load dataset and sort by (asset_class, asset_id, date)."""

    df = pd.read_parquet(input_path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # STEP 2 — Stable sort as requested.
    df = df.sort_values(["asset_class", "asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return df


def _choose_train_size(n: int, train_frac: float, min_train: int, horizon: int) -> int:
    """Choose a per-asset training size while leaving room for horizon forecasts."""

    base = max(int(math.floor(n * float(train_frac))), int(min_train))
    base = min(base, n - horizon)  # ensure at least one target exists after training
    return int(max(base, 30))


def prepare_assets(
    df: pd.DataFrame,
    horizon: int,
    train_frac: float,
    min_train: int,
    min_eval: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """STEP 3/4 — Build per-asset standardized MACD and targets.

    For each asset_id:
    - compute train_size
    - standardize macd using training window only: (macd - mean_train) / std_train
    - create macd_target = macd_z shifted by -horizon
    - do NOT fill NaNs

    Returns
    -------
    df_prepared:
        Original rows plus: macd_z, macd_target, train_size
    asset_stats:
        Per-asset stats (mean/std and sizes)
    skipped_assets:
        List of assets skipped during preparation with reasons
    """

    parts: list[pd.DataFrame] = []
    stats_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for asset_id, g in df.groupby("asset_id", sort=False):
        asset_class = str(g["asset_class"].iloc[0])
        g = g.sort_values(["date"], kind="mergesort").copy()

        y = pd.to_numeric(g["macd"], errors="coerce").to_numpy(dtype=float)
        n = int(len(y))

        # Ensure enough observations for a train window + evaluation.
        if n < (int(min_train) + int(horizon) + int(min_eval)):
            skipped_rows.append(
                {
                    "asset_id": asset_id,
                    "asset_class": asset_class,
                    "n_rows": n,
                    "reason": "too_few_rows",
                }
            )
            continue

        train_size = _choose_train_size(n, train_frac=float(train_frac), min_train=int(min_train), horizon=int(horizon))
        y_train = y[:train_size]

        # No NaN/inf in training window.
        if not np.isfinite(y_train).all():
            skipped_rows.append(
                {
                    "asset_id": asset_id,
                    "asset_class": asset_class,
                    "n_rows": n,
                    "reason": "non_finite_values_in_train",
                }
            )
            continue

        std_train = float(np.nanstd(y_train))
        if std_train < 1e-12:
            skipped_rows.append(
                {
                    "asset_id": asset_id,
                    "asset_class": asset_class,
                    "n_rows": n,
                    "reason": "near_constant_train_series",
                }
            )
            continue

        mean_train = float(np.nanmean(y_train))
        macd_z = (y - mean_train) / std_train

        # Do NOT fill; if z-score produces non-finite values, skip the asset.
        if not np.isfinite(macd_z).all():
            skipped_rows.append(
                {
                    "asset_id": asset_id,
                    "asset_class": asset_class,
                    "n_rows": n,
                    "reason": "non_finite_values_after_standardize",
                }
            )
            continue

        g["macd_z"] = macd_z

        # Create target: macd(t+horizon)
        g["macd_target"] = g["macd_z"].shift(-int(horizon))

        # Keep train size per asset for later forecasting.
        g["train_size"] = int(train_size)

        parts.append(g)
        stats_rows.append(
            {
                "asset_id": asset_id,
                "asset_class": asset_class,
                "n_rows": n,
                "train_size": int(train_size),
                "mean_train": float(mean_train),
                "std_train": float(std_train),
            }
        )

    df_prepared = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    asset_stats = pd.DataFrame(stats_rows)
    skipped_assets = pd.DataFrame(skipped_rows)

    # Preserve requested sorting.
    if len(df_prepared) > 0:
        df_prepared = df_prepared.sort_values(["asset_class", "asset_id", "date"], kind="mergesort").reset_index(drop=True)

    return df_prepared, asset_stats, skipped_assets


# ---------------------------
# ARIMA training and forecasting
# ---------------------------


@dataclass(frozen=True)
class ArimaFit:
    order: tuple[int, int, int]
    bic: float
    result: object  # SARIMAXResults


@dataclass(frozen=True)
class ClassModel:
    asset_class: str
    order: tuple[int, int, int]
    bic: float
    params: np.ndarray


def _safe_float(x: object) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


def fit_arima_bic(
    y_train: np.ndarray,
    p_values: Iterable[int] = (0, 1, 2, 3),
    d_values: Iterable[int] = (0, 1),
    q_values: Iterable[int] = (0, 1, 2, 3),
    grid_maxiter: int = 60,
    final_maxiter: int = 200,
) -> ArimaFit:
    """Fit ARIMA(p,d,q) via SARIMAX with a small BIC grid search."""

    if y_train.ndim != 1:
        raise ValueError("y_train must be 1D")
    if len(y_train) < 50:
        raise ValueError("Too few training points for ARIMA")
    if not np.isfinite(y_train).all():
        raise ValueError("Training series contains NaN/inf")
    if np.nanstd(y_train) < 1e-12:
        raise ValueError("Training series is (near) constant")

    best_order: Optional[tuple[int, int, int]] = None
    best_bic = float("inf")

    # Grid search with capped iterations for speed.
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = SARIMAX(
                            y_train,
                            order=(int(p), int(d), int(q)),
                            trend="n",
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        res = model.fit(disp=False, maxiter=int(grid_maxiter), optim_score="harvey")

                    bic = _safe_float(getattr(res, "bic", float("inf")))
                    if math.isfinite(bic) and bic < best_bic:
                        best_bic = float(bic)
                        best_order = (int(p), int(d), int(q))
                except Exception:  # noqa: BLE001
                    continue

    if best_order is None or not math.isfinite(best_bic):
        raise RuntimeError("ARIMA grid search failed for all candidate orders")

    # Refit the selected order with a higher iteration budget.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model = SARIMAX(
            y_train,
            order=best_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        final_res = final_model.fit(disp=False, maxiter=int(final_maxiter), optim_score="harvey")

    final_bic = _safe_float(getattr(final_res, "bic", best_bic))
    return ArimaFit(order=best_order, bic=float(final_bic), result=final_res)


def forecast_with_fixed_params(
    y_full: np.ndarray,
    order: tuple[int, int, int],
    params: np.ndarray,
    train_size: int,
    horizon: int,
) -> tuple[np.ndarray, int]:
    """Forecast y(t+horizon) using fixed ARIMA params (walk-forward, no leakage)."""

    n = int(len(y_full))
    if train_size < 30:
        raise ValueError("train_size too small")
    if train_size > n:
        raise ValueError("train_size cannot exceed series length")

    t0 = int(train_size) - 1
    last_t = n - int(horizon) - 1
    if last_t < t0:
        raise ValueError("Not enough data for evaluation given train_size and horizon")

    n_eval = last_t - t0 + 1
    preds = np.full(n_eval, np.nan, dtype=float)

    # Initialize filtered state on the training window using the class-level parameters.
    y_train = y_full[: int(train_size)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMAX(
            y_train,
            order=order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = m.filter(params)

    # Walk-forward: at each time t, only use observations up to t.
    for i in range(n_eval):
        t = t0 + i
        try:
            if i > 0:
                res = res.append([float(y_full[t])], refit=False)

            fc = res.get_forecast(steps=int(horizon))
            pm = fc.predicted_mean
            if hasattr(pm, "iloc"):
                preds[i] = float(pm.iloc[-1])
            else:
                preds[i] = float(np.asarray(pm).ravel()[-1])
        except Exception:  # noqa: BLE001
            preds[i] = np.nan

    return preds, int(t0)


def evaluate_predictions(
    y_full: np.ndarray,
    y_pred: np.ndarray,
    t0: int,
    horizon: int,
) -> dict[str, float | int]:
    """STEP 7 — Correlation + trend accuracy for predicted vs actual y(t+h)."""

    n = int(len(y_full))
    last_t = n - int(horizon) - 1
    n_eval = last_t - int(t0) + 1

    if len(y_pred) != n_eval:
        raise ValueError("Prediction length does not match evaluation window")

    actual = y_full[int(t0) + int(horizon) : last_t + int(horizon) + 1]
    current = y_full[int(t0) : last_t + 1]

    ok = np.isfinite(y_pred) & np.isfinite(actual) & np.isfinite(current)
    n_ok = int(ok.sum())

    if n_ok < 3:
        corr = float("nan")
        trend_acc = float("nan")
    else:
        pred_ok = y_pred[ok]
        actual_ok = actual[ok]
        current_ok = current[ok]

        if np.nanstd(pred_ok) < 1e-12 or np.nanstd(actual_ok) < 1e-12:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(pred_ok, actual_ok)[0, 1])

        pred_dir = np.sign(pred_ok - current_ok)
        actual_dir = np.sign(actual_ok - current_ok)
        trend_acc = float(np.mean(pred_dir == actual_dir))

    return {
        "n_eval": int(n_eval),
        "n_pred_ok": int(n_ok),
        "pred_ok_rate": float(n_ok / n_eval) if n_eval else float("nan"),
        "corr_pred_vs_actual": float(corr),
        "trend_accuracy": float(trend_acc),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Exploratory per-asset-class ARIMA on standardized MACD")
    parser.add_argument(
        "-i",
        "--input",
        default=str(Path("data") / "data_15mins" / "macd_features_15min.parquet"),
        help="Input parquet path",
    )
    parser.add_argument("-o", "--output-dir", default="results_MACD/per_class_arima_exploratory", help="Output directory")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon in steps")

    # Simple time-based setup per asset: train window then walk-forward.
    parser.add_argument("--train-frac", type=float, default=0.60, help="Training fraction per asset")
    parser.add_argument("--min-train", type=int, default=200, help="Minimum training size per asset")
    parser.add_argument("--min-eval", type=int, default=50, help="Minimum evaluation points required per asset")

    # ARIMA search space and fit budgets.
    parser.add_argument("--grid-maxiter", type=int, default=60, help="Max iterations per candidate (grid)")
    parser.add_argument("--final-maxiter", type=int, default=200, help="Max iterations for final refit")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon = int(args.horizon)

    print("STEP 1 — Load dataset")
    print("=" * 80)
    df = load_dataset(input_path)
    print(f"Input shape: {df.shape}")

    print("STEP 3/4 — Build per-asset standardized MACD + targets")
    print("=" * 80)
    df_prep, asset_stats, skipped_prep = prepare_assets(
        df,
        horizon=horizon,
        train_frac=float(args.train_frac),
        min_train=int(args.min_train),
        min_eval=int(args.min_eval),
    )

    if len(df_prep) == 0:
        raise ValueError("No assets available after preparation")

    print(f"Prepared rows: {len(df_prep)}")
    print(f"Assets prepared: {asset_stats['asset_id'].nunique() if len(asset_stats) else 0}")
    print(f"Assets skipped (prep): {len(skipped_prep)}")

    # STEP 3 — drop rows where target is NaN (supervised dataset)
    df_supervised = df_prep.dropna(subset=["macd_target"]).copy()

    # Build quick lookup for train_size.
    train_size_by_asset = dict(zip(asset_stats["asset_id"], asset_stats["train_size"], strict=False))

    print("STEP 5 — Fit one ARIMA model per asset_class (BIC grid)")
    print("=" * 80)

    class_models: dict[str, ClassModel] = {}
    class_fit_failures: dict[str, str] = {}

    for asset_class, df_c in df_supervised.groupby("asset_class", sort=False):
        # Stack ONLY training windows from all assets in the class.
        y_parts: list[np.ndarray] = []
        assets_in_class = []

        for asset_id, g in df_c.groupby("asset_id", sort=False):
            ts = int(train_size_by_asset.get(asset_id, 0))
            if ts < 30:
                continue

            g = g.sort_values(["date"], kind="mergesort")
            y = g["macd_z"].to_numpy(dtype=float)[:ts]
            if len(y) < 30 or not np.isfinite(y).all():
                continue

            y_parts.append(y)
            assets_in_class.append(asset_id)

        if not y_parts:
            class_fit_failures[str(asset_class)] = "no_training_data_after_filter"
            continue

        y_train_class = np.concatenate(y_parts)

        print(f"Class={asset_class} assets={len(set(assets_in_class))} train_points={len(y_train_class)}")

        try:
            fit = fit_arima_bic(
                y_train_class,
                p_values=(0, 1, 2, 3),
                d_values=(0, 1),
                q_values=(0, 1, 2, 3),
                grid_maxiter=int(args.grid_maxiter),
                final_maxiter=int(args.final_maxiter),
            )

            params = np.asarray(getattr(fit.result, "params"))
            class_models[str(asset_class)] = ClassModel(
                asset_class=str(asset_class),
                order=fit.order,
                bic=float(fit.bic),
                params=params,
            )

            print(f"  Selected order={fit.order} BIC={fit.bic:.2f}")

        except Exception as e:  # noqa: BLE001
            class_fit_failures[str(asset_class)] = str(e)
            print(f"  Failed to fit class model: {e}")

    print("=" * 80)
    print(f"Class models fitted: {len(class_models)}")
    if class_fit_failures:
        print("Class fit failures:")
        for k, v in class_fit_failures.items():
            print(f"- {k}: {v}")

    print("STEP 6/7 — Forecast per asset using its class model + evaluate")
    print("=" * 80)

    results_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for asset_id, g in df_prep.groupby("asset_id", sort=False):
        asset_class = str(g["asset_class"].iloc[0])
        n_rows = int(len(g))

        # If the asset was skipped during preparation, it won't appear in df_prep.
        # So here we only handle assets that made it through preparation.
        if asset_class not in class_models:
            skipped_rows.append(
                {
                    "asset_id": asset_id,
                    "asset_class": asset_class,
                    "n_rows": n_rows,
                    "reason": f"no_class_model: {class_fit_failures.get(asset_class, 'unknown')} ",
                }
            )
            continue

        model = class_models[asset_class]
        g = g.sort_values(["date"], kind="mergesort").reset_index(drop=True)

        y_full = g["macd_z"].to_numpy(dtype=float)
        train_size = int(g["train_size"].iloc[0])

        row: dict[str, object] = {
            "asset_id": asset_id,
            "asset_class": asset_class,
            "status": "ok",
            "reason": "",
            "n_rows_total": n_rows,
            "horizon": int(horizon),
            "train_size": int(train_size),
            "p": int(model.order[0]),
            "d": int(model.order[1]),
            "q": int(model.order[2]),
            "class_bic": float(model.bic),
        }

        try:
            preds, t0 = forecast_with_fixed_params(
                y_full=y_full,
                order=model.order,
                params=model.params,
                train_size=int(train_size),
                horizon=int(horizon),
            )

            metrics = evaluate_predictions(y_full=y_full, y_pred=preds, t0=int(t0), horizon=int(horizon))
            row.update(metrics)

        except Exception as e:  # noqa: BLE001
            row["status"] = "skipped"
            row["reason"] = str(e)

        results_rows.append(row)

    # Merge in the preparation-level skips as well.
    if len(skipped_prep) > 0:
        for r in skipped_prep.to_dict(orient="records"):
            skipped_rows.append(
                {
                    "asset_id": r.get("asset_id"),
                    "asset_class": r.get("asset_class"),
                    "n_rows": r.get("n_rows"),
                    "reason": f"prep_skip: {r.get('reason')}",
                }
            )

    results = pd.DataFrame(results_rows)
    skipped_assets = pd.DataFrame(skipped_rows)

    # STEP 7 — Per-class averages over successfully modeled assets.
    ok = results[results["status"] == "ok"].copy() if len(results) else pd.DataFrame()
    if len(ok) > 0:
        summary = (
            ok.groupby("asset_class", dropna=False)
            .agg(
                n_assets_modeled=("asset_id", "nunique"),
                avg_corr=("corr_pred_vs_actual", "mean"),
                avg_trend_accuracy=("trend_accuracy", "mean"),
                avg_pred_ok_rate=("pred_ok_rate", "mean"),
                p=("p", "first"),
                d=("d", "first"),
                q=("q", "first"),
                class_bic=("class_bic", "first"),
            )
            .reset_index()
            .sort_values(["asset_class"], kind="mergesort")
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "asset_class",
                "n_assets_modeled",
                "avg_corr",
                "avg_trend_accuracy",
                "avg_pred_ok_rate",
                "p",
                "d",
                "q",
                "class_bic",
            ]
        )

    # STEP 8 — Save outputs
    results_path = out_dir / "per_class_arima_results.csv"
    summary_path = out_dir / "per_class_summary.csv"
    skipped_path = out_dir / "per_class_arima_skipped_assets.csv"

    results.to_csv(results_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    if len(skipped_assets) > 0:
        skipped_assets.to_csv(skipped_path, index=False, encoding="utf-8")
    else:
        skipped_assets.head(0).to_csv(skipped_path, index=False, encoding="utf-8")

    print("STEP 8 — Save outputs")
    print("=" * 80)
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {skipped_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
