"""Stochastic baseline — per-asset-class ARIMA to forecast stoch_k(t+15).

Dataset
-------
Input: stoch_features_15min.parquet
Required columns:
- asset_id, asset_class, date, stoch_k

Goal
----
Fit ONE ARIMA model per asset_class (not per asset) and evaluate per asset.
Prediction target:

    stoch_k_target = stoch_k(t + horizon)

Constraints
-----------
- Build targets per asset_id (no mixing across assets).
- Respect strict time order.
- Avoid leakage: class model parameters are fitted on training windows only.
- Forecasting per asset uses only that asset's past observations.

Outputs
-------
- stoch_arima_results.csv  (per asset)
- stoch_arima_summary.csv  (per asset_class)

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


REQUIRED_COLS = ["asset_id", "asset_class", "date", "stoch_k"]


def load_dataset(input_path: str | Path) -> pd.DataFrame:
    """STEP 1 — Load dataset and sort by (asset_id, date)."""

    input_path = Path(input_path)
    df = pd.read_parquet(input_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Stable ordering.
    df = df.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return df


def add_target(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """STEP 2 — Build target per asset_id: stoch_k_target = stoch_k shifted by -horizon."""

    parts: list[pd.DataFrame] = []

    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values(["date"], kind="mergesort").copy()
        g["stoch_k_target"] = g["stoch_k"].shift(-int(horizon))
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return out


def _choose_train_size(n: int, train_frac: float, min_train: int, horizon: int) -> int:
    """Choose a per-asset train size while leaving room for horizon forecasts."""

    base = max(int(math.floor(n * float(train_frac))), int(min_train))
    base = min(base, n - int(horizon))
    return int(max(base, 30))


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
    grid_maxiter: int = 10,
    final_maxiter: int = 20,
) -> ArimaFit:
    """STEP 3 — Fit ARIMA(p,d,q) by BIC grid search on y_train."""

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

    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = SARIMAX(
                            y_train,
                            order=(int(p), int(d), int(q)),
                            trend="n",
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            concentrate_scale=True,
                        )
                        res = m.fit(
                            disp=False,
                            method="lbfgs",
                            maxiter=int(grid_maxiter),
                            low_memory=True,
                        )

                    bic = _safe_float(getattr(res, "bic", float("inf")))
                    if math.isfinite(bic) and float(bic) < best_bic:
                        best_bic = float(bic)
                        best_order = (int(p), int(d), int(q))
                except Exception:  # noqa: BLE001
                    continue

    if best_order is None or not math.isfinite(best_bic):
        raise RuntimeError("ARIMA grid search failed for all candidate orders")

    # Refit the best order with a larger iteration cap.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_final = SARIMAX(
            y_train,
            order=best_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True,
        )
        res_final = m_final.fit(
            disp=False,
            method="lbfgs",
            maxiter=int(final_maxiter),
            low_memory=True,
        )

    final_bic = _safe_float(getattr(res_final, "bic", best_bic))
    return ArimaFit(order=best_order, bic=float(final_bic), result=res_final)


def forecast_with_fixed_params(
    y_full: np.ndarray,
    order: tuple[int, int, int],
    params: np.ndarray,
    train_size: int,
    horizon: int,
) -> tuple[np.ndarray, int]:
    """STEP 4 — Walk-forward forecast y(t+horizon) using fixed class-level params.

    This avoids leakage: at time t, the forecast only uses observations <= t.
    """

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

    y_train = y_full[: int(train_size)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SARIMAX(
            y_train,
            order=order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True,
        )
        res = m.filter(params)

    for i in range(n_eval):
        t = t0 + i
        try:
            if i > 0:
                # Update filtered state with observed y[t].
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
    """STEP 5 — Compute correlation + trend accuracy."""

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
    default_input = Path("data") / "data_15mins" / "stoch_features_15min.parquet"

    parser = argparse.ArgumentParser(description="Baseline per-class ARIMA for stoch_k(t+h)")
    parser.add_argument("-i", "--input", default=str(default_input), help="Input parquet path")
    parser.add_argument("-o", "--output-dir", default=".", help="Output directory")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon in steps")

    # Per-asset train window configuration (to avoid leakage).
    parser.add_argument("--train-frac", type=float, default=0.60, help="Training fraction per asset")
    parser.add_argument("--min-train", type=int, default=200, help="Minimum training size per asset")
    parser.add_argument("--min-eval", type=int, default=50, help="Minimum evaluation points required per asset")

    # ARIMA grid search fit budgets.
    parser.add_argument("--grid-maxiter", type=int, default=10, help="Max iterations per candidate (grid)")
    parser.add_argument("--final-maxiter", type=int, default=20, help="Max iterations for final refit")

    args = parser.parse_args()

    horizon = int(args.horizon)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("STEP 1 — Load dataset")
    print("=" * 80)
    df = load_dataset(args.input)
    print(f"Input shape: {df.shape}")

    print("STEP 2 — Build target per asset_id")
    print("=" * 80)
    df = add_target(df, horizon=horizon)

    # Supervised dataset for class aggregation (drop rows without target).
    df_supervised = df.dropna(subset=["stoch_k_target"]).copy()

    # Compute train_size per asset and keep it in a dict.
    train_size_by_asset: dict[str, int] = {}
    skipped_prep: list[dict[str, object]] = []

    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values(["date"], kind="mergesort")
        y = pd.to_numeric(g["stoch_k"], errors="coerce").to_numpy(dtype=float)
        n = int(len(y))

        if n < (int(args.min_train) + int(horizon) + int(args.min_eval)):
            skipped_prep.append({"asset_id": asset_id, "asset_class": str(g["asset_class"].iloc[0]), "reason": "too_few_rows", "n_rows": n})
            continue

        ts = _choose_train_size(n, train_frac=float(args.train_frac), min_train=int(args.min_train), horizon=horizon)
        train_size_by_asset[str(asset_id)] = int(ts)

    print(f"Assets eligible (prep): {len(train_size_by_asset)}")
    if skipped_prep:
        print(f"Assets skipped (prep): {len(skipped_prep)}")

    print("STEP 3 — Fit ARIMA per asset_class (BIC)")
    print("=" * 80)

    class_models: dict[str, ClassModel] = {}
    class_fit_failures: dict[str, str] = {}

    for asset_class, df_c in df_supervised.groupby("asset_class", sort=False):
        # Combine all eligible assets in this class using TRAINING windows only.
        y_parts: list[np.ndarray] = []
        assets_used: list[str] = []

        for asset_id, g in df_c.groupby("asset_id", sort=False):
            asset_id_str = str(asset_id)
            ts = int(train_size_by_asset.get(asset_id_str, 0))
            if ts < 30:
                continue

            g = g.sort_values(["date"], kind="mergesort")
            y = pd.to_numeric(g["stoch_k"], errors="coerce").to_numpy(dtype=float)[:ts]

            if len(y) < 30 or not np.isfinite(y).all():
                continue

            y_parts.append(y)
            assets_used.append(asset_id_str)

        if not y_parts:
            class_fit_failures[str(asset_class)] = "no_training_data_after_filter"
            continue

        y_train_class = np.concatenate(y_parts)
        print(f"Class={asset_class} assets={len(set(assets_used))} train_points={len(y_train_class)}")

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

    print("STEP 4/5 — Forecast per asset and evaluate")
    print("=" * 80)

    results_rows: list[dict[str, object]] = []

    for asset_id, g in df.groupby("asset_id", sort=False):
        asset_class = str(g["asset_class"].iloc[0])
        n_rows = int(len(g))

        row: dict[str, object] = {
            "asset_id": str(asset_id),
            "asset_class": asset_class,
            "status": "ok",
            "reason": "",
            "n_rows_total": n_rows,
            "horizon": int(horizon),
        }

        if str(asset_id) not in train_size_by_asset:
            row["status"] = "skipped"
            row["reason"] = "prep_skip"
            results_rows.append(row)
            continue

        if asset_class not in class_models:
            row["status"] = "skipped"
            row["reason"] = f"no_class_model: {class_fit_failures.get(asset_class, 'unknown')}"
            results_rows.append(row)
            continue

        model = class_models[asset_class]
        row.update(
            {
                "train_size": int(train_size_by_asset[str(asset_id)]),
                "p": int(model.order[0]),
                "d": int(model.order[1]),
                "q": int(model.order[2]),
                "class_bic": float(model.bic),
            }
        )

        try:
            g = g.sort_values(["date"], kind="mergesort").reset_index(drop=True)
            y_full = pd.to_numeric(g["stoch_k"], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(y_full).all():
                raise ValueError("non_finite_values_in_series")

            preds, t0 = forecast_with_fixed_params(
                y_full=y_full,
                order=model.order,
                params=model.params,
                train_size=int(train_size_by_asset[str(asset_id)]),
                horizon=int(horizon),
            )

            metrics = evaluate_predictions(y_full=y_full, y_pred=preds, t0=int(t0), horizon=int(horizon))
            row.update(metrics)

        except Exception as e:  # noqa: BLE001
            row["status"] = "skipped"
            row["reason"] = str(e)

        results_rows.append(row)

    results = pd.DataFrame(results_rows)

    # STEP 5 — Per-class averages over successfully modeled assets.
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

    print("STEP 6 — Save results")
    print("=" * 80)

    results_path = out_dir / "stoch_arima_results.csv"
    summary_path = out_dir / "stoch_arima_summary.csv"

    results.to_csv(results_path, index=False, encoding="utf-8")
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
