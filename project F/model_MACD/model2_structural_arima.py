"""Model 2 — Structural MACD forecasting via separate EMA forecasts.

Goal
----
Per asset_id, fit two ARIMA models:
- EMA12(t)  ~ ARIMA(p,d,q)
- EMA26(t)  ~ ARIMA(p,d,q)
Then reconstruct MACD forecast:

    macd_pred(t+h) = ema12_pred(t+h) - ema26_pred(t+h)

We evaluate macd_pred against the actual MACD(t+h) for h=15.

Key points
----------
- Per-asset modeling (NOT per-class).
- Small grid search for (p,d,q) with BIC selection.
- Expanding-window walk-forward evaluation for robustness:
  * pick an initial train window
  * keep the chosen order fixed
  * update the state with newly observed EMA values as we walk forward
  * optionally refit parameters every N steps

Outputs
-------
- model2_structural_results.csv  (per asset)
- model2_structural_summary.csv  (by asset_class + overall)

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


@dataclass(frozen=True)
class ArimaFit:
    order: tuple[int, int, int]
    bic: float
    result: object  # SARIMAXResults


def load_dataset(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    required = {"asset_id", "asset_class", "date", "ema12", "ema26", "macd"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Ensure datetime and stable ordering.
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return df


def _safe_float(x: object) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


def fit_arima(
    y_train: np.ndarray,
    p_values: Iterable[int] = (0, 1, 2, 3),
    d_values: Iterable[int] = (0, 1),
    q_values: Iterable[int] = (0, 1, 2, 3),
    grid_maxiter: int = 60,
    final_maxiter: int = 200,
) -> ArimaFit:
    """Fit ARIMA(p,d,q) on y_train by BIC grid search.

    Returns the best (lowest) BIC model and its fitted results.

    Notes
    -----
    We keep the search space small on purpose to reduce overfitting risk.
    """

    if y_train.ndim != 1:
        raise ValueError("y_train must be 1D")

    if len(y_train) < 30:
        raise ValueError("Too few training points for ARIMA")

    if not np.isfinite(y_train).all():
        raise ValueError("y_train contains NaN/inf")

    # If the series is (near) constant, ARIMA estimation is ill-posed.
    if np.nanstd(y_train) < 1e-12:
        raise ValueError("Training series is (near) constant")

    best_order: Optional[tuple[int, int, int]] = None
    best_bic = float("inf")

    # Grid search with a strict iteration cap.
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = SARIMAX(
                            y_train,
                            order=(p, d, q),
                            trend="n",
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        res = model.fit(
                            disp=False,
                            maxiter=grid_maxiter,
                            optim_score="harvey",
                        )
                    bic = _safe_float(getattr(res, "bic", float("inf")))
                    if math.isfinite(bic) and bic < best_bic:
                        best_bic = bic
                        best_order = (p, d, q)
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
        final_res = final_model.fit(
            disp=False,
            maxiter=final_maxiter,
            optim_score="harvey",
        )

    final_bic = _safe_float(getattr(final_res, "bic", best_bic))
    return ArimaFit(order=best_order, bic=final_bic, result=final_res)


def forecast_series(
    y_full: np.ndarray,
    fit: ArimaFit,
    train_size: int,
    horizon: int,
    refit_every: int = 0,
    refit_maxiter: int = 120,
) -> tuple[np.ndarray, int]:
    """Expanding-window walk-forward forecast for y[t+horizon].

    Parameters
    ----------
    y_full:
        Full series of EMA values for one asset.
    fit:
        Fitted ARIMA results on y_full[:train_size].
    train_size:
        Initial window size (number of observations) used for the initial fit.
    horizon:
        Forecast horizon in steps.
    refit_every:
        If >0, refit parameters every N evaluation steps using expanding data,
        keeping the *same* (p,d,q) order.

    Returns
    -------
    preds:
        Array of length n_eval, where n_eval = n - horizon - (train_size - 1).
        preds[i] corresponds to time t = (train_size - 1) + i.
    t0:
        The starting t index used for evaluation: t0 = train_size - 1.

    Notes
    -----
    This walk-forward approach is more robust than a single end-of-sample forecast.
    It avoids using observations after time t when predicting t+horizon.
    """

    n = int(len(y_full))
    if train_size < 30:
        raise ValueError("train_size too small")
    if train_size > n:
        raise ValueError("train_size cannot exceed series length")
    if horizon < 1:
        raise ValueError("horizon must be >=1")

    t0 = train_size - 1
    last_t = n - horizon - 1
    if last_t < t0:
        raise ValueError("Not enough data for evaluation given train_size and horizon")

    n_eval = last_t - t0 + 1
    preds = np.full(n_eval, np.nan, dtype=float)

    # We keep a rolling results object. It starts with the initial training fit.
    res = fit.result

    for i in range(n_eval):
        t = t0 + i

        try:
            if i > 0:
                # Append the newly observed value y[t] to update the filtered state.
                res = res.append([float(y_full[t])], refit=False)

            # Optional periodic refit (same order) on expanding data.
            if refit_every and i > 0 and (i % refit_every == 0):
                y_up_to_t = y_full[: t + 1]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = SARIMAX(
                        y_up_to_t,
                        order=fit.order,
                        trend="n",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = m.fit(disp=False, maxiter=refit_maxiter)

            fc = res.get_forecast(steps=horizon)
            pm = fc.predicted_mean
            # predicted_mean can be ndarray or pandas Series; always take the last step.
            if hasattr(pm, "iloc"):
                preds[i] = float(pm.iloc[-1])
            else:
                preds[i] = float(np.asarray(pm).ravel()[-1])
        except Exception:  # noqa: BLE001
            preds[i] = np.nan

    return preds, t0


def evaluate(
    macd_full: np.ndarray,
    macd_pred: np.ndarray,
    t0: int,
    horizon: int,
) -> dict[str, float | int]:
    """Compute correlation + trend accuracy for macd_pred vs actual macd(t+h)."""

    n = len(macd_full)
    last_t = n - horizon - 1
    n_eval = last_t - t0 + 1
    if len(macd_pred) != n_eval:
        raise ValueError("macd_pred length does not match evaluation window")

    actual = macd_full[t0 + horizon : last_t + horizon + 1]
    current = macd_full[t0 : last_t + 1]

    ok = np.isfinite(macd_pred) & np.isfinite(actual) & np.isfinite(current)
    n_pred_ok = int(ok.sum())

    if n_pred_ok < 3:
        corr = float("nan")
        trend_acc = float("nan")
    else:
        pred_ok = macd_pred[ok]
        actual_ok = actual[ok]
        current_ok = current[ok]

        # Correlation (guard against constant series).
        if np.nanstd(pred_ok) < 1e-12 or np.nanstd(actual_ok) < 1e-12:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(pred_ok, actual_ok)[0, 1])

        # Trend accuracy.
        pred_dir = np.sign(pred_ok - current_ok)
        actual_dir = np.sign(actual_ok - current_ok)
        trend_acc = float(np.mean(pred_dir == actual_dir))

    return {
        "n_eval": int(n_eval),
        "n_pred_ok": int(n_pred_ok),
        "pred_ok_rate": float(n_pred_ok / n_eval) if n_eval else float("nan"),
        "corr_pred_vs_actual": float(corr),
        "trend_accuracy": float(trend_acc),
    }


def _choose_train_size(n: int, train_frac: float, min_train: int, horizon: int) -> int:
    # Pick a train size that leaves at least 1 evaluation point.
    base = max(int(math.floor(n * train_frac)), min_train)
    base = min(base, n - horizon)  # ensure train_size <= n - horizon
    return int(max(base, 30))


def main() -> int:
    parser = argparse.ArgumentParser(description="Model 2: Structural MACD forecasting (per-asset ARIMA on EMA12/EMA26)")
    parser.add_argument("-i", "--input", default="macd_features_15min.parquet", help="Input parquet path")
    parser.add_argument("-o", "--output-dir", default="data/model2_ema", help="Output directory")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon in steps")
    parser.add_argument("--train-frac", type=float, default=0.60, help="Initial training fraction per asset")
    parser.add_argument("--min-train", type=int, default=200, help="Minimum initial training size per asset")
    parser.add_argument("--min-eval", type=int, default=50, help="Minimum evaluation points required per asset")
    parser.add_argument("--grid-maxiter", type=int, default=60, help="Max iterations per candidate model during grid search")
    parser.add_argument("--final-maxiter", type=int, default=200, help="Max iterations for final refit of selected order")
    parser.add_argument("--refit-every", type=int, default=0, help="Refit parameters every N eval steps (0 disables)")
    parser.add_argument("--refit-maxiter", type=int, default=120, help="Max iterations for periodic refits")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("STEP 1 — Load dataset")
    print("=" * 80)
    df = load_dataset(input_path)
    print(f"Input dataset shape: {df.shape}")
    print("=" * 80)

    horizon = int(args.horizon)

    results_rows: list[dict[str, object]] = []

    print("STEP 2 — Loop over each asset_id")
    print("=" * 80)

    for asset_id, g in df.groupby("asset_id", sort=False):
        asset_class = str(g["asset_class"].iloc[0])
        g = g.sort_values(["date"], kind="mergesort").reset_index(drop=True)

        print(f"Asset={asset_id} class={asset_class} n={len(g)}")

        row: dict[str, object] = {
            "asset_id": asset_id,
            "asset_class": asset_class,
            "status": "ok",
            "reason": "",
            "n_rows_total": int(len(g)),
            "horizon": horizon,
        }

        try:
            ema12 = g["ema12"].to_numpy(dtype=float)
            ema26 = g["ema26"].to_numpy(dtype=float)
            macd = g["macd"].to_numpy(dtype=float)

            if len(g) < (args.min_train + horizon + args.min_eval):
                raise ValueError("Too few rows for requested min_train/min_eval/horizon")

            train_size = _choose_train_size(len(g), float(args.train_frac), int(args.min_train), horizon)
            t0 = train_size - 1
            n_eval = (len(g) - horizon) - t0
            if n_eval < int(args.min_eval):
                raise ValueError(f"Too few evaluation points: n_eval={n_eval}")

            # STEP 3/4 — Fit ARIMA on EMA12 and EMA26 separately (training window only)
            y12_train = ema12[:train_size]
            y26_train = ema26[:train_size]

            fit12 = fit_arima(
                y12_train,
                grid_maxiter=int(args.grid_maxiter),
                final_maxiter=int(args.final_maxiter),
            )
            fit26 = fit_arima(
                y26_train,
                grid_maxiter=int(args.grid_maxiter),
                final_maxiter=int(args.final_maxiter),
            )

            print(
                f"  Selected EMA12 order={fit12.order} BIC={fit12.bic:.2f} | "
                f"EMA26 order={fit26.order} BIC={fit26.bic:.2f} | train_size={train_size}"
            )

            row.update(
                {
                    "ema12_p": fit12.order[0],
                    "ema12_d": fit12.order[1],
                    "ema12_q": fit12.order[2],
                    "ema12_bic": float(fit12.bic),
                    "ema26_p": fit26.order[0],
                    "ema26_d": fit26.order[1],
                    "ema26_q": fit26.order[2],
                    "ema26_bic": float(fit26.bic),
                    "train_size": int(train_size),
                }
            )

            # STEP 5 — Forecast EMA12/EMA26 h steps ahead using walk-forward
            pred12, t0_12 = forecast_series(
                ema12,
                fit12,
                train_size=train_size,
                horizon=horizon,
                refit_every=int(args.refit_every),
                refit_maxiter=int(args.refit_maxiter),
            )
            pred26, t0_26 = forecast_series(
                ema26,
                fit26,
                train_size=train_size,
                horizon=horizon,
                refit_every=int(args.refit_every),
                refit_maxiter=int(args.refit_maxiter),
            )
            if t0_12 != t0_26:
                raise RuntimeError("Internal error: t0 mismatch between EMA forecasts")

            # STEP 6 — Reconstruct MACD
            macd_pred = pred12 - pred26

            # STEP 7 — Evaluation (predicted macd(t+h) vs actual macd(t+h))
            metrics = evaluate(macd, macd_pred, t0=t0_12, horizon=horizon)
            row.update(metrics)

            print(
                f"  Eval: n_eval={metrics['n_eval']} ok_rate={metrics['pred_ok_rate']:.3f} "
                f"corr={metrics['corr_pred_vs_actual']:.3f} trend_acc={metrics['trend_accuracy']:.3f}"
            )

        except Exception as e:  # noqa: BLE001
            row["status"] = "skipped"
            row["reason"] = str(e)

            print(f"  Skipped: {e}")

        results_rows.append(row)

    results = pd.DataFrame(results_rows)

    print("=" * 80)
    ok_results = results[results["status"] == "ok"].copy()
    print(f"Assets modeled: {len(ok_results)} / {len(results)}")

    # STEP 8 — Save outputs
    results_path = out_dir / "model2_structural_results.csv"
    summary_path = out_dir / "model2_structural_summary.csv"

    results.to_csv(results_path, index=False, encoding="utf-8")

    # Summary: by asset_class + overall.
    if len(ok_results) > 0:
        grp = (
            ok_results.groupby("asset_class", dropna=False)
            .agg(
                n_assets_modeled=("asset_id", "nunique"),
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
                    "n_assets_modeled": int(ok_results["asset_id"].nunique()),
                    "avg_corr": float(ok_results["corr_pred_vs_actual"].mean()),
                    "avg_trend_accuracy": float(ok_results["trend_accuracy"].mean()),
                    "avg_pred_ok_rate": float(ok_results["pred_ok_rate"].mean()),
                }
            ]
        )
        summary = pd.concat([grp, overall], ignore_index=True)
    else:
        summary = pd.DataFrame(
            [
                {
                    "asset_class": "__all__",
                    "n_assets_modeled": 0,
                    "avg_corr": float("nan"),
                    "avg_trend_accuracy": float("nan"),
                    "avg_pred_ok_rate": float("nan"),
                }
            ]
        )

    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print("STEP 8 — Save outputs")
    print("=" * 80)
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
