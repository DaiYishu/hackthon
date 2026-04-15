"""Model 3 — ARIMAX with a simple regime (jump/closure) exogenous variable.

Goal
----
Improve ARIMA for MACD by adding an exogenous regime variable that flags market
closures / discontinuities.

Dataset: macd_features_15min.parquet
Required columns: asset_id, asset_class, date, macd

Scope
-----
Apply ONLY to asset classes affected by market closure:
- index
- stock
- forex

Do NOT apply to crypto.

Regime variable
--------------
Per asset_id:
- time_gap = date - date.shift(1)
- regime = 1 if time_gap > 30 minutes else 0

Target
------
Predict macd(t+15).

Model
-----
Per asset_id (selected classes): ARIMAX via statsmodels SARIMAX
- endog: macd
- exog: regime
- order (p,d,q) grid:
  p in [0..3], d in [0..1], q in [0..3]
- select best order by BIC (small search space to limit overfitting)

Forecasting assumption
----------------------
For future steps, we assume regime = 0.

Evaluation
----------
Per asset:
- correlation(pred, actual)
- trend accuracy: sign(pred - macd_current) vs sign(actual - macd_current)

Outputs
-------
- data/model3_arimax/model3_arimax_results.csv
- data/model3_arimax/model3_arimax_summary.csv

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


SELECTED_CLASSES = {"index", "stock", "forex"}


@dataclass(frozen=True)
class ArimaxFit:
    order: tuple[int, int, int]
    bic: float
    result: object  # SARIMAXResults


def load_dataset(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    required = {"asset_id", "asset_class", "date", "macd"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return df


def add_regime_variable(df_asset: pd.DataFrame, gap_minutes: int) -> pd.DataFrame:
    df_asset = df_asset.copy()
    time_gap = df_asset["date"].diff()
    threshold = pd.Timedelta(minutes=int(gap_minutes))
    regime = (time_gap > threshold).astype(int)
    df_asset["regime"] = regime.fillna(0).astype(int)
    return df_asset


def add_target(df_asset: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df_asset = df_asset.copy()
    df_asset["macd_target"] = df_asset["macd"].shift(-int(horizon))
    return df_asset


def _safe_float(x: object) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:  # noqa: BLE001
        return float("nan")


def fit_arimax(
    y_train: np.ndarray,
    x_train: np.ndarray,
    p_values: Iterable[int] = (0, 1, 2, 3),
    d_values: Iterable[int] = (0, 1),
    q_values: Iterable[int] = (0, 1, 2, 3),
    grid_maxiter: int = 60,
    final_maxiter: int = 200,
) -> ArimaxFit:
    """Fit ARIMAX by BIC grid search using SARIMAX.

    Notes
    -----
    - trend is disabled (trend='n') to keep the model simple.
    - enforce_stationarity / invertibility are disabled for robustness.
    - optim_score='harvey' uses an analytic score, often faster than numeric.
    """

    if y_train.ndim != 1:
        raise ValueError("y_train must be 1D")
    if x_train.ndim != 2 or x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train must be 2D with same number of rows as y_train")

    if len(y_train) < 50:
        raise ValueError("Too few training points for ARIMAX")
    if not np.isfinite(y_train).all() or not np.isfinite(x_train).all():
        raise ValueError("Training data contains NaN/inf")
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
                        model = SARIMAX(
                            y_train,
                            exog=x_train,
                            order=(p, d, q),
                            trend="n",
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            concentrate_scale=True,
                        )
                        res = model.fit(
                            disp=False,
                            method="lbfgs",
                            maxiter=int(grid_maxiter),
                            optim_score="harvey",
                            low_memory=True,
                        )
                    bic = _safe_float(getattr(res, "bic", float("inf")))
                    if math.isfinite(bic) and bic < best_bic:
                        best_bic = bic
                        best_order = (p, d, q)
                except Exception:  # noqa: BLE001
                    continue

    if best_order is None or not math.isfinite(best_bic):
        raise RuntimeError("ARIMAX grid search failed for all candidate orders")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model = SARIMAX(
            y_train,
            exog=x_train,
            order=best_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=True,
        )
        final_res = final_model.fit(
            disp=False,
            method="lbfgs",
            maxiter=int(final_maxiter),
            optim_score="harvey",
            low_memory=True,
        )

    final_bic = _safe_float(getattr(final_res, "bic", best_bic))
    return ArimaxFit(order=best_order, bic=final_bic, result=final_res)


def _choose_train_size(n: int, train_frac: float, min_train: int, horizon: int) -> int:
    base = max(int(math.floor(n * train_frac)), int(min_train))
    base = min(base, n - horizon)
    return int(max(base, 50))


def forecast_arimax_walk_forward(
    y_full: np.ndarray,
    x_full: np.ndarray,
    fit: ArimaxFit,
    train_size: int,
    horizon: int,
    future_regime_value: int = 0,
    refit_every: int = 0,
    refit_maxiter: int = 120,
) -> tuple[np.ndarray, int]:
    """Expanding-window walk-forward forecasts for y[t+horizon].

    We append one observation at a time (endog + exog), and each step forecasts
    'horizon' steps ahead with future exog assumed constant (regime=0).
    """

    n = int(len(y_full))
    if x_full.ndim != 2 or x_full.shape[0] != n:
        raise ValueError("x_full must be 2D with same length as y_full")

    t0 = train_size - 1
    last_t = n - horizon - 1
    if last_t < t0:
        raise ValueError("Not enough data for evaluation")

    n_eval = last_t - t0 + 1
    preds = np.full(n_eval, np.nan, dtype=float)

    res = fit.result

    future_exog = np.full((int(horizon), x_full.shape[1]), float(future_regime_value), dtype=float)

    for i in range(n_eval):
        t = t0 + i
        try:
            if i > 0:
                y_new = np.asarray([float(y_full[t])], dtype=float)
                x_new = np.asarray(x_full[t : t + 1], dtype=float)
                res = res.append(y_new, exog=x_new, refit=False)

            if refit_every and i > 0 and (i % int(refit_every) == 0):
                y_up_to_t = y_full[: t + 1]
                x_up_to_t = x_full[: t + 1]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = SARIMAX(
                        y_up_to_t,
                        exog=x_up_to_t,
                        order=fit.order,
                        trend="n",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        concentrate_scale=True,
                    )
                    res = m.fit(
                        disp=False,
                        method="lbfgs",
                        maxiter=int(refit_maxiter),
                        optim_score="harvey",
                        low_memory=True,
                    )

            fc = res.get_forecast(steps=int(horizon), exog=future_exog)
            pm = fc.predicted_mean
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

        if np.nanstd(pred_ok) < 1e-12 or np.nanstd(actual_ok) < 1e-12:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(pred_ok, actual_ok)[0, 1])

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Model 3: ARIMAX with regime exogenous variable (per asset)")
    parser.add_argument("-i", "--input", default="macd_features_15min.parquet", help="Input parquet path")
    parser.add_argument("-o", "--output-dir", default="data/model3_arimax", help="Output directory")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon in steps")
    parser.add_argument("--gap-minutes", type=int, default=30, help="Regime threshold: time_gap > N minutes")
    parser.add_argument("--train-frac", type=float, default=0.60, help="Initial training fraction per asset")
    parser.add_argument("--min-train", type=int, default=200, help="Minimum initial training size per asset")
    parser.add_argument("--min-eval", type=int, default=50, help="Minimum evaluation points per asset")
    parser.add_argument("--grid-maxiter", type=int, default=60, help="Max iterations per candidate model in grid search")
    parser.add_argument("--final-maxiter", type=int, default=200, help="Max iterations for final refit")
    parser.add_argument("--refit-every", type=int, default=0, help="Refit parameters every N eval steps (0 disables)")
    parser.add_argument("--refit-maxiter", type=int, default=120, help="Max iterations for periodic refits")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon = int(args.horizon)

    print("STEP 1 — Load dataset")
    print("=" * 80)
    df = load_dataset(input_path)
    print(f"Input dataset shape: {df.shape}")
    print("=" * 80)

    print("STEP 2/3 — Build regime variable and target per asset")
    print("=" * 80)

    results_rows: list[dict[str, object]] = []

    for asset_id, g in df.groupby("asset_id", sort=False):
        asset_class = str(g["asset_class"].iloc[0])
        g = g.sort_values(["date"], kind="mergesort").reset_index(drop=True)

        row: dict[str, object] = {
            "asset_id": asset_id,
            "asset_class": asset_class,
            "status": "ok",
            "reason": "",
            "n_rows_total": int(len(g)),
            "horizon": horizon,
            "gap_minutes": int(args.gap_minutes),
        }

        if asset_class not in SELECTED_CLASSES:
            row["status"] = "skipped"
            row["reason"] = "excluded_by_scope"
            results_rows.append(row)
            continue

        try:
            g2 = add_regime_variable(g, gap_minutes=int(args.gap_minutes))
            g2 = add_target(g2, horizon=horizon)

            y = g2["macd"].to_numpy(dtype=float)
            x = g2[["regime"]].to_numpy(dtype=float)

            # Quick check: do we have enough rows to evaluate?
            if len(g2) < (int(args.min_train) + horizon + int(args.min_eval)):
                raise ValueError("Too few rows for requested min_train/min_eval/horizon")

            train_size = _choose_train_size(len(g2), float(args.train_frac), int(args.min_train), horizon)
            t0 = train_size - 1
            n_eval = (len(g2) - horizon) - t0
            if n_eval < int(args.min_eval):
                raise ValueError(f"Too few evaluation points: n_eval={n_eval}")

            y_train = y[:train_size]
            x_train = x[:train_size]

            print(f"Asset={asset_id} class={asset_class} n={len(g2)} train_size={train_size} n_eval={n_eval}")

            fit = fit_arimax(
                y_train,
                x_train,
                grid_maxiter=int(args.grid_maxiter),
                final_maxiter=int(args.final_maxiter),
            )
            row.update(
                {
                    "arima_p": fit.order[0],
                    "arima_d": fit.order[1],
                    "arima_q": fit.order[2],
                    "model_bic": float(fit.bic),
                    "train_size": int(train_size),
                    "regime_rate": float(np.mean(x[:, 0] > 0.5)),
                }
            )

            preds, t0_used = forecast_arimax_walk_forward(
                y,
                x,
                fit,
                train_size=train_size,
                horizon=horizon,
                future_regime_value=0,
                refit_every=int(args.refit_every),
                refit_maxiter=int(args.refit_maxiter),
            )

            metrics = evaluate(y, preds, t0=t0_used, horizon=horizon)
            row.update(metrics)

            print(
                f"  Selected order={fit.order} BIC={fit.bic:.2f} | "
                f"corr={metrics['corr_pred_vs_actual']:.3f} trend_acc={metrics['trend_accuracy']:.3f} ok_rate={metrics['pred_ok_rate']:.3f}"
            )

        except Exception as e:  # noqa: BLE001
            row["status"] = "skipped"
            row["reason"] = str(e)
            print(f"Asset={asset_id} skipped: {e}")

        results_rows.append(row)

    results = pd.DataFrame(results_rows)

    # STEP 7 — Save results
    results_path = out_dir / "model3_arimax_results.csv"
    summary_path = out_dir / "model3_arimax_summary.csv"

    results.to_csv(results_path, index=False, encoding="utf-8")

    ok = results[results["status"] == "ok"].copy()
    if len(ok) > 0:
        summary = (
            ok.groupby("asset_class", dropna=False)
            .agg(
                n_assets_modeled=("asset_id", "nunique"),
                avg_corr=("corr_pred_vs_actual", "mean"),
                avg_trend_accuracy=("trend_accuracy", "mean"),
                avg_pred_ok_rate=("pred_ok_rate", "mean"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(
            columns=["asset_class", "n_assets_modeled", "avg_corr", "avg_trend_accuracy", "avg_pred_ok_rate"]
        )

    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print("STEP 7 — Save results")
    print("=" * 80)
    print(f"Saved: {results_path}")
    print(f"Saved: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
