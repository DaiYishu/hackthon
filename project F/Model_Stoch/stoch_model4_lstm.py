"""LSTM baseline to predict Stochastic stoch_k(t+15).

Dataset
-------
Input parquet must contain:
- asset_id, asset_class, date, stoch_k

Goal
----
Predict:

    stoch_k(t + horizon)

Sequence building (per asset_id)
--------------------------------
Window size = 20
Input:
    [stoch_k(t-19), ..., stoch_k(t)]
Target:
    stoch_k(t+15)

Train/test split
----------------
Time-based split 80/20 per asset_id (no shuffle).
We split by target index to avoid label leakage:
- train if target_idx < cutoff_idx
- test  if target_idx >= cutoff_idx

Model
-----
- LSTM(32)
- Dropout(0.2)
- Dense(1)
Loss: MSE
Optimizer: Adam

Outputs
-------
- stoch_lstm_results.csv (per asset_id metrics on the test set)

"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


REQUIRED_COLS = ["asset_id", "asset_class", "date", "stoch_k"]

MEAN_LEVEL = 50.0


def load_dataset(input_path: str | Path) -> pd.DataFrame:
    """STEP 1 — Load data and sort by (asset_id, date)."""

    input_path = Path(input_path)
    df = pd.read_parquet(input_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stoch_k"] = pd.to_numeric(df["stoch_k"], errors="coerce")

    df = df.dropna(subset=["asset_id", "asset_class", "date", "stoch_k"]).copy()
    df = df.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return df


@dataclass(frozen=True)
class Sequences:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_meta: pd.DataFrame
    skipped_assets: pd.DataFrame


def build_sequences(
    df: pd.DataFrame,
    window_size: int,
    horizon: int,
    train_frac: float,
) -> Sequences:
    """STEP 2/3 — Build rolling sequences per asset_id and split 80/20 by time."""

    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not (0.0 < float(train_frac) < 1.0):
        raise ValueError("train_frac must be in (0, 1)")

    X_train_list: list[np.ndarray] = []
    y_train_list: list[float] = []
    X_test_list: list[np.ndarray] = []
    y_test_list: list[float] = []
    test_rows: list[dict[str, object]] = []

    skipped: list[dict[str, object]] = []

    for asset_id, g in df.groupby("asset_id", sort=False):
        g = g.sort_values(["date"], kind="mergesort").copy()

        asset_id_str = str(asset_id)
        asset_class = str(g["asset_class"].iloc[0])

        k = pd.to_numeric(g["stoch_k"], errors="coerce").to_numpy(dtype=np.float32)
        dates = pd.to_datetime(g["date"], errors="coerce").to_numpy()

        ok = np.isfinite(k) & pd.notna(dates)
        k = k[ok]
        dates = dates[ok]

        n = int(len(k))
        if n < window_size + horizon + 3:
            skipped.append(
                {
                    "asset_id": asset_id_str,
                    "asset_class": asset_class,
                    "reason": "too_few_rows",
                    "n_rows": n,
                }
            )
            continue

        cutoff_idx = int(math.floor(float(train_frac) * n))
        cutoff_idx = max(1, min(cutoff_idx, n - 1))

        # end_idx is the index of t in [t-19..t]
        for end_idx in range(window_size - 1, n - horizon):
            start_idx = end_idx - window_size + 1
            target_idx = end_idx + horizon

            seq = k[start_idx : end_idx + 1]
            if seq.shape[0] != window_size:
                continue

            y = float(k[target_idx])
            current = float(k[end_idx])

            if not (np.isfinite(seq).all() and np.isfinite(y) and np.isfinite(current)):
                continue

            if target_idx < cutoff_idx:
                X_train_list.append(seq)
                y_train_list.append(y)
            else:
                X_test_list.append(seq)
                y_test_list.append(y)
                test_rows.append(
                    {
                        "asset_id": asset_id_str,
                        "asset_class": asset_class,
                        "date": pd.Timestamp(dates[end_idx]),
                        "date_target": pd.Timestamp(dates[target_idx]),
                        "stoch_k": current,
                        "y": y,
                    }
                )

    if not X_train_list:
        raise RuntimeError("No training sequences built")

    X_train = np.stack(X_train_list, axis=0).astype(np.float32)
    y_train = np.asarray(y_train_list, dtype=np.float32)

    if X_test_list:
        X_test = np.stack(X_test_list, axis=0).astype(np.float32)
        y_test = np.asarray(y_test_list, dtype=np.float32)
    else:
        X_test = np.zeros((0, window_size), dtype=np.float32)
        y_test = np.zeros((0,), dtype=np.float32)

    # Add channel dim: (N, window, 1)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    test_meta = pd.DataFrame(test_rows)
    skipped_assets = pd.DataFrame(skipped)

    return Sequences(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        test_meta=test_meta,
        skipped_assets=skipped_assets,
    )


def build_model(window_size: int) -> tf.keras.Model:
    """STEP 4 — Build LSTM model."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(int(window_size), 1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mse",
    )

    return model


def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    val_frac: float,
    patience: int,
) -> tf.keras.callbacks.History:
    """STEP 5 — Train with early stopping (no shuffle)."""

    if len(X_train) < 100:
        raise ValueError("Too few training samples for LSTM")

    val_size = int(math.floor(float(val_frac) * len(X_train)))
    val_size = max(1, min(val_size, len(X_train) - 1))

    X_tr = X_train[:-val_size]
    y_tr = y_train[:-val_size]
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=int(patience),
        restore_best_weights=True,
    )

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=int(epochs),
        batch_size=int(batch_size),
        shuffle=False,
        callbacks=[cb],
        verbose=1,
    )

    return history


def predict(model: tf.keras.Model, X_test: np.ndarray, batch_size: int) -> np.ndarray:
    """STEP 6/7 — Predict and clip into [0, 100]."""

    if len(X_test) == 0:
        return np.zeros((0,), dtype=np.float32)

    y_pred = model.predict(X_test, batch_size=int(batch_size), verbose=0)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)

    y_pred = np.clip(y_pred, 0.0, 100.0)
    return y_pred


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


def evaluate(test_meta: pd.DataFrame) -> pd.DataFrame:
    """STEP 8 — Per-asset evaluation on test set."""

    rows: list[dict[str, object]] = []

    if test_meta.empty:
        return pd.DataFrame(
            columns=[
                "asset_id",
                "asset_class",
                "status",
                "reason",
                "n_eval",
                "n_pred_ok",
                "pred_ok_rate",
                "corr_pred_vs_actual",
                "trend_accuracy",
            ]
        )

    for asset_id, g in test_meta.groupby("asset_id", sort=False):
        g = g.sort_values(["date"], kind="mergesort")
        asset_class = str(g["asset_class"].iloc[0])

        pred = pd.to_numeric(g["y_pred"], errors="coerce").to_numpy(dtype=float)
        actual = pd.to_numeric(g["y"], errors="coerce").to_numpy(dtype=float)
        current = pd.to_numeric(g["stoch_k"], errors="coerce").to_numpy(dtype=float)

        ok = np.isfinite(pred) & np.isfinite(actual) & np.isfinite(current)
        n_ok = int(ok.sum())
        n_eval = int(len(g))

        row: dict[str, object] = {
            "asset_id": str(asset_id),
            "asset_class": asset_class,
            "status": "ok",
            "reason": "",
            "n_eval": n_eval,
            "n_pred_ok": n_ok,
            "pred_ok_rate": float(n_ok / n_eval) if n_eval else float("nan"),
        }

        if n_ok < 3:
            row["status"] = "skipped"
            row["reason"] = "too_few_finite_predictions"
            row["corr_pred_vs_actual"] = float("nan")
            row["trend_accuracy"] = float("nan")
            rows.append(row)
            continue

        pred_ok = pred[ok]
        actual_ok = actual[ok]
        current_ok = current[ok]

        corr = _safe_corr(pred_ok, actual_ok)

        pred_dir = np.sign(pred_ok - current_ok)
        actual_dir = np.sign(actual_ok - current_ok)
        trend_acc = float(np.mean(pred_dir == actual_dir))

        row["corr_pred_vs_actual"] = float(corr)
        row["trend_accuracy"] = float(trend_acc)

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> int:
    default_input = Path("data") / "data_15mins" / "stoch_features_15min.parquet"
    default_output_dir = Path("results_stoch") / "model4_lstm"

    parser = argparse.ArgumentParser(description="LSTM model for stoch_k(t+h)")
    parser.add_argument("-i", "--input", default=str(default_input), help="Input parquet path")
    parser.add_argument("-o", "--output-dir", default=str(default_output_dir), help="Output directory")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon in steps")
    parser.add_argument("--window-size", type=int, default=20, help="Rolling window size")
    parser.add_argument("--train-frac", type=float, default=0.80, help="Train fraction per asset (time-based)")

    parser.add_argument("--epochs", type=int, default=30, help="Epochs (20-30 recommended)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Validation fraction from training")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Reproducibility
    np.random.seed(int(args.seed))
    tf.random.set_seed(int(args.seed))

    horizon = int(args.horizon)
    window_size = int(args.window_size)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("STEP 1 — Load data")
    print("=" * 80)
    df = load_dataset(args.input)
    print(f"Rows: {len(df)} assets: {df['asset_id'].nunique()} classes: {df['asset_class'].nunique()}")

    print("STEP 2/3 — Build sequences and split train/test")
    print("=" * 80)
    seq = build_sequences(df, window_size=window_size, horizon=horizon, train_frac=float(args.train_frac))
    print(f"Train samples: {len(seq.X_train)}")
    print(f"Test samples:  {len(seq.X_test)}")
    if not seq.skipped_assets.empty:
        print(f"Skipped assets: {len(seq.skipped_assets)}")

    print("STEP 4 — Build model")
    print("=" * 80)
    model = build_model(window_size=window_size)

    print("STEP 5 — Train")
    print("=" * 80)
    _ = train_model(
        model,
        X_train=seq.X_train,
        y_train=seq.y_train,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        val_frac=float(args.val_frac),
        patience=int(args.patience),
    )

    print("STEP 6/7 — Predict and clip")
    print("=" * 80)
    y_pred = predict(model, seq.X_test, batch_size=int(args.batch_size))

    test_meta = seq.test_meta.copy()
    if len(test_meta) != len(y_pred):
        raise RuntimeError("test_meta length mismatch with predictions")
    test_meta["y_pred"] = y_pred.astype(float)

    print("STEP 8 — Evaluation")
    print("=" * 80)
    results = evaluate(test_meta)

    # Add skipped assets (if any) as skipped rows.
    if not seq.skipped_assets.empty:
        add_rows = seq.skipped_assets.copy()
        add_rows["status"] = "skipped"
        add_rows["reason"] = add_rows.get("reason", "skipped")
        add_rows = add_rows.rename(columns={"n_rows": "n_eval"})
        for col in ["n_pred_ok", "pred_ok_rate", "corr_pred_vs_actual", "trend_accuracy"]:
            add_rows[col] = float("nan")
        add_rows = add_rows[[
            "asset_id",
            "asset_class",
            "status",
            "reason",
            "n_eval",
            "n_pred_ok",
            "pred_ok_rate",
            "corr_pred_vs_actual",
            "trend_accuracy",
        ]]
        results = pd.concat([results, add_rows], ignore_index=True)

    results = results.sort_values(["asset_class", "asset_id"], kind="mergesort").reset_index(drop=True)

    print("STEP 9 — Save results")
    print("=" * 80)
    results_path = out_dir / "stoch_lstm_results.csv"
    results.to_csv(results_path, index=False, encoding="utf-8")
    print(f"Saved: {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
