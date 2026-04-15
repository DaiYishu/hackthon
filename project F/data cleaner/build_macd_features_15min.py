"""Build MACD feature dataset from 15-minute OHLCV bars.

Input
-----
resampled_15min.parquet with columns:
- asset_id, asset_class, date, open, high, low, close, volume

Output
------
macd_features_15min.parquet with columns:
- asset_id, asset_class, date, open, high, low, close, volume,
  ema12, ema26, macd, macd_signal, macd_hist

Constraints
-----------
- Compute indicators separately for each asset_id.
- Respect strict time order.
- No filling / interpolation.
- Drop rows where indicators cannot be computed (keep only real observations).

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "asset_id",
    "asset_class",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

FEATURE_COLS = [
    "ema12",
    "ema26",
    "macd",
    "macd_signal",
    "macd_hist",
]

OUTPUT_COLS = REQUIRED_COLS + FEATURE_COLS


def load_dataset(path: str | Path) -> pd.DataFrame:
    """STEP 1 — Load dataset (parquet or csv) and standardize dtypes."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        # Best-effort: try parquet first, then csv.
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)

    return df


def compute_indicators_one_asset(g: pd.DataFrame) -> pd.DataFrame:
    """STEP 2–5 — Compute EMA + MACD for one asset_id.

    Notes:
    - Uses pandas ewm(adjust=False) as requested.
    - Does NOT fill NaNs.
    - Drops rows where indicators cannot be computed.
    """

    g = g.sort_values("date", kind="mergesort").copy()

    close = pd.to_numeric(g["close"], errors="coerce")

    # STEP 3 — EMA components (per asset, time-ordered)
    g["ema12"] = close.ewm(span=12, adjust=False).mean()
    g["ema26"] = close.ewm(span=26, adjust=False).mean()

    # STEP 4 — MACD and related indicators
    g["macd"] = g["ema12"] - g["ema26"]
    g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
    g["macd_hist"] = g["macd"] - g["macd_signal"]

    # STEP 5 — Clean output: remove rows where indicators cannot be computed
    g = g.dropna(subset=["ema12", "ema26", "macd", "macd_signal", "macd_hist"], how="any")

    return g


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """STEP 2/6/7 — Group by asset_id, compute indicators, then combine and sort."""

    parts: list[pd.DataFrame] = []

    # STEP 2 — group by asset_id (no mixing across assets)
    for asset_id, g in df.groupby("asset_id", sort=False):
        out_g = compute_indicators_one_asset(g)
        if len(out_g) == 0:
            continue
        parts.append(out_g)

    if not parts:
        raise ValueError("No rows left after indicator computation; check input data.")

    out = pd.concat(parts, ignore_index=True)

    # STEP 6 — Keep only requested columns
    out = out[OUTPUT_COLS]

    # STEP 7 — Sort by (asset_id, date)
    out = out.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)

    return out


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """STEP 8 — Save output parquet."""

    path = Path(path)
    df.to_parquet(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build EMA/MACD feature dataset from resampled 15-minute OHLCV")
    parser.add_argument(
        "-i",
        "--input",
        default="resampled_15min.parquet",
        help="Input dataset path (.parquet or .csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="macd_features_15min.parquet",
        help="Output parquet path",
    )

    args = parser.parse_args()

    df = load_dataset(args.input)
    print(f"Loaded: shape={df.shape}")

    out = build_feature_dataset(df)
    print(f"Features built: shape={out.shape}")

    # Quick sanity check: no NaNs in feature columns
    n_missing_features = int(out[FEATURE_COLS].isna().any(axis=1).sum())
    print(f"Rows with missing feature values (should be 0): {n_missing_features}")

    save_dataset(out, args.output)
    print(f"Saved: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
