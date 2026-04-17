"""Build Stochastic(14,3,5) features from 15-minute OHLC data (per asset).

Input
-----
resampled_15min.parquet with columns:
- asset_id, asset_class, date, open, high, low, close, volume

Output
------
stoch_features_15min.parquet with columns:
- asset_id, asset_class, date, close, high, low,
  high_14, low_14, stoch_k, stoch_d, stoch_d_slow

Constraints
-----------
- Compute per asset_id (no mixing across assets).
- Respect strict time order.
- Do NOT fill missing values.
- Drop rows where rolling computations are not available.

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

OUTPUT_COLS = [
    "asset_id",
    "asset_class",
    "date",
    "close",
    "high",
    "low",
    "high_14",
    "low_14",
    "stoch_k",
    "stoch_d",
    "stoch_d_slow",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """STEP 1 — Load data and sort by (asset_id, date)."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_parquet(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Keep a stable order.
    df = df.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return df


def compute_stoch_one_asset(g: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """STEP 2–6 — Compute Stochastic features for one asset_id."""

    # Ensure time order within asset.
    g = g.sort_values("date", kind="mergesort").copy()

    # Convert to numeric (safe guard).
    high = pd.to_numeric(g["high"], errors="coerce")
    low = pd.to_numeric(g["low"], errors="coerce")
    close = pd.to_numeric(g["close"], errors="coerce")

    # STEP 2 — Rolling max/min over 14 bars (per asset).
    g["high_14"] = high.rolling(window=int(window), min_periods=int(window)).max()
    g["low_14"] = low.rolling(window=int(window), min_periods=int(window)).min()

    # STEP 3 — %K with division-by-zero handling.
    denom = g["high_14"] - g["low_14"]
    raw_k = 100.0 * (close - g["low_14"]) / denom

    # If denom == 0, set stoch_k = 50.
    g["stoch_k"] = np.where(denom.to_numpy() == 0.0, 50.0, raw_k.to_numpy())

    # STEP 4 — %D (3-period moving average of %K).
    g["stoch_d"] = g["stoch_k"].rolling(window=3, min_periods=3).mean()

    # STEP 5 — Slow D (5-period moving average of %D).
    g["stoch_d_slow"] = g["stoch_d"].rolling(window=5, min_periods=5).mean()

    # STEP 6 — Drop rows with NaN caused by rolling computations.
    g = g.dropna(subset=["high_14", "low_14", "stoch_k", "stoch_d", "stoch_d_slow"], how="any")

    return g


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Compute features per asset_id and combine results."""

    parts: list[pd.DataFrame] = []

    # STEP 2 — groupby(asset_id)
    for asset_id, g in df.groupby("asset_id", sort=False):
        out_g = compute_stoch_one_asset(g)
        if len(out_g) == 0:
            continue
        parts.append(out_g)

    if not parts:
        raise ValueError("No rows left after feature computation; check input data.")

    out = pd.concat(parts, ignore_index=True)

    # STEP 7 — Sort by (asset_id, date)
    out = out.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)

    # STEP 7 — Keep requested columns only.
    out = out[OUTPUT_COLS]

    return out


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """STEP 7 — Save output parquet."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> int:
    default_input = Path("data") / "data_15mins" / "resampled_15min.parquet"
    default_output = Path("data") / "data_15mins" / "stoch_features_15min.parquet"

    parser = argparse.ArgumentParser(description="Build Stochastic(14,3,5) features from 15-minute OHLC per asset")
    parser.add_argument("-i", "--input", default=str(default_input), help="Input parquet path")
    parser.add_argument("-o", "--output", default=str(default_output), help="Output parquet path")

    args = parser.parse_args()

    df = load_dataset(args.input)
    print(f"Loaded: shape={df.shape}")

    out = build_feature_dataset(df)
    print(f"Features built: shape={out.shape}")

    save_dataset(out, args.output)
    print(f"Saved: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
