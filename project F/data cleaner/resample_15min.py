"""Resample processed2.0 5-min bars to 15-min bars (per asset).

Constraints respected:
- Resample separately for each asset_id.
- Do NOT forward fill / interpolate.
- Do NOT keep empty bins (drop rows with missing open/close).
- Preserve market gaps (no artificial bars across gaps).

Output:
- resampled_15min.parquet

"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def load_dataset(path: str | Path) -> pd.DataFrame:
    """STEP 1 — Load dataset (parquet or csv)."""

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
    return df


def resample_one_asset(g: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """STEP 2/3 — Resample one asset_id to 15-minute OHLCV, drop empty bins."""

    asset_id = str(g["asset_id"].iloc[0])
    asset_class = g["asset_class"].iloc[0]

    # Ensure datetime index
    g = g.sort_values("date", kind="mergesort").set_index("date")

    # STEP 2 — Resample to 15-minute with OHLCV aggregations
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    r = g.resample(freq).agg(agg)

    # STEP 3 — Drop rows where open/close are missing (removes empty bins)
    r = r.dropna(subset=["open", "close"], how="any")

    # Re-attach identifiers
    r["asset_id"] = asset_id
    r["asset_class"] = asset_class

    return r.reset_index()


def resample_all_assets(df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """STEP 2/4/5 — Resample each asset_id separately, combine and sort."""

    parts: list[pd.DataFrame] = []
    for asset_id, g in df.groupby("asset_id", sort=False):
        parts.append(resample_one_asset(g, freq=freq))

    out = pd.concat(parts, ignore_index=True)

    # STEP 5 — Sort by (asset_id, date)
    out = out.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)

    # Keep column order stable
    out = out[["asset_id", "asset_class", "date", "open", "high", "low", "close", "volume"]]
    return out


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """STEP 6 — Save output parquet."""

    path = Path(path)
    df.to_parquet(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resample processed2.0 from 5-min to 15-min per asset")
    parser.add_argument(
        "-i",
        "--input",
        default=str(Path("data") / "processed2.0" / "processed2.0.parquet"),
        help="Input dataset path (.parquet or .csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="resampled_15min.parquet",
        help="Output parquet path",
    )
    parser.add_argument("--freq", default="15min", help="Resample frequency (default: 15min)")

    args = parser.parse_args()

    df = load_dataset(args.input)
    print(f"Loaded: shape={df.shape}")

    # STEP 4/5 — Combine resampled assets and sort
    out = resample_all_assets(df, freq=args.freq)
    print(f"Resampled: shape={out.shape}")

    # STEP 6 — Save
    save_parquet(out, args.output)
    print(f"Saved: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
