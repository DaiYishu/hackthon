from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------
# Asset universe (given)
# -----------------------------
CRYPTO = ["BTCUSD", "ETHUSD", "SOLUSD"]
FOREX = ["EURUSD", "USDJPY", "USDCNY"]
INDICES = ["^GSPC", "^IXIC", "^DJI", "^FCHI", "^NDX"]
STOCKS = ["NVDA", "AAPL", "GOOG", "MC.PA"]

ASSET_CLASS_MAP: dict[str, str] = {
    **{a: "crypto" for a in CRYPTO},
    **{a: "forex" for a in FOREX},
    **{a: "index" for a in INDICES},
    **{a: "stock" for a in STOCKS},
}

REQUIRED_COLUMNS = ["asset_id", "date", "open", "high", "low", "close", "volume"]


def load_csv(path: Path) -> pd.DataFrame:
    """Load a pre-concatenated CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv input file, got: {path.name}")

    return pd.read_csv(path, low_memory=False)


def print_structure(df: pd.DataFrame, title: str) -> None:
    """Print basic dataset structure info."""
    print("=" * 80)
    print(title)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Dtypes:")
    # Using to_string for compact, notebook-friendly printing
    print(df.dtypes.to_string())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names (strip + lower) without changing data."""
    df = df.copy()
    new_cols = [str(c).strip().lower() for c in df.columns]
    if len(set(new_cols)) != len(new_cols):
        dupes = pd.Series(new_cols).value_counts()
        dupes = dupes[dupes > 1]
        raise ValueError(
            "Duplicate columns after normalization: "
            + ", ".join([f"{k} (x{v})" for k, v in dupes.items()])
        )
    df.columns = new_cols
    return df


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize key dtypes: asset_id to string, date to datetime, OHLCV to numeric."""
    df = df.copy()

    # Keep missing asset_id as <NA> (do NOT fill); strip spaces for robustness.
    df["asset_id"] = df["asset_id"].astype("string").str.strip()

    # Parse timestamps; invalid parses become NaT (reported later).
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def sort_by_asset_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by (asset_id, date) and reset index."""
    df = df.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return df


def add_asset_class(df: pd.DataFrame) -> pd.DataFrame:
    """Add asset_class column from the provided mapping."""
    df = df.copy()
    df["asset_class"] = df["asset_id"].map(ASSET_CLASS_MAP).fillna("unknown")
    return df


def compute_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute row-level quality flags without modifying the input dataframe."""
    flags = pd.DataFrame(index=df.index)

    # Exact duplicates: mark rows that would be removed by drop_duplicates(keep='first')
    flags["is_exact_duplicate"] = df.duplicated(keep="first")

    # Invalid/missing timestamp after parsing
    flags["is_invalid_date"] = df["date"].isna()

    # Duplicated timestamps within each asset (exclude missing asset_id/date)
    mask_valid_key = df["asset_id"].notna() & df["date"].notna()
    dup_ts = pd.Series(False, index=df.index)
    if mask_valid_key.any():
        dup_ts.loc[mask_valid_key] = df.loc[mask_valid_key].duplicated(
            subset=["asset_id", "date"], keep=False
        )
    flags["is_duplicate_timestamp"] = dup_ts

    # Missing values in required columns (row-level)
    flags["has_missing_required"] = df[REQUIRED_COLUMNS].isna().any(axis=1)

    # OHLC logical validity (only evaluates where values are present; NaNs are handled by missing checks)
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    close = df["close"]

    flags["ohlc_high_lt_low"] = (high < low) & high.notna() & low.notna()
    flags["ohlc_open_outside_range"] = (
        ((open_ < low) | (open_ > high))
        & open_.notna()
        & low.notna()
        & high.notna()
    )
    flags["ohlc_close_outside_range"] = (
        ((close < low) | (close > high))
        & close.notna()
        & low.notna()
        & high.notna()
    )
    flags["is_ohlc_invalid"] = (
        flags["ohlc_high_lt_low"]
        | flags["ohlc_open_outside_range"]
        | flags["ohlc_close_outside_range"]
    )

    # Negative volume
    flags["is_negative_volume"] = (df["volume"] < 0) & df["volume"].notna()

    # Unknown asset class (unmapped asset_id)
    flags["is_unknown_asset"] = df["asset_id"].notna() & (df["asset_class"] == "unknown")

    return flags


def build_quality_report_by_asset(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Aggregate quality diagnostics by asset_id."""
    # Grouping: include missing asset_id as its own bucket in the report
    asset_key = df["asset_id"].fillna("<MISSING_ASSET_ID>")

    report = (
        pd.concat(
            [
                asset_key.rename("asset_id"),
                df["asset_class"],
                df["date"],
                flags[
                    [
                        "is_exact_duplicate",
                        "is_duplicate_timestamp",
                        "has_missing_required",
                        "is_invalid_date",
                        "is_ohlc_invalid",
                        "is_negative_volume",
                        "is_unknown_asset",
                    ]
                ],
            ],
            axis=1,
        )
        .groupby(["asset_id", "asset_class"], dropna=False)
        .agg(
            n_rows=("asset_id", "size"),
            n_exact_duplicate_rows=("is_exact_duplicate", "sum"),
            n_duplicate_timestamp_rows=("is_duplicate_timestamp", "sum"),
            n_rows_missing_required=("has_missing_required", "sum"),
            n_rows_invalid_date=("is_invalid_date", "sum"),
            n_rows_invalid_ohlc=("is_ohlc_invalid", "sum"),
            n_rows_negative_volume=("is_negative_volume", "sum"),
            n_rows_unknown_asset=("is_unknown_asset", "sum"),
            min_date=("date", "min"),
            max_date=("date", "max"),
        )
        .reset_index()
        .sort_values(["asset_class", "asset_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    return report


def build_suspicious_rows(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Collect suspicious rows with explicit boolean flags and a compact issue label."""
    flag_cols = [
        "is_exact_duplicate",
        "is_duplicate_timestamp",
        "has_missing_required",
        "is_invalid_date",
        "is_ohlc_invalid",
        "is_negative_volume",
        "is_unknown_asset",
    ]

    any_issue = flags[flag_cols].any(axis=1)
    suspicious = df.loc[any_issue].copy()

    for c in flag_cols:
        suspicious[c] = flags.loc[any_issue, c].astype(bool).values

    def _issue_label(row: pd.Series) -> str:
        issues = [c for c in flag_cols if bool(row[c])]
        return ";".join(issues)

    suspicious["issues"] = suspicious.apply(_issue_label, axis=1)

    # Put the issues columns first for easier scanning
    lead_cols = ["issues"] + flag_cols
    remaining = [c for c in suspicious.columns if c not in lead_cols]
    suspicious = suspicious[lead_cols + remaining]

    return suspicious


def print_quality_summary(df: pd.DataFrame, flags: pd.DataFrame) -> None:
    """Print a clear summary of Step 4 checks."""
    print("=" * 80)
    print("STEP 4 — Basic data quality checks")

    n_rows = len(df)
    n_exact_dupes = int(flags["is_exact_duplicate"].sum())
    n_dup_ts = int(flags["is_duplicate_timestamp"].sum())
    n_missing_required = int(flags["has_missing_required"].sum())
    n_invalid_date = int(flags["is_invalid_date"].sum())
    n_invalid_ohlc = int(flags["is_ohlc_invalid"].sum())
    n_neg_vol = int(flags["is_negative_volume"].sum())
    n_unknown_asset = int(flags["is_unknown_asset"].sum())

    print(f"Total rows: {n_rows:,}")
    print(f"Exact duplicate rows (would be removed): {n_exact_dupes:,}")
    print(f"Duplicated timestamps within asset_id: {n_dup_ts:,}")
    print(f"Rows missing any required field: {n_missing_required:,}")
    print(f"Rows with invalid/unparsed date: {n_invalid_date:,}")
    print(f"Rows failing OHLC logic checks: {n_invalid_ohlc:,}")
    print(f"Rows with negative volume: {n_neg_vol:,}")
    print(f"Rows with unknown/unmapped asset_id: {n_unknown_asset:,}")

    print("\nMissing values by column:")
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (missing / max(len(df), 1) * 100).round(4)
    missing_tbl = pd.DataFrame({"missing": missing, "missing_pct": missing_pct})
    print(missing_tbl.to_string())


def compute_time_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Add time_gap = diff(date) within each asset_id. Keeps true gaps (no filling)."""
    df = df.copy()
    # Only compute gaps where asset_id is present; missing asset_id rows keep NaT.
    time_gap = pd.Series(pd.NaT, index=df.index, dtype="timedelta64[ns]")
    mask = df["asset_id"].notna()
    if mask.any():
        time_gap.loc[mask] = df.loc[mask].groupby("asset_id")["date"].diff()
    df["time_gap"] = time_gap
    return df


def summarize_gaps(
    df_with_gaps: pd.DataFrame, *, group_col: str, top_n: int = 10
) -> pd.DataFrame:
    """Summarize most frequent time gaps and counts of gaps > 5 minutes.

    Returns a long table of top-N gap values per group.
    """
    if group_col not in df_with_gaps.columns:
        raise KeyError(f"group_col not found in dataframe: {group_col}")

    expected = pd.Timedelta(minutes=5)

    base = df_with_gaps.copy()
    base = base.loc[base["time_gap"].notna()].copy()

    # Aggregate gap counts
    gap_counts = (
        base.groupby([group_col, "time_gap"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )

    # Total gaps and >5m gaps per group
    gap_stats = (
        base.assign(is_gt_5m=base["time_gap"] > expected)
        .groupby(group_col, dropna=False)
        .agg(total_gaps=("time_gap", "size"), n_gaps_gt_5m=("is_gt_5m", "sum"))
        .reset_index()
    )
    gap_stats["pct_gaps_gt_5m"] = (
        gap_stats["n_gaps_gt_5m"] / gap_stats["total_gaps"].replace(0, np.nan)
    ).fillna(0.0)

    gap_counts = gap_counts.merge(gap_stats, on=group_col, how="left")
    gap_counts["pct_of_gaps"] = gap_counts["count"] / gap_counts["total_gaps"].replace(0, np.nan)
    gap_counts["pct_of_gaps"] = gap_counts["pct_of_gaps"].fillna(0.0)

    # For readability in CSV
    gap_counts["time_gap_minutes"] = gap_counts["time_gap"].dt.total_seconds() / 60.0

    # Keep only top-N gap values per group
    gap_counts = gap_counts.sort_values([group_col, "count"], ascending=[True, False])
    gap_counts = gap_counts.groupby(group_col, dropna=False).head(top_n).reset_index(drop=True)

    return gap_counts


def build_cleaned_dataset(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Create cleaned dataframe:

    - Keeps original rows
    - Removes only exact duplicates (rows flagged is_exact_duplicate)
    - Preserves true market gaps (no fill/interpolation)
    - Adds time_gap
    """
    cleaned = df.loc[~flags["is_exact_duplicate"]].copy()
    cleaned = compute_time_gaps(cleaned)

    # Column order: required + asset_class + time_gap first; keep any extra columns after
    lead = [
        "asset_id",
        "asset_class",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "time_gap",
    ]
    extras = [c for c in cleaned.columns if c not in lead]
    cleaned = cleaned[lead + extras]

    return cleaned


def save_outputs(
    *,
    out_dir: Path,
    cleaned: pd.DataFrame,
    quality_report: pd.DataFrame,
    suspicious_rows: pd.DataFrame,
    gap_by_asset: pd.DataFrame,
    gap_by_class: pd.DataFrame,
    save_parquet: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) cleaned dataset
    cleaned_csv = out_dir / "cleaned_dataset.csv"
    cleaned.to_csv(cleaned_csv, index=False)

    if save_parquet:
        cleaned_parquet = out_dir / "cleaned_dataset.parquet"
        try:
            cleaned.to_parquet(cleaned_parquet, index=False)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] Failed to save Parquet ({cleaned_parquet.name}): {e}")

    # 2) quality report by asset
    quality_report.to_csv(out_dir / "quality_report_by_asset.csv", index=False)

    # 3) suspicious rows
    suspicious_rows.to_csv(out_dir / "suspicious_rows.csv", index=False)

    # 4) gap summary by asset
    gap_by_asset.to_csv(out_dir / "gap_summary_by_asset.csv", index=False)

    # 5) gap summary by asset_class
    gap_by_class.to_csv(out_dir / "gap_summary_by_asset_class.csv", index=False)

    print("=" * 80)
    print("STEP 7 — Saved outputs to:", str(out_dir))
    print("- cleaned_dataset.csv" + (" (+ parquet)" if save_parquet else ""))
    print("- quality_report_by_asset.csv")
    print("- suspicious_rows.csv")
    print("- gap_summary_by_asset.csv")
    print("- gap_summary_by_asset_class.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess a pre-concatenated 5-minute financial time series dataset. "
            "This script preserves true market gaps (no forward-fill, no interpolation, no artificial rows)."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Path to input CSV (must include asset_id,date,open,high,low,close,volume).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=Path("data/processed"),
        type=Path,
        help="Directory to write outputs (default: data/processed).",
    )
    parser.add_argument(
        "--top-gaps",
        type=int,
        default=10,
        help="Top-N most frequent gap values to keep per asset / asset_class (default: 10).",
    )
    parser.add_argument(
        "--save-parquet",
        action="store_true",
        help="Also save cleaned dataset as Parquet (requires pyarrow or fastparquet).",
    )

    args = parser.parse_args()

    # STEP 1 — Load
    print("STEP 1 — Load the dataset")
    df_raw = load_csv(args.input)
    print_structure(df_raw, title="Raw dataset (as loaded)")

    # STEP 2 — Standardize and sort
    print("=" * 80)
    print("STEP 2 — Standardize and sort")
    df = normalize_columns(df_raw)
    ensure_required_columns(df)
    df = standardize_types(df)
    df = sort_by_asset_and_time(df)
    print("Sorted by (asset_id, date) and reset index.")

    # STEP 3 — Add asset_class
    print("=" * 80)
    print("STEP 3 — Add asset_class")
    df = add_asset_class(df)
    n_unknown = int((df["asset_class"] == "unknown").sum())
    if n_unknown:
        unknown_assets = (
            df.loc[df["asset_class"] == "unknown", "asset_id"]
            .dropna()
            .astype(str)
            .value_counts()
            .head(20)
        )
        print(f"Unmapped asset_id rows: {n_unknown:,}")
        print("Top unmapped asset_ids (up to 20):")
        print(unknown_assets.to_string())
    else:
        print("All asset_id values mapped to an asset_class.")

    # STEP 4 — Basic data quality checks
    flags = compute_quality_flags(df)
    print_quality_summary(df, flags)

    quality_report = build_quality_report_by_asset(df, flags)
    suspicious_rows = build_suspicious_rows(df, flags)

    # STEP 5 — Time structure analysis (on dataset with exact duplicates removed)
    print("=" * 80)
    print("STEP 5 — Time structure analysis")
    print("Note: gap analysis uses the dataset with exact duplicates removed (keeping first occurrence).")

    df_for_gaps = df.loc[~flags["is_exact_duplicate"]].copy()
    df_for_gaps = compute_time_gaps(df_for_gaps)

    gap_by_asset = summarize_gaps(df_for_gaps, group_col="asset_id", top_n=args.top_gaps)
    gap_by_class = summarize_gaps(df_for_gaps, group_col="asset_class", top_n=args.top_gaps)

    # Console-friendly preview
    print("\nMost frequent gaps by asset_id (top rows):")
    print(gap_by_asset.head(20).to_string(index=False))
    print("\nMost frequent gaps by asset_class:")
    print(gap_by_class.to_string(index=False))

    # STEP 6 — Output cleaned dataset
    print("=" * 80)
    print("STEP 6 — Output cleaned dataset")
    cleaned = build_cleaned_dataset(df, flags)
    print(f"Cleaned rows: {len(cleaned):,} (removed exact duplicates: {int(flags['is_exact_duplicate'].sum()):,})")

    # STEP 7 — Save outputs
    save_outputs(
        out_dir=args.output_dir,
        cleaned=cleaned,
        quality_report=quality_report,
        suspicious_rows=suspicious_rows,
        gap_by_asset=gap_by_asset,
        gap_by_class=gap_by_class,
        save_parquet=args.save_parquet,
    )


if __name__ == "__main__":
    main()
