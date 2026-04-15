"""processed2.0 — second-stage lightweight validation (no over-cleaning).

Input dataset: intraday 5-minute OHLCV already cleaned.

Key decisions:
- Remove USDCNY entirely.
- Do NOT interpolate / forward-fill / create artificial rows.
- Keep real gaps (they are part of market structure).
- Only drop exact duplicate rows if present.

Outputs (under output dir):
- processed2.0.parquet (preferred) and processed2.0.csv
- processed2.0_report.csv (per-asset summary)
- processed2.0_suspicious.csv (rows that fail checks)

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


def load_dataset(input_path: str | Path) -> pd.DataFrame:
    """Load parquet/csv dataset."""

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        # Try parquet first, then csv.
        try:
            df = pd.read_parquet(input_path)
        except Exception:
            df = pd.read_csv(input_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def remove_problem_asset(df: pd.DataFrame, asset_id: str = "USDCNY") -> pd.DataFrame:
    before = len(df)
    out = df[df["asset_id"].astype(str) != asset_id].copy()
    removed = before - len(out)
    print(f"STEP 2 — Remove {asset_id}: removed_rows={removed}")
    return out


def sort_prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["asset_id", "date"], kind="mergesort").reset_index(drop=True)
    return out


def find_exact_duplicates(df: pd.DataFrame) -> pd.Series:
    """Exact duplicates over the original 5-min OHLCV columns.

    NOTE: We exclude computed columns like time_gap.
    """

    key_cols = [
        "asset_id",
        "asset_class",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    return df.duplicated(subset=key_cols, keep="first")


def compute_time_gaps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_gap"] = out.groupby("asset_id", sort=False)["date"].diff()
    return out


def price_logic_violations(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    close = df["close"]

    bad_hl = high < low
    bad_open = (open_ < low) | (open_ > high)
    bad_close = (close < low) | (close > high)

    return (bad_hl | bad_open | bad_close).fillna(False)


def duplicate_timestamps(df: pd.DataFrame) -> pd.Series:
    return df.duplicated(subset=["asset_id", "date"], keep=False)


def build_validation_summary(
    df: pd.DataFrame,
    abnormal_gap_mask: pd.Series,
    price_bad_mask: pd.Series,
    dup_ts_mask: pd.Series,
) -> pd.DataFrame:
    """Per-asset summary table."""

    # Missing values count per asset (total missing cells across all columns)
    missing_by_asset = df.groupby("asset_id", sort=False).apply(lambda g: int(g.isna().sum().sum()))

    summary = (
        df.groupby("asset_id", sort=False)
        .agg(total_rows=("asset_id", "size"))
        .reset_index()
        .merge(
            df[["asset_id", "asset_class"]].drop_duplicates(subset=["asset_id"]),
            on="asset_id",
            how="left",
        )
    )

    # Count events per asset
    abnormal_gaps = abnormal_gap_mask.groupby(df["asset_id"], sort=False).sum().astype(int).rename("n_abnormal_gaps")
    price_viol = price_bad_mask.groupby(df["asset_id"], sort=False).sum().astype(int).rename("n_price_violations")
    dup_ts = dup_ts_mask.groupby(df["asset_id"], sort=False).sum().astype(int).rename("n_duplicated_timestamps")

    summary = summary.merge(abnormal_gaps.reset_index(), on="asset_id", how="left")
    summary = summary.merge(price_viol.reset_index(), on="asset_id", how="left")
    summary = summary.merge(dup_ts.reset_index(), on="asset_id", how="left")

    summary["n_missing_values"] = summary["asset_id"].map(missing_by_asset).fillna(0).astype(int)

    # Fill NaNs for counts
    for c in ["n_abnormal_gaps", "n_price_violations", "n_duplicated_timestamps"]:
        summary[c] = summary[c].fillna(0).astype(int)

    # Keep requested fields (asset_id + per-asset metrics). asset_class is useful context.
    summary = summary[
        [
            "asset_id",
            "asset_class",
            "total_rows",
            "n_abnormal_gaps",
            "n_price_violations",
            "n_duplicated_timestamps",
            "n_missing_values",
        ]
    ].sort_values(["asset_class", "asset_id"], kind="mergesort")

    return summary


def save_outputs(
    df_final: pd.DataFrame,
    report: pd.DataFrame,
    suspicious: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "processed2.0.parquet"
    csv_path = out_dir / "processed2.0.csv"
    report_path = out_dir / "processed2.0_report.csv"
    suspicious_path = out_dir / "processed2.0_suspicious.csv"

    # Final dataset
    wrote_parquet = False
    try:
        df_final.to_parquet(parquet_path, index=False)
        wrote_parquet = True
    except Exception as e:  # noqa: BLE001
        print(f"WARN: Failed to write parquet ({parquet_path}): {e}")

    df_final.to_csv(csv_path, index=False, encoding="utf-8")

    # Reports
    report.to_csv(report_path, index=False, encoding="utf-8")

    if len(suspicious) > 0:
        suspicious.to_csv(suspicious_path, index=False, encoding="utf-8")
    else:
        # Always write an empty file with header for stability.
        suspicious.head(0).to_csv(suspicious_path, index=False, encoding="utf-8")

    print("STEP 8 — Save outputs")
    print("=" * 80)
    if wrote_parquet:
        print(f"Saved: {parquet_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {suspicious_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="processed2.0 lightweight validation step")
    parser.add_argument(
        "-i",
        "--input",
        default=str(Path("data") / "processed" / "cleaned_dataset.parquet"),
        help="Input dataset path (.parquet or .csv)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(Path("data") / "processed2.0"),
        help="Output directory",
    )
    parser.add_argument("--expected-minutes", type=int, default=5, help="Expected bar frequency in minutes")
    parser.add_argument("--abnormal-gap-minutes", type=int, default=60, help="Abnormal gap threshold (minutes)")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    print("STEP 1 — Load dataset")
    print("=" * 80)
    df = load_dataset(args.input)
    print(f"Input shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    df = remove_problem_asset(df, asset_id="USDCNY")

    print("STEP 3 — Sort and prepare")
    print("=" * 80)
    df = sort_prepare(df)

    # Duplicate checks (do not remove yet)
    dup_ts_mask_before = duplicate_timestamps(df)
    n_dup_ts_rows = int(dup_ts_mask_before.sum())
    print(f"STEP 5.5 — Duplicate timestamp check: duplicated_rows={n_dup_ts_rows}")

    # Only remove exact duplicate rows (as allowed)
    exact_dup_mask = find_exact_duplicates(df)
    n_exact_dups = int(exact_dup_mask.sum())
    if n_exact_dups > 0:
        print(f"STEP 7 — Removing exact duplicate rows: n_exact_duplicates={n_exact_dups}")
    df_final = df.loc[~exact_dup_mask].copy().reset_index(drop=True)

    print("STEP 4 — Compute time gaps")
    print("=" * 80)
    df_final = compute_time_gaps(df_final)

    expected_gap = pd.Timedelta(minutes=int(args.expected_minutes))
    abnormal_gap = pd.Timedelta(minutes=int(args.abnormal_gap_minutes))

    # STEP 5 — Checks
    print("STEP 5 — Lightweight validation checks")
    print("=" * 80)

    # 1) time continuity check
    non_first = df_final["time_gap"].notna()
    not_expected = non_first & (df_final["time_gap"] != expected_gap)
    n_not_expected = int(not_expected.sum())
    print(f"1) Time continuity: gaps != {expected_gap}: {n_not_expected}")

    # 2) abnormal gap > 1 hour
    abnormal_gap_mask = non_first & (df_final["time_gap"] > abnormal_gap)
    n_abnormal = int(abnormal_gap_mask.sum())
    print(f"2) Abnormal gaps > {abnormal_gap}: {n_abnormal}")
    if n_abnormal:
        abnormal_by_asset = (
            df_final.loc[abnormal_gap_mask].groupby("asset_id", sort=False)["time_gap"].size().sort_values(ascending=False)
        )
        print("   Abnormal gaps by asset (top):")
        print(abnormal_by_asset.head(20).to_string())

    # 3) price logic
    price_bad_mask = price_logic_violations(df_final)
    n_price_bad = int(price_bad_mask.sum())
    print(f"3) Price logic violations: {n_price_bad}")

    # 4) missing values per column
    missing_per_col = df_final.isna().sum().sort_values(ascending=False)
    print("4) Missing values per column (top):")
    print(missing_per_col.head(30).to_string())

    # 5) duplicate timestamps (after exact-dupe removal)
    dup_ts_mask = duplicate_timestamps(df_final)
    n_dup_ts_after = int(dup_ts_mask.sum())
    print(f"5) Duplicate timestamps (asset_id,date) rows (after exact-dupe drop): {n_dup_ts_after}")

    # STEP 6 — Summary table
    print("STEP 6 — Build validation summary")
    print("=" * 80)
    report = build_validation_summary(
        df_final,
        abnormal_gap_mask=abnormal_gap_mask,
        price_bad_mask=price_bad_mask,
        dup_ts_mask=dup_ts_mask,
    )
    print(report.to_string(index=False))

    # Suspicious rows
    has_missing_row = df_final[REQUIRED_COLS].isna().any(axis=1)
    suspicious_mask = abnormal_gap_mask | price_bad_mask | dup_ts_mask | has_missing_row
    suspicious = df_final.loc[suspicious_mask].copy()
    suspicious["is_abnormal_gap_gt_1h"] = abnormal_gap_mask.loc[suspicious.index].astype(int)
    suspicious["is_price_violation"] = price_bad_mask.loc[suspicious.index].astype(int)
    suspicious["is_duplicate_timestamp"] = dup_ts_mask.loc[suspicious.index].astype(int)
    suspicious["has_missing"] = has_missing_row.loc[suspicious.index].astype(int)

    # STEP 8 — Save
    save_outputs(df_final=df_final, report=report, suspicious=suspicious, out_dir=out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
