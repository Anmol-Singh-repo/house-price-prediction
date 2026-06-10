"""
Mumbai house prices — Kaggle: dravidvaishnav/mumbai-house-prices
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_CSV_NAME = "Mumbai House Prices.csv"
PREPARED_CSV = DATA_DIR / "mumbai_houses.csv"
KAGGLE_DATASET = "dravidvaishnav/mumbai-house-prices"
TARGET = "price_lakhs"
TOP_LOCALITIES = 400
MIN_REGION_LISTINGS = 60
DROP_COLS = ("price", "price_unit")


def download_mumbai_raw() -> Path:
    import kagglehub

    cache = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    src = cache / RAW_CSV_NAME
    if not src.exists():
        matches = list(cache.glob("*.csv"))
        if not matches:
            raise FileNotFoundError(f"No CSV found in {cache}")
        src = matches[0]
    return src


def price_to_lakhs(df: pd.DataFrame) -> pd.Series:
    """Normalize mixed L (lakhs) and Cr (crore) listings to lakhs."""
    lakhs = np.where(df["price_unit"].eq("L"), df["price"], df["price"] * 100.0)
    return pd.Series(lakhs, index=df.index, name=TARGET)


def prepare_mumbai_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TARGET] = price_to_lakhs(out)
    out = out.drop(columns=list(DROP_COLS), errors="ignore")

    top_localities = out["locality"].value_counts().head(TOP_LOCALITIES).index
    out["locality"] = out["locality"].where(out["locality"].isin(top_localities), "Other")

    region_counts = out["region"].value_counts()
    keep_regions = region_counts[region_counts >= MIN_REGION_LISTINGS].index
    out["region"] = out["region"].where(out["region"].isin(keep_regions), "Other")

    out["bhk"] = out["bhk"].astype(int)
    out["area"] = out["area"].astype(int)
    for col in ("type", "locality", "region", "status", "age"):
        out[col] = out[col].astype(str).str.strip()

    out = out.dropna(subset=[TARGET, "area", "bhk"])
    out = out[(out[TARGET] > 0) & (out["area"] > 0)]
    return out.reset_index(drop=True)


def load_or_prepare_data(force_refresh: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PREPARED_CSV.exists() and not force_refresh:
        df = pd.read_csv(PREPARED_CSV)
        if "region" in df.columns and (df["region"] == "Other").mean() < 0.05:
            return df
        force_refresh = True

    print(f"Downloading Mumbai data ({KAGGLE_DATASET})...")
    raw_path = download_mumbai_raw()
    df = pd.read_csv(raw_path)
    prepared = prepare_mumbai_df(df)
    prepared.to_csv(PREPARED_CSV, index=False)
    print(f"Saved {len(prepared):,} rows to {PREPARED_CSV}")
    return prepared


def format_inr(lakhs: float) -> str:
    """Format lakhs as Indian ₹ string (L or Cr)."""
    if lakhs >= 100:
        return f"₹{lakhs / 100:.2f} Cr"
    return f"₹{lakhs:.2f} L"
