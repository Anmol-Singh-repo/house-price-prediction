"""
Location-based features for Mumbai house price prediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mumbai_data import TARGET

ARTIFACTS_STATS = Path(__file__).resolve().parent / "artifacts" / "location_stats.json"

BASE_FEATURE_COLS = ["bhk", "type", "locality", "area", "region", "status", "age"]

LOCATION_NUMERIC = [
    "region_median_price_lakhs",
    "region_median_price_per_sqft",
    "region_listing_count_log",
    "locality_median_price_lakhs",
    "locality_median_price_per_sqft",
    "locality_listing_count_log",
    "area_x_region_psf",
    "location_premium_index",
]

LOCATION_CATEGORICAL = ["region_price_tier", "mumbai_zone"]


def _sorted_str(items) -> list[str]:
    return sorted({str(x) for x in items})


def infer_mumbai_zone(region: str) -> str:
    r = region.lower()
    rules = [
        ("South Mumbai", ("churchgate", "marine", "colaba", "fort", "malabar", "peddar", "napean", "tardeo", "babulnath")),
        ("Western Suburbs", ("andheri", "bandra", "borival", "malad", "kandival", "goregaon", "juhu", "versova", "khar", "santacruz", "vile parle", "oshivara")),
        ("Central Mumbai", ("chembur", "ghatkopar", "kurla", "sion", "parel", "dadar", "bkc", "wadala", "matunga", "mahim")),
        ("Harbour Line", ("panvel", "vashi", "nerul", "belapur", "kharghar", "kamothe", "airoli", "kopar khairane")),
        ("Thane / Navi Mumbai", ("thane", "mulund", "bhandup", "dombiv", "kalyan", "ambernath", "badlapur", "mira road", "bhayandar", "vasai", "virar", "naigaon", "boisar")),
    ]
    for zone, keywords in rules:
        if any(k in r for k in keywords):
            return zone
    return "Other Mumbai"


def _tier(median_price: float, q25: float, q50: float, q75: float) -> str:
    if median_price >= q75:
        return "Premium"
    if median_price >= q50:
        return "Upper-Mid"
    if median_price >= q25:
        return "Mid"
    return "Budget"


def build_location_stats(df: pd.DataFrame) -> dict:
    """Aggregate location statistics from training data (uses target)."""
    work = df.copy()
    work["price_per_sqft"] = work[TARGET] / work["area"].clip(lower=1)

    city_median_psf = float(work["price_per_sqft"].median())
    city_median_price = float(work[TARGET].median())
    region_prices = work.groupby("region")[TARGET]
    q25, q50, q75 = region_prices.median().quantile([0.25, 0.5, 0.75])

    region_stats = {}
    for region, grp in work.groupby("region"):
        region_stats[region] = {
            "median_price_lakhs": float(grp[TARGET].median()),
            "median_price_per_sqft": float(grp["price_per_sqft"].median()),
            "listing_count": int(len(grp)),
            "price_tier": _tier(float(grp[TARGET].median()), q25, q50, q75),
            "mumbai_zone": infer_mumbai_zone(region),
        }

    locality_stats = {}
    locality_to_region = {}
    for locality, grp in work.groupby("locality"):
        mode_region = grp["region"].mode().iloc[0]
        locality_stats[locality] = {
            "median_price_lakhs": float(grp[TARGET].median()),
            "median_price_per_sqft": float(grp["price_per_sqft"].median()),
            "listing_count": int(len(grp)),
            "primary_region": mode_region,
        }
        locality_to_region[locality] = mode_region

    region_localities = {
        region: _sorted_str(grp["locality"].value_counts().head(30).index)
        for region, grp in work.groupby("region")
    }

    return {
        "city_median_price_lakhs": city_median_price,
        "city_median_price_per_sqft": city_median_psf,
        "region_stats": region_stats,
        "locality_stats": locality_stats,
        "locality_to_region": locality_to_region,
        "region_localities": region_localities,
    }


def _lookup_region(region: str, stats: dict) -> dict:
    rs = stats["region_stats"]
    if region in rs:
        return rs[region]
    fallback = {
        "median_price_lakhs": stats["city_median_price_lakhs"],
        "median_price_per_sqft": stats["city_median_price_per_sqft"],
        "listing_count": 1,
        "price_tier": "Mid",
        "mumbai_zone": infer_mumbai_zone(region),
    }
    return fallback


def _lookup_locality(locality: str, region: str, stats: dict) -> dict:
    ls = stats["locality_stats"]
    if locality in ls:
        return ls[locality]
    reg = _lookup_region(region, stats)
    return {
        "median_price_lakhs": reg["median_price_lakhs"],
        "median_price_per_sqft": reg["median_price_per_sqft"],
        "listing_count": 1,
        "primary_region": region,
    }
 

def add_location_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    rows = []
    for _, row in out.iterrows():
        region = str(row["region"])
        locality = str(row["locality"])
        area = float(row["area"])

        reg = _lookup_region(region, stats)
        loc = _lookup_locality(locality, region, stats)

        reg_med = reg["median_price_lakhs"]
        loc_med = loc["median_price_lakhs"]
        reg_psf = reg["median_price_per_sqft"]
        loc_psf = loc["median_price_per_sqft"]

        rows.append(
            {
                "region_median_price_lakhs": reg_med,
                "region_median_price_per_sqft": reg_psf,
                "region_listing_count_log": np.log1p(reg["listing_count"]),
                "locality_median_price_lakhs": loc_med,
                "locality_median_price_per_sqft": loc_psf,
                "locality_listing_count_log": np.log1p(loc["listing_count"]),
                "area_x_region_psf": area * reg_psf,
                "location_premium_index": loc_med / max(reg_med, 0.01),
                "region_price_tier": reg["price_tier"],
                "mumbai_zone": reg["mumbai_zone"],
            }
        )

    loc_df = pd.DataFrame(rows, index=out.index)
    return pd.concat([out, loc_df], axis=1)


def prepare_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    return add_location_features(df[BASE_FEATURE_COLS], stats)


def all_feature_columns() -> list[str]:
    return BASE_FEATURE_COLS + LOCATION_NUMERIC + LOCATION_CATEGORICAL


def save_location_stats(stats: dict, path: Path = ARTIFACTS_STATS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def load_location_stats(path: Path = ARTIFACTS_STATS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
