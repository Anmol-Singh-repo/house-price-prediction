"""
Mumbai House Price Predictor — Streamlit UI (location-aware)
Run: streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from location_features import (
    BASE_FEATURE_COLS,
    _lookup_region,
    load_location_stats,
    prepare_features,
)
from mumbai_data import TARGET, format_inr, load_or_prepare_data

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "house_price_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"
STATS_PATH = ROOT / "artifacts" / "location_stats.json"

FEATURE_LABELS = {
    "bhk": "BHK (bedrooms)",
    "type": "Property type",
    "region": "Area / neighbourhood (region)",
    "locality": "Society / building (locality)",
    "area": "Carpet area (sq ft)",
    "status": "Construction status",
    "age": "Property age",
}
THEMES = {
    "Light": {
        "bg": "#f0f2f6",
        "sidebar_bg": "#ffffff",
        "text": "#202020",
        "muted": "#5c6370",
        "card": "#0077cc",
        "loc_bg": "#e0e4eb",
        "loc_border": "#cbd2db",
    },
    "Dark": {
        "bg": "#0e1117",
        "sidebar_bg": "#262730",
        "text": "#fafafa",
        "muted": "#b0b4bc",
        "card": "#0077cc",
        "loc_bg": "#2d333b",
        "loc_border": "#464c56",
    },
}

def _sorted_str(items) -> list:
    return sorted({str(x) for x in items})


def _sorted_num(items) -> list:
    return sorted({int(x) for x in items})


@st.cache_data
def load_training_data() -> pd.DataFrame:
    return load_or_prepare_data()


@st.cache_resource
def load_model_bundle():
    import joblib

    if not MODEL_PATH.exists():
        st.error("Model not found. Run: `python train_house_price_model.py`")
        st.stop()
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict):
        return obj["pipeline"], obj["location_stats"]
    stats = load_location_stats() if STATS_PATH.exists() else {}
    return obj, stats


@st.cache_data
def get_location_stats(train: pd.DataFrame) -> dict:
    if STATS_PATH.exists():
        return load_location_stats()
    from location_features import build_location_stats

    return build_location_stats(train)


@st.cache_data
def regions_alpha(stats: dict, train: pd.DataFrame) -> list[str]:
    regions = list(stats.get("region_stats", {}).keys()) or train["region"].unique().tolist()
    return _sorted_str(regions)


@st.cache_data
def localities_for_region(stats: dict, region: str, train: pd.DataFrame) -> list[str]:
    mapped = stats.get("region_localities", {}).get(region, [])
    if mapped:
        return _sorted_str(mapped)
    subset = train.loc[train["region"] == region, "locality"].unique()
    return _sorted_str(subset) if len(subset) else ["Other"]


@st.cache_data
def preset_rows(train: pd.DataFrame) -> dict[str, tuple[dict, float]]:
    ranked = train.sort_values(TARGET)
    picks = {
        "Budget flat": ranked.iloc[len(ranked) // 10],
        "Mid-range flat": ranked.iloc[len(ranked) // 2],
        "Premium flat": ranked.iloc[int(len(ranked) * 0.9)],
    }
    out: dict[str, tuple[dict, float]] = {}
    for label, row in picks.items():
        base = {c: row[c] for c in BASE_FEATURE_COLS}
        out[label] = (base, float(row[TARGET]))
    return out


def _index(options: list, value) -> int:
    val = value.item() if hasattr(value, "item") else value
    try:
        return list(options).index(val)
    except ValueError:
        return 0


def apply_theme(theme_name: str) -> None:
    t = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {t["bg"]} !important;
            color: {t["text"]} !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {t["sidebar_bg"]} !important;
        }}
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stMarkdown p {{
            color: {t["text"]} !important;
        }}
        .stApp h1, .stApp h2, .stApp h3,
        .stApp label, .stApp p, .stApp span,
        .stApp [data-testid="stMarkdownContainer"] {{
            color: {t["text"]} !important;
        }}
        .stApp .stCaption, .stApp [data-testid="stCaptionContainer"] {{
            color: {t["muted"]} !important;
        }}
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricValue"] {{
            color: {t["text"]} !important;
        }}
        .price-card {{
            background: {t["card"]};
            color: white;
            padding: 1.5rem 2rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 1rem;
        }}
        .price-card h1 {{ margin: 0; font-size: 2.75rem; font-weight: 700; color: white !important; }}
        .price-card p {{ margin: 0.25rem 0 0; opacity: 0.95; color: white !important; }}
        .loc-box {{
            background: {t["loc_bg"]};
            border: 1px solid {t["loc_border"]};
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
            color: {t["text"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def predict_price(pipeline, features: pd.DataFrame, stats: dict) -> float:
    X = prepare_features(features, stats)
    if X.isnull().any().any():
        X = X.fillna(X.median(numeric_only=True))
    return float(np.expm1(pipeline.predict(X)[0]))


def main() -> None:
    st.set_page_config(
        page_title="Mumbai House Price Predictor",
        page_icon="🏙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "theme" not in st.session_state:
        st.session_state.theme = "Light"

    train = load_training_data()
    pipeline, stats = load_model_bundle()
    if not stats:
        stats = get_location_stats(train)

    apply_theme(st.session_state.theme)

    st.title("🏙️ Mumbai House Price Predictor")
    st.caption("Location-aware model · Kaggle Mumbai House Prices · ₹ lakhs / crore")

    presets = preset_rows(train)
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    preset_actual: float | None = None
    base = {
        c: train[c].mode().iloc[0] if train[c].dtype == object else train[c].median()
        for c in BASE_FEATURE_COLS
    }
    base["bhk"] = int(train["bhk"].median())
    base["area"] = int(train["area"].median())

    region_list = regions_alpha(stats, train)
    preset_options = ["Custom"] + _sorted_str(presets.keys())

    with st.sidebar:
        st.radio("Theme", list(THEMES.keys()), horizontal=True, key="theme")

        st.divider()
        st.header("Property details")
        preset_name = st.selectbox("Load example", preset_options)
        if preset_name != "Custom":
            base, preset_actual = presets[preset_name]

        st.subheader("📍 Location")
        region = st.selectbox(
            FEATURE_LABELS["region"],
            region_list,
            index=_index(region_list, base["region"]),
            help="Mumbai area — strongly affects price",
        )

        loc_options = localities_for_region(stats, region, train)
        locality = st.selectbox(
            FEATURE_LABELS["locality"],
            loc_options,
            index=_index(loc_options, base["locality"] if base["locality"] in loc_options else "Other"),
        )

        reg_info = _lookup_region(region, stats)
        st.caption(
            f"**{reg_info['mumbai_zone']}** · {reg_info['price_tier']} tier · "
            f"Area avg **{format_inr(reg_info['median_price_lakhs'])}** "
            f"({reg_info['median_price_per_sqft']:.2f} L/sqft)"
        )

        st.subheader("Home specs")
        bhk_opts = _sorted_num(train["bhk"].unique())
        bhk = st.selectbox(
            FEATURE_LABELS["bhk"],
            bhk_opts,
            index=_index(bhk_opts, int(base["bhk"])),
        )
        type_opts = _sorted_str(train["type"].unique())
        prop_type = st.selectbox(
            FEATURE_LABELS["type"],
            type_opts,
            index=_index(type_opts, base["type"]),
        )
        area = st.number_input(
            FEATURE_LABELS["area"],
            min_value=int(train["area"].min()),
            max_value=int(train["area"].max()),
            value=int(base["area"]),
            step=50,
        )
        status_opts = _sorted_str(train["status"].unique())
        status = st.selectbox(
            FEATURE_LABELS["status"],
            status_opts,
            index=_index(status_opts, base["status"]),
        )
        age_opts = _sorted_str(train["age"].unique())
        age = st.selectbox(
            FEATURE_LABELS["age"],
            age_opts,
            index=_index(age_opts, base["age"]),
        )

    base_features = pd.DataFrame(
        [
            {
                "bhk": bhk,
                "type": prop_type,
                "locality": locality,
                "area": area,
                "region": region,
                "status": status,
                "age": age,
            }
        ]
    )

    try:
        price_lakhs = predict_price(pipeline, base_features, stats)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    price_display = format_inr(price_lakhs)
    implied_psf = price_lakhs / max(area, 1)

    st.markdown(
        f"""
        <div class="price-card">
            <p>Estimated price in {region}</p>
            <h1>{price_display}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="loc-box">'
        f"<b>Location breakdown</b><br>"
        f"Zone: {reg_info['mumbai_zone']} · Tier: {reg_info['price_tier']}<br>"
        f"Typical {region} price: <b>{format_inr(reg_info['median_price_lakhs'])}</b> "
        f"({reg_info['median_price_per_sqft']:.2f} L/sqft) · "
        f"Your estimate: <b>{implied_psf:.2f} L/sqft</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    if metrics:
        c1.metric("Model R²", f"{metrics.get('r2', 0):.3f}")
        c2.metric("Avg error (MAPE)", f"{metrics.get('mape_percent', 0):.1f}%")
        c3.metric("Typical error", format_inr(metrics.get("mae_lakhs", 0)))
        c4.metric("CV log-RMSE", f"{metrics.get('cv_log_rmse', 0):.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Your inputs")
        full_x = prepare_features(base_features, stats)
        show_cols = BASE_FEATURE_COLS + [
            "mumbai_zone",
            "region_price_tier",
            "region_median_price_per_sqft",
        ]
        show = full_x[[c for c in show_cols if c in full_x.columns]]
        st.dataframe(
            pd.DataFrame({"Feature": show.columns, "Value": show.iloc[0].tolist()}),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Market context")
        prices = train[TARGET]
        region_prices = train.loc[train["region"] == region, TARGET]
        pct_region = (
            float((region_prices < price_lakhs).mean() * 100) if len(region_prices) else 0.0
        )
        st.progress(pct_region / 100, text=f"Pricier than {pct_region:.0f}% of homes in {region}")
        if len(region_prices):
            st.write(
                f"**{region}** range: {format_inr(region_prices.min())} – "
                f"{format_inr(region_prices.max())} (median {format_inr(region_prices.median())})"
            )
        st.write(
            f"All Mumbai: {format_inr(prices.min())} – {format_inr(prices.max())} "
            f"(median {format_inr(prices.median())})"
        )
        if preset_actual is not None:
            st.info(f"**{preset_name}** listed at {format_inr(preset_actual)}")

    st.caption(f"{len(train):,} listings · location features enabled")


if __name__ == "__main__":
    main()
