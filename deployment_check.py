"""
Pre-deployment validation — run: python deployment_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data" / "mumbai_houses.csv"
MODEL = ARTIFACTS / "house_price_model.joblib"
METRICS = ARTIFACTS / "metrics.json"
STATS = ARTIFACTS / "location_stats.json"
META = ARTIFACTS / "dataset_meta.json"

from location_features import BASE_FEATURE_COLS, all_feature_columns, prepare_features  # noqa: E402
from mumbai_data import TARGET  # noqa: E402


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    raise SystemExit(1)


def check_files() -> None:
    print("1. Required files")
    for path in (MODEL, METRICS, STATS, META, DATA):
        if path.exists():
            ok(str(path.relative_to(ROOT)))
        else:
            fail(f"Missing: {path.relative_to(ROOT)}")


def check_data() -> pd.DataFrame:
    print("\n2. Training data quality")
    df = pd.read_csv(DATA)
    ok(f"{len(df):,} rows")

    missing = df[BASE_FEATURE_COLS + [TARGET]].isnull().sum()
    bad = missing[missing > 0]
    if len(bad):
        fail(f"Null values in columns: {bad.to_dict()}")
    ok("No missing values in feature/target columns")

    if (df["area"] <= 0).any():
        fail("Non-positive area values found")
    ok("All areas positive")

    if (df[TARGET] <= 0).any():
        fail("Non-positive prices found")
    ok("All prices positive")

    dup_pct = df.duplicated(subset=BASE_FEATURE_COLS).mean() * 100
    print(f"  [INFO] Duplicate feature rows: {dup_pct:.1f}%")
    return df


def check_artifacts(df: pd.DataFrame) -> tuple:
    print("\n3. Model bundle")
    bundle = joblib.load(MODEL)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        fail("Model must be a dict with 'pipeline' and 'location_stats'")
    pipeline = bundle["pipeline"]
    stats = bundle["location_stats"]
    ok("Bundle structure valid")

    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    for key in ("r2", "mae_lakhs", "mape_percent"):
        if key not in metrics:
            fail(f"metrics.json missing '{key}'")
    ok(f"Metrics loaded (R2={metrics['r2']:.3f})")

    return pipeline, stats


def check_predictions(pipeline, stats: dict, df: pd.DataFrame) -> None:
    print("\n4. Prediction smoke tests")
    samples = df.sample(5, random_state=42)
    X = prepare_features(samples[BASE_FEATURE_COLS], stats)
    preds = np.expm1(pipeline.predict(X))

    if np.any(np.isnan(preds)) or np.any(preds <= 0):
        fail(f"Invalid predictions: {preds}")
    ok("5 random rows predict without NaN/zero")

    if X.isnull().any().any():
        cols = X.columns[X.isnull().any()].tolist()
        fail(f"NaN in prepared features: {cols}")
    ok("Prepared features have no NaN")

    custom = pd.DataFrame(
        [
            {
                "bhk": 2,
                "type": "Apartment",
                "locality": "Other",
                "area": 650,
                "region": "Andheri West",
                "status": "Ready to move",
                "age": "New",
            }
        ]
    )
    p = float(np.expm1(pipeline.predict(prepare_features(custom, stats))[0]))
    ok(f"UI-style input -> {p:.2f} lakhs")

    expected_cols = set(all_feature_columns())
    got_cols = set(X.columns)
    if not expected_cols.issubset(got_cols):
        fail(f"Missing feature columns: {expected_cols - got_cols}")
    ok(f"All {len(expected_cols)} feature columns present")


def check_imports() -> None:
    print("\n5. App import")
    import app  # noqa: F401

    ok("app.py imports successfully")


def main() -> None:
    print("=== Mumbai House Price — deployment check ===\n")
    check_files()
    df = check_data()
    pipeline, stats = check_artifacts(df)
    check_predictions(pipeline, stats, df)
    check_imports()
    print("\n=== All checks passed — ready to deploy ===\n")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        sys.exit(1)
    except Exception as exc:
        print(f"\n  [FAIL] Unexpected error: {exc}")
        sys.exit(1)
