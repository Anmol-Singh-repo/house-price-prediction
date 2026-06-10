"""
Predict Mumbai house prices (with location features).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from location_features import load_location_stats, prepare_features
from mumbai_data import TARGET, format_inr

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "artifacts" / "house_price_model.joblib"
BASE_COLS = ["bhk", "type", "locality", "area", "region", "status", "age"]


def load_bundle(model_path: Path) -> tuple:
    obj = joblib.load(model_path)
    if isinstance(obj, dict):
        return obj["pipeline"], obj["location_stats"]
    stats = load_location_stats()
    return obj, stats


def predict(model_path: Path, features: pd.DataFrame) -> np.ndarray:
    pipeline, stats = load_bundle(model_path)
    X = prepare_features(features[BASE_COLS], stats)
    return np.expm1(pipeline.predict(X))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Mumbai house prices")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.input)
    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    prices = predict(args.model, df)
    result = df.copy()
    result[f"predicted_{TARGET}"] = prices
    result["predicted_display"] = [format_inr(p) for p in prices]

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Predictions written to {args.output}")
    else:
        print(result[["predicted_display"]].to_string(index=False))


if __name__ == "__main__":
    main()
