"""
Mumbai house price prediction with location-aware features.
Optional hyperparameter fine-tuning (RandomizedSearchCV).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from location_features import (
    all_feature_columns,
    build_location_stats,
    prepare_features,
    save_location_stats,
)
from mumbai_data import TARGET, load_or_prepare_data

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
RANDOM_STATE = 42

DEFAULT_REGRESSOR_KWARGS: dict[str, Any] = {
    "n_estimators": 600,
    "max_depth": 7,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
}


def build_pipeline(X: pd.DataFrame, regressor_kwargs: dict[str, Any] | None = None) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    kw = {**DEFAULT_REGRESSOR_KWARGS, **(regressor_kwargs or {})}
    model = XGBRegressor(**kw)

    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model)])


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> dict:
    y_pred = np.expm1(pipeline.predict(X))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))
    mape = float(np.mean(np.abs((y - y_pred) / y)) * 100)
    return {"rmse_lakhs": rmse, "mae_lakhs": mae, "r2": r2, "mape_percent": mape}


def cross_validate_log_rmse(pipeline: Pipeline, X: pd.DataFrame, y_log: np.ndarray) -> float:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline,
        X,
        y_log,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return float(-scores.mean())


def tune_hyperparams(
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    n_iter: int,
    cv: int,
    max_samples: int,
    random_state: int,
) -> dict[str, Any]:
    """Random search on a subsample for speed; returns best regressor kwargs."""
    n = len(X_train)
    if n > max_samples:
        idx = np.random.default_rng(random_state).choice(n, size=max_samples, replace=False)
        Xs = X_train.iloc[idx].copy()
        ys = y_train_log[idx]
    else:
        Xs, ys = X_train, y_train_log

    base = build_pipeline(Xs)
    param_dist = {
        "regressor__n_estimators": randint(400, 1400),
        "regressor__max_depth": randint(4, 11),
        "regressor__learning_rate": loguniform(0.02, 0.12),
        "regressor__subsample": uniform(0.65, 0.30),
        "regressor__colsample_bytree": uniform(0.65, 0.30),
        "regressor__colsample_bylevel": uniform(0.65, 0.30),
        "regressor__reg_alpha": loguniform(1e-4, 2.0),
        "regressor__reg_lambda": loguniform(0.3, 8.0),
        "regressor__min_child_weight": randint(1, 12),
        "regressor__gamma": loguniform(1e-4, 0.5),
    }

    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=1,
        verbose=1,
        refit=True,
    )
    search.fit(Xs, ys)

    best = search.best_estimator_.named_steps["regressor"].get_params()
    keep = {
        k: v
        for k, v in best.items()
        if k
        in {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "colsample_bylevel",
            "reg_alpha",
            "reg_lambda",
            "min_child_weight",
            "gamma",
        }
    }
    # sklearn may return None for some; drop
    keep = {k: v for k, v in keep.items() if v is not None}
    print(f"\nBest CV log-RMSE (search): {-search.best_score_:.5f}")
    print("Best params:", json.dumps(keep, indent=2, default=str))
    return keep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Mumbai house price model")
    p.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter search (faster, default params)",
    )
    p.add_argument(
        "--tune-samples",
        type=int,
        default=35_000,
        help="Max rows used for hyperparameter search (subsampled)",
    )
    p.add_argument("--tune-iter", type=int, default=28, help="Random search iterations")
    p.add_argument("--tune-cv", type=int, default=3, help="CV folds during search")
    p.add_argument(
        "--no-refresh-data",
        action="store_true",
        help="Use cached mumbai_houses.csv without re-downloading",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = load_or_prepare_data(force_refresh=not args.no_refresh_data)
    except Exception as exc:
        print(f"\nCould not load Mumbai data.\nError: {exc}\n", file=sys.stderr)
        sys.exit(1)

    y = df[TARGET].values
    location_stats = build_location_stats(df)
    save_location_stats(location_stats)

    X = prepare_features(df, location_stats)
    y_log = np.log1p(y)

    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    y_train_log = np.log1p(y_train)

    print(f"\nDataset: {len(df):,} Mumbai listings", flush=True)
    print(f"Features: {len(all_feature_columns())} (incl. location stats)", flush=True)

    regressor_kw: dict[str, Any] | None = None
    tuned = False
    if not args.no_tune:
        print(
            f"\nFine-tuning (RandomizedSearchCV: n_iter={args.tune_iter}, "
            f"cv={args.tune_cv}, up to {args.tune_samples:,} rows)...",
            flush=True,
        )
        regressor_kw = tune_hyperparams(
            X_train,
            y_train_log,
            n_iter=args.tune_iter,
            cv=args.tune_cv,
            max_samples=args.tune_samples,
            random_state=RANDOM_STATE,
        )
        tuned = True
        # Slightly more trees on final fit after tuning best depth / lr
        if "n_estimators" in regressor_kw:
            regressor_kw["n_estimators"] = int(regressor_kw["n_estimators"] * 1.12)

    pipeline = build_pipeline(X_train, regressor_kw)

    print("\n5-fold cross-validation (log RMSE) on full train split...")
    cv_rmse = cross_validate_log_rmse(pipeline, X_train, y_train_log)
    print(f"  CV log-RMSE: {cv_rmse:.5f}")

    print("\nTraining on 85% split...")
    pipeline.fit(X_train, y_train_log)

    hold_metrics = evaluate(pipeline, X_hold, y_hold)
    print("\nHold-out accuracy (lakhs):")
    print(f"  R2:   {hold_metrics['r2']:.4f}")
    print(f"  RMSE: {hold_metrics['rmse_lakhs']:.2f} lakhs")
    print(f"  MAE:  {hold_metrics['mae_lakhs']:.2f} lakhs")
    print(f"  MAPE: {hold_metrics['mape_percent']:.2f}%")

    print("\nRefitting on full dataset...")
    pipeline.fit(X, y_log)

    bundle = {"pipeline": pipeline, "location_stats": location_stats}
    model_path = ARTIFACTS_DIR / "house_price_model.joblib"
    joblib.dump(bundle, model_path)

    metrics = {
        "cv_log_rmse": cv_rmse,
        **hold_metrics,
        "n_rows": len(df),
        "tuned": tuned,
    }
    if tuned and regressor_kw:
        metrics["best_regressor_params"] = dict(regressor_kw)

    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "dataset_meta.json").write_text(
        json.dumps(
            {
                "city": "Mumbai",
                "target": TARGET,
                "features": all_feature_columns(),
                "location_features": True,
                "hyperparameter_tuning": tuned,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nModel saved: {model_path}")


if __name__ == "__main__":
    main()
