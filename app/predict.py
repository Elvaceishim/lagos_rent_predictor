"""Model loading and inference utilities for the rent estimator."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "trained_model_pipeline.joblib"
LAMBDA_PATH = ROOT / "models" / "lambda_boxcox.joblib"
CONFORMAL_SOURCE = ROOT / "data" / "true_vs_predicted_naira.csv"
DEFAULT_CONFORMAL_SCALE = 9_243_250.165298764  # 90th percentile |y_true - y_pred| from Nov 2025 residuals

# These are the raw feature columns the training pipeline expected.
FEATURE_COLUMNS = [
    "location",
    "type",
    "bedrooms",
    "bathrooms",
    "toilets",
    "area_sqm",
    "Lagos_Area",
    "month",
    "year",
    "inflation",
    "exchange_rate",
    "gdp_growth",
    "inflation_lag1Q",
    "exchange_rate_lag1Q",
    "gdp_growth_lag1Q",
]


@lru_cache(maxsize=1)
def _load_model():
    """Load the serialized sklearn pipeline exactly once per process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Export the pipeline from training and place it in models/."
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_lambda() -> float:
    if not LAMBDA_PATH.exists():
        raise FileNotFoundError(
            f"Box-Cox lambda not found at {LAMBDA_PATH}. "
            "Export lambda_boxcox.joblib from training."
        )
    lam = joblib.load(LAMBDA_PATH)
    return float(lam)


@lru_cache(maxsize=1)
def _load_conformal_quantile(quantile: float = 0.9) -> Optional[float]:
    if not CONFORMAL_SOURCE.exists():
        return DEFAULT_CONFORMAL_SCALE
    df = pd.read_csv(CONFORMAL_SOURCE)
    if not {"y_true_naira", "y_pred_naira"}.issubset(df.columns):
        return DEFAULT_CONFORMAL_SCALE
    y_true = pd.to_numeric(df["y_true_naira"], errors="coerce")
    y_pred = pd.to_numeric(df["y_pred_naira"], errors="coerce")
    residuals = (y_true - y_pred).abs().dropna()
    if residuals.empty:
        return DEFAULT_CONFORMAL_SCALE
    value = float(residuals.quantile(quantile))
    return value if np.isfinite(value) else DEFAULT_CONFORMAL_SCALE


def _inverse_boxcox(values: np.ndarray, lam: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if lam == 0:
        return np.exp(values)
    adjusted = lam * values + 1.0
    adjusted = np.clip(adjusted, a_min=1e-9, a_max=None)
    return np.power(adjusted, 1.0 / lam)


def predict_rent(features: Dict[str, Any]) -> float:
    """
    Run a single prediction.

    Parameters
    ----------
    features: dict
        Must contain the feature keys listed in FEATURE_COLUMNS.

    Returns
    -------
    float
        Predicted monthly rent in naira.
    """

    model = _load_model()
    payload = {key: features.get(key) for key in FEATURE_COLUMNS}
    frame = pd.DataFrame([payload])
    prediction = model.predict(frame)
    lam = _load_lambda()
    naira_values = _inverse_boxcox(prediction, lam)
    return float(naira_values[0])


def conformal_band(price: float, quantile: float = 0.9) -> Optional[tuple[float, float]]:
    """Return lower/upper bounds using absolute residual quantile."""
    scale = _load_conformal_quantile(quantile)
    if scale is None:
        return None
    lower = max(price - scale, 0.0)
    upper = price + scale
    return lower, upper
