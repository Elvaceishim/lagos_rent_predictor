"""Model loading and inference utilities for the rent estimator."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "trained_model_pipeline.joblib"

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
    prediction = model.predict(frame)[0]
    return float(prediction)
