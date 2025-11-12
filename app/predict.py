"""Model loading and inference utilities for the rent estimator."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "trained_model_pipeline.joblib")
LAMBDA_FILENAME = os.getenv("LAMBDA_FILENAME", "lambda_boxcox.joblib")
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID")
MODEL_REPO_TYPE = os.getenv("MODEL_REPO_TYPE", "dataset")

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


def _ensure_local_artifact(local_path: Path, remote_filename: str) -> Path:
    """
    Ensure an artifact exists locally. If it is missing and MODEL_REPO_ID is configured,
    download the file from the Hugging Face Hub (dataset or model repo).
    """

    if local_path.exists():
        return local_path

    if not MODEL_REPO_ID:
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=remote_filename,
            repo_type=MODEL_REPO_TYPE,
            token=os.getenv("HF_TOKEN"),
        )
        shutil.copy(downloaded_path, local_path)
    except Exception as exc:  # pragma: no cover - we simply bubble up a better message
        raise FileNotFoundError(
            f"Unable to download {remote_filename} from {MODEL_REPO_ID}. "
            "Set MODEL_REPO_ID (and optionally HF_TOKEN) or place the file in models/."
        ) from exc

    return local_path


MODEL_PATH = _ensure_local_artifact(MODELS_DIR / MODEL_FILENAME, MODEL_FILENAME)
LAMBDA_PATH = _ensure_local_artifact(MODELS_DIR / LAMBDA_FILENAME, LAMBDA_FILENAME)


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
