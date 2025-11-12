"""Minimal smoke test to ensure the prediction pipeline loads and returns a value."""

from __future__ import annotations

from pprint import pprint

from app.predict import FEATURE_COLUMNS, predict_rent

DEFAULT_PAYLOAD = {
    "location": "Ikate Lekki Lagos",
    "type": "Flat Apartment FOR Rent",
    "bedrooms": 2,
    "bathrooms": 2,
    "toilets": 2,
    "area_sqm": 120,
    "Lagos_Area": "Island",
    "month": 1,
    "year": 2025,
    "inflation": 28.2,
    "exchange_rate": 1500,
    "gdp_growth": 3.0,
    "inflation_lag1Q": 24.0,
    "exchange_rate_lag1Q": 1200,
    "gdp_growth_lag1Q": 2.5,
}


def main() -> None:
    missing = [feature for feature in FEATURE_COLUMNS if feature not in DEFAULT_PAYLOAD]
    if missing:
        raise SystemExit(f"Update DEFAULT_PAYLOAD, missing keys: {missing}")
    prediction = predict_rent(DEFAULT_PAYLOAD)
    pprint({"prediction_naira": prediction})


if __name__ == "__main__":
    main()
