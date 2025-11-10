"""Gradio application for the Lagos rent price estimator."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.predict import predict_rent, FEATURE_COLUMNS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "combined_properties.csv"


def _load_reference_data() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found at {DATASET_PATH}. "
            "Run the scrapers to populate combined_properties.csv."
        )
    df = pd.read_csv(DATASET_PATH)
    df["price_naira"] = pd.to_numeric(df["price_naira"], errors="coerce")
    df["location_lower"] = df["location"].str.lower()
    return df


REFERENCE_DF = _load_reference_data()
LOCATION_CHOICES = sorted(
    {loc.strip() for loc in REFERENCE_DF["location"].dropna() if isinstance(loc, str)}
)
PROPERTY_TYPES = sorted(
    {prop.strip() for prop in REFERENCE_DF["type"].dropna() if isinstance(prop, str)}
) or ["Flat Apartment", "House"]

ISLAND_KEYWORDS = [
    "lekki",
    "victoria island",
    "ikoyi",
    "oniru",
    "banana island",
    "ajah",
    "chevron",
    "sangotedo",
]

MAINLAND_KEYWORDS = [
    "ikeja",
    "yaba",
    "surulere",
    "maryland",
    "ogba",
    "ojodu",
    "berger",
    "mushin",
    "ilupeju",
    "oshodi",
    "gbagada",
    "shomolu",
    "bariga",
    "ketu",
    "magodo",
    "isolo",
    "abule egba",
    "ipaja",
    "egbeda",
    "agege",
    "ikotun",
    "ojota",
    "apapa",
]

AREAS = ["Mainland", "Island", "Outskirts"]


def infer_area_from_location(location: str) -> str:
    lower = (location or "").lower()
    if any(keyword in lower for keyword in ISLAND_KEYWORDS):
        return "Island"
    if any(keyword in lower for keyword in MAINLAND_KEYWORDS):
        return "Mainland"
    return "Outskirts"


def _comparable_properties(location: str, limit: int = 8) -> pd.DataFrame:
    needle = location or ""
    mask = REFERENCE_DF["location"].str.contains(needle, case=False, regex=False, na=False)
    comps = (
        REFERENCE_DF.loc[mask, ["location", "bedrooms", "bathrooms", "area_sqm", "type", "price_naira"]]
        .dropna(subset=["price_naira"])
        .sort_values("price_naira")
        .head(limit)
    )
    return comps.rename(columns={"price_naira": "price (₦)"})


def run_inference(
    location: str,
    property_type: str,
    lagos_area: str,
    bedrooms: float,
    bathrooms: float,
    toilets: float,
    area_sqm: float,
    month: int,
    year: int,
    inflation: float,
    exchange_rate: float,
    gdp_growth: float,
    inflation_lag1: float,
    exchange_rate_lag1: float,
    gdp_growth_lag1: float,
) -> Tuple[str, pd.DataFrame]:
    inferred_area = lagos_area or infer_area_from_location(location)

    payload = {
        "location": location,
        "type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "toilets": toilets,
        "area_sqm": area_sqm,
        "Lagos_Area": inferred_area,
        "month": month,
        "year": year,
        "inflation": inflation,
        "exchange_rate": exchange_rate,
        "gdp_growth": gdp_growth,
        "inflation_lag1Q": inflation_lag1,
        "exchange_rate_lag1Q": exchange_rate_lag1,
        "gdp_growth_lag1Q": gdp_growth_lag1,
    }

    predicted_price = predict_rent(payload)
    summary = (
        f"### Estimated rent\n"
        f"**₦{predicted_price:,.0f} per year**\n\n"
        f"- Location: **{location}**\n"
        f"- Property type: **{property_type}**\n"
        f"- {int(bedrooms)} bed / {int(bathrooms)} bath / {int(toilets)} toilets\n"
        f"- Area: **{area_sqm:.0f} sqm**"
    )

    comps = _comparable_properties(location)
    if comps.empty:
        comps = pd.DataFrame({"location": ["No comparable listings found"], "price (₦)": [""]})

    return summary, comps


with gr.Blocks(title="Lagos Rent Estimator") as demo:
    gr.Markdown(
        """
        # Lagos Rent Price Estimator
        Estimate annual rent for Lagos mainland apartments using the trained regression model.
        """
    )

    with gr.Row():
        location_input = gr.Dropdown(
            choices=LOCATION_CHOICES,
            label="Location / neighborhood",
            value=LOCATION_CHOICES[0] if LOCATION_CHOICES else None,
            allow_custom_value=True,
        )
        property_type_input = gr.Dropdown(
            choices=PROPERTY_TYPES,
            label="Property type",
            value=PROPERTY_TYPES[0],
            allow_custom_value=True,
        )
        area_zone_input = gr.Dropdown(
            choices=AREAS,
            label="Lagos area",
            value=None,
            info="Leave blank to auto-detect from the location.",
            allow_custom_value=True,
        )

    with gr.Row():
        bedroom_input = gr.Slider(1, 8, step=1, value=2, label="Bedrooms")
        bathroom_input = gr.Slider(1, 8, step=1, value=2, label="Bathrooms")
        toilet_input = gr.Slider(1, 8, step=1, value=2, label="Toilets")
        area_input = gr.Number(value=120, label="Area (sqm)")

    with gr.Row():
        month_input = gr.Slider(1, 12, step=1, value=1, label="Month")
        year_input = gr.Slider(2019, 2025, step=1, value=2025, label="Year")

    with gr.Accordion("Macro indicators (adjust to scenario)", open=False):
        with gr.Row():
            inflation_input = gr.Number(value=28.2, label="Inflation (%)")
            exchange_input = gr.Number(value=1500, label="Exchange rate (₦/$)")
            gdp_input = gr.Number(value=3.0, label="GDP growth (%)")
        with gr.Row():
            inflation_lag_input = gr.Number(value=24.0, label="Inflation lag (prior quarter %)")
            exchange_lag_input = gr.Number(value=1200, label="Exchange rate lag (₦/$)")
            gdp_lag_input = gr.Number(value=2.5, label="GDP growth lag (%)")

    predict_button = gr.Button("Predict rent", variant="primary")
    summary_output = gr.Markdown()
    comps_output = gr.Dataframe(headers=["location", "bedrooms", "bathrooms", "area_sqm", "type", "price (₦)"])

    predict_button.click(
        run_inference,
        inputs=[
            location_input,
            property_type_input,
            area_zone_input,
            bedroom_input,
            bathroom_input,
            toilet_input,
            area_input,
            month_input,
            year_input,
            inflation_input,
            exchange_input,
            gdp_input,
            inflation_lag_input,
            exchange_lag_input,
            gdp_lag_input,
        ],
        outputs=[summary_output, comps_output],
    )

    gr.Markdown(
        "Model features: " + ", ".join(f"`{col}`" for col in FEATURE_COLUMNS)
    )


def main() -> None:
    demo.launch()


if __name__ == "__main__":
    main()
