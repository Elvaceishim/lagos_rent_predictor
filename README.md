---
title: Lagos Rent Predictor
emoji: 🐢
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app/ui.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Lagos Rent Predictor

This project contains the data scrapers, training notebooks, and a Gradio UI for estimating annual rents on the Lagos mainland.

### Run the Gradio app locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app/ui.py
```

The app loads the serialized pipeline in `models/trained_model_pipeline.joblib` and uses `data/combined_properties.csv` to populate dropdowns and display comparable listings.
