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

### Confidence bands via conformal residuals

The UI now surfaces a ± rent band with ~90 % empirical coverage. The scale is computed from the absolute residuals stored in `data/true_vs_predicted_naira.csv`.

1. After training in Colab (or any notebook), export a CSV with two columns: `y_true_naira` and `y_pred_naira` for a held-out validation set.
2. Copy that file into `data/true_vs_predicted_naira.csv` and rebuild the Space or restart the local app.
3. `app/predict.py` automatically loads the file, takes the 90th percentile of `|y_true - y_pred|`, and uses it to display the confidence band in the UI.

If the CSV is absent or malformed, the app falls back to a baked-in scale (`DEFAULT_CONFORMAL_SCALE` in `app/predict.py`). Update the CSV (or the constant) whenever you retrain so the coverage stays calibrated.
