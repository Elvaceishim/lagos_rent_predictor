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

### Model artifacts and automated download

The UI expects two artifacts inside `models/`:

- `trained_model_pipeline.joblib`
- `lambda_boxcox.joblib`

If you prefer not to commit these binaries, upload them to a Hugging Face dataset (or private model repo) and set:

```bash
export MODEL_REPO_ID=theelvace/lagos-rent-models  # replace with your repo
export MODEL_REPO_TYPE=dataset                     # or \"model\" if applicable
export HF_TOKEN=hf_xxx                              # only needed for private repos
```

`app/predict.py` will automatically download the missing files at startup and copy them into `models/`. Without `MODEL_REPO_ID`, it simply falls back to whatever is already on disk.

### Refresh comparable listings

`data/combined_properties.csv` powers the dropdown choices and comps table. Regenerate it from the source scrapes with:

```bash
make refresh-data
```

This concatenates `data/propertypro.csv`, `data/properties.csv`, and `data/privateproperty.csv` (if present) and writes the combined file.

### Smoke test / CI hook

Run a fast end-to-end check that ensures the artifacts load and the pipeline returns a prediction:

```bash
make smoke-test
```

This is handy for CI or before pushing to Hugging Face to verify dependencies/artifacts are wired correctly.

### Deploying to Hugging Face Spaces

1. Push the latest code to both GitHub and the Space repo (`git push hf main`).
2. Upload the `.joblib` artifacts to the storage that matches `MODEL_REPO_ID` (or drop them into the Space’s Files tab).
3. Trigger **Restart** / **Rebuild** on the Space dashboard. The startup logs will show the auto-download step if the models were fetched from the Hub.

You can upload the artifacts programmatically with:

```bash
python -m scripts.upload_models
```

which copies the files under `models/` to `theelvace/lagos_rent_predictor` on Hugging Face.
