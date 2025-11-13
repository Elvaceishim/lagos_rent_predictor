"""Upload local model artifacts to the configured Hugging Face Space."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import upload_file

REPO_ID = "theelvace/lagos_rent_predictor"
FILES = {
    Path("models/trained_model_pipeline.joblib"): "trained_model_pipeline.joblib",
    Path("models/lambda_boxcox.joblib"): "lambda_boxcox.joblib",
}


def main() -> None:
    for local_path, remote_name in FILES.items():
        if not local_path.exists():
            raise SystemExit(f"Missing artifact: {local_path}")
        upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=REPO_ID,
            repo_type="space",
        )
        print(f"Uploaded {local_path} -> {REPO_ID}/{remote_name}")


if __name__ == "__main__":
    main()
