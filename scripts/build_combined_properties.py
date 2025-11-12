"""Utility to (re)build data/combined_properties.csv from available source CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "combined_properties.csv"
SOURCE_FILES: Dict[str, Path] = {
    "propertypro": DATA_DIR / "propertypro.csv",
    "properties": DATA_DIR / "properties.csv",
    "privateproperty": DATA_DIR / "privateproperty.csv",
}


def load_source(name: str, path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "source" not in df.columns:
        df["source"] = name
    else:
        df["source"] = df["source"].fillna(name)
    return df


def build_combined(output: Path = DEFAULT_OUTPUT) -> Path:
    frames = [load_source(name, path) for name, path in SOURCE_FILES.items()]
    frames = [frame for frame in frames if frame is not None]
    if not frames:
        raise FileNotFoundError("No source CSV files found in data/ directory.")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # Keep a consistent column order if the files share these fields.
    preferred_columns = [
        "location",
        "bedrooms",
        "bathrooms",
        "toilets",
        "area_sqm",
        "type",
        "price_naira",
        "source",
    ]
    ordered_cols = [col for col in preferred_columns if col in combined.columns]
    remainder = [col for col in combined.columns if col not in ordered_cols]
    combined = combined[ordered_cols + remainder]

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV (default: data/combined_properties.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_combined(args.output)
    print(f"Wrote {result}")


if __name__ == "__main__":
    main()
