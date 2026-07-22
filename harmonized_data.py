"""Shared input selection for analyses of harmonized PNADC data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


SAMPLES = ("full", "higher-ed")
HARMONIZED_FILES = {
    "full": "PNADC_harmonized_full_2023_2024.parquet",
    "higher-ed": "PNADC_harmonized_VD3004_7_2016_2025.parquet",
}


def add_sample_argument(parser, default: str = "full") -> None:
    parser.add_argument(
        "--sample",
        choices=SAMPLES,
        default=default,
        help="Use the harmonized full sample or the VD3004 == 7 sample.",
    )


def harmonized_path(cleaned_dir: Path, sample: str) -> Path:
    if sample not in HARMONIZED_FILES:
        raise ValueError(f"Unknown sample {sample!r}; choose from {SAMPLES}")
    path = cleaned_dir / HARMONIZED_FILES[sample]
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Harmonized {sample!r} input is missing or empty: {path}. "
            f"Run `python 2_harmonize.py --sample {sample}` first."
        )
    return path


def load_harmonized(
    cleaned_dir: Path,
    sample: str,
    columns: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
) -> pd.DataFrame:
    path = harmonized_path(cleaned_dir, sample)
    available = set(pq.ParquetFile(path).schema_arrow.names)
    requested = list(columns) if columns is not None else None
    use_columns = [column for column in requested if column in available] if requested else None
    if requested:
        essential = {"id_pessoa", "ano", "trimestre"}.intersection(requested)
        missing = sorted(essential.difference(available))
        if missing:
            raise ValueError(f"{path.name} lacks required panel columns: {missing}")

    selected_years = sorted(set(years)) if years is not None else None
    filters = [("ano", "in", selected_years)] if selected_years else None
    print(f"Loading harmonized sample={sample}: {path.name}")
    frame = pd.read_parquet(path, columns=use_columns, filters=filters)
    print(f"Loaded {len(frame):,} rows and {len(frame.columns):,} columns")
    return frame
