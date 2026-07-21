"""Temporarily extract complete-tertiary-education observations for 2023/24.

The full cleaned PNADC Stata files are very large.  This script reads them in
chunks, keeps ``VD3004 == 7`` (renamed ``nivel_instrucao`` by the cleaning
pipeline), and writes one compressed Parquet file per year.  Parquet is used
because a Stata file cannot be appended safely chunk by chunk.

Default inputs
--------------
Cleaned Data/PNADC_limpo_2023.dta
Cleaned Data/PNADC_limpo_2024.dta

Default outputs
---------------
Cleaned Data/PNADC_limpo_VD3004_7_2023.parquet
Cleaned Data/PNADC_limpo_VD3004_7_2024.parquet

Run from any directory with, for example::

    python 2_filter_vd3004_7_2023_2024.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterator

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "This script requires pandas, pyarrow, and pyreadstat. Install them with "
        "`python -m pip install pandas pyarrow pyreadstat`."
    ) from exc


DEFAULT_YEARS = (2023, 2024)
DEFAULT_CHUNK_SIZE = 25_000


def default_pnadc_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def stata_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_stata(
        path,
        chunksize=chunk_size,
        convert_categoricals=False,
        preserve_dtypes=False,
    )


def education_column(columns: list[str]) -> str:
    lookup = {column.lower(): column for column in columns}
    for candidate in ("nivel_instrucao", "vd3004"):
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(
        "Input has neither `nivel_instrucao` nor `VD3004`; cannot apply the "
        "complete-tertiary-education filter."
    )


def filter_year(
    input_path: Path,
    output_path: Path,
    year: int,
    chunk_size: int,
    overwrite: bool,
) -> tuple[int, int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if input_path.stat().st_size == 0:
        raise RuntimeError(
            f"Input is empty: {input_path}\n"
            "If this is a Dropbox online-only placeholder, make it available "
            "offline and run the script again."
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    rows_read = 0
    rows_kept = 0
    try:
        for chunk_number, chunk in enumerate(stata_chunks(input_path, chunk_size), 1):
            rows_read += len(chunk)
            column = education_column(chunk.columns.tolist())
            selected = chunk.loc[
                pd.to_numeric(chunk[column], errors="coerce").eq(7)
            ].copy()
            rows_kept += len(selected)

            if selected.empty:
                print(
                    f"  {year}: chunk {chunk_number:,}; "
                    f"read {rows_read:,}, kept {rows_kept:,}"
                )
                continue

            table = pa.Table.from_pandas(selected, preserve_index=False)
            if writer is None:
                output_schema = table.schema
                writer = pq.ParquetWriter(
                    temporary_path,
                    output_schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            elif not table.schema.equals(output_schema, check_metadata=False):
                table = table.cast(output_schema)
            writer.write_table(table)
            print(
                f"  {year}: chunk {chunk_number:,}; "
                f"read {rows_read:,}, kept {rows_kept:,}"
            )

        if writer is None:
            raise RuntimeError(f"No observations with VD3004 == 7 found in {input_path}")
        writer.close()
        writer = None
        os.replace(temporary_path, output_path)
    finally:
        if writer is not None:
            writer.close()
        if temporary_path.exists():
            temporary_path.unlink()

    return rows_read, rows_kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS))
    parser.add_argument("--pnadc-dir", type=Path, default=default_pnadc_dir())
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--input-template",
        default="PNADC_limpo_{year}.dta",
        help="Filename template inside Cleaned Data (use {year}).",
    )
    parser.add_argument(
        "--output-template",
        default="PNADC_limpo_VD3004_7_{year}.parquet",
        help="Filename template inside Cleaned Data (use {year}).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")

    cleaned_dir = args.pnadc_dir.expanduser().resolve() / "Cleaned Data"
    print(f"Cleaned-data directory: {cleaned_dir}")
    for year in sorted(set(args.years)):
        input_path = cleaned_dir / args.input_template.format(year=year)
        output_path = cleaned_dir / args.output_template.format(year=year)
        print(f"Filtering {input_path.name} -> {output_path.name}")
        rows_read, rows_kept = filter_year(
            input_path=input_path,
            output_path=output_path,
            year=year,
            chunk_size=args.chunk_size,
            overwrite=args.overwrite,
        )
        print(
            f"Finished {year}: {rows_kept:,} of {rows_read:,} rows kept "
            f"({rows_kept / rows_read:.2%})."
        )


if __name__ == "__main__":
    main()
