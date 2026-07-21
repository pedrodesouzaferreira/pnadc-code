"""Merge and harmonize the cleaned PNADC waves.

Two independent scopes are supported:

``higher-ed``
    All available ``PNADC_limpo_VD3004_7_<year>`` files.  The local archive
    currently contains 2016-2019, 2022, and 2025 as Stata files; the temporary
    filtering step adds 2023 and 2024 as Parquet files.

``full``
    The two full cleaned files, ``PNADC_limpo_2023.dta`` and
    ``PNADC_limpo_2024.dta``.

The script standardizes raw IBGE variable-name capitalization, aligns columns,
adds a source-file field, and deflates all readable wage/income variables to
2025-Q3 prices.  Nominal variables are retained; harmonized variables receive
the suffix ``_real_2025q3``.  Following ``11_fillins_pnadc_fulltime.do``, the
factor is CO2(year, quarter, UF) / CO2(2025 Q3, UF).

Outputs are compressed Parquet files written in chunks, so the complete merge
does not have to fit in memory.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyreadstat
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "This script requires pandas, numpy, pyarrow, pyreadstat, and xlrd. "
        "Install them with `python -m pip install pandas numpy pyarrow "
        "pyreadstat xlrd`."
    ) from exc


Scope = Literal["higher-ed", "full"]
DEFAULT_CHUNK_SIZE = 25_000
FULL_YEARS = (2023, 2024)
HIGHER_ED_YEARS = (2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025)
RAW_IBGE_NAME = re.compile(r"^(vd?)(\d.*)$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class Source:
    path: Path
    year: int


def default_pnadc_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def canonical_name(name: str) -> str:
    """Canonicalize raw v*/vd* IBGE names while keeping readable names."""
    match = RAW_IBGE_NAME.match(name)
    if not match:
        return name
    prefix, remainder = match.groups()
    return prefix.upper() + remainder


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: canonical_name(str(column)) for column in frame.columns}
    canonical = list(renamed.values())
    duplicates = sorted({name for name in canonical if canonical.count(name) > 1})
    if duplicates:
        raise ValueError(f"Canonical variable-name collision: {duplicates}")
    return frame.rename(columns=renamed)


def extract_year(path: Path) -> int:
    match = re.search(r"_(20\d{2})\.(?:dta|parquet)$", path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer year from {path.name}")
    return int(match.group(1))


def discover_sources(
    cleaned_dir: Path,
    scope: Scope,
    higher_ed_years: tuple[int, ...] = HIGHER_ED_YEARS,
) -> list[Source]:
    if scope == "full":
        paths = [cleaned_dir / f"PNADC_limpo_{year}.dta" for year in FULL_YEARS]
    else:
        candidates = [
            path
            for path in cleaned_dir.glob("PNADC_limpo_VD3004_7_*")
            if path.suffix.lower() in {".dta", ".parquet"}
            and "_sample_" not in path.name
        ]
        by_year: dict[int, Path] = {}
        for path in sorted(candidates):
            year = extract_year(path)
            previous = by_year.get(year)
            if previous is None or (previous.suffix == ".dta" and path.suffix == ".parquet"):
                by_year[year] = path
        missing_years = sorted(set(higher_ed_years).difference(by_year))
        if missing_years:
            raise FileNotFoundError(
                "Missing higher-education input years: "
                + ", ".join(map(str, missing_years))
                + ". Run 2_filter_vd3004_7_2023_2024.py after making the full "
                "2023/24 files available offline."
            )
        paths = [by_year[year] for year in sorted(set(higher_ed_years))]

    missing = [path for path in paths if not path.exists()]
    empty = [path for path in paths if path.exists() and path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing input(s): " + ", ".join(map(str, missing)))
    if empty:
        raise RuntimeError(
            "Empty input(s), probably Dropbox online-only placeholders: "
            + ", ".join(map(str, empty))
            + ". Make them available offline and retry."
        )
    if not paths:
        raise FileNotFoundError(f"No inputs found for scope {scope!r} in {cleaned_dir}")
    return [Source(path=path, year=extract_year(path)) for path in paths]


def source_types(path: Path) -> OrderedDict[str, str]:
    """Return canonical column names and portable logical types."""
    result: OrderedDict[str, str] = OrderedDict()
    if path.suffix.lower() == ".dta":
        _, metadata = pyreadstat.read_dta(path, metadataonly=True)
        raw_types = metadata.readstat_variable_types
        for raw_name in metadata.column_names:
            readstat_type = raw_types[raw_name]
            if readstat_type == "string":
                logical = "string"
            elif readstat_type in {"double", "float"}:
                logical = "float"
            else:
                logical = "integer"
            result[canonical_name(raw_name)] = logical
    else:
        schema = pq.ParquetFile(path).schema_arrow
        for field in schema:
            if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
                logical = "string"
            elif pa.types.is_floating(field.type):
                logical = "float"
            elif pa.types.is_integer(field.type) or pa.types.is_boolean(field.type):
                logical = "integer"
            elif pa.types.is_timestamp(field.type) or pa.types.is_date(field.type):
                logical = "datetime"
            else:
                logical = "string"
            result[canonical_name(field.name)] = logical
    if len(result) != len(set(result)):
        raise ValueError(f"Canonical variable-name collision in {path}")
    return result


def promote_type(left: str | None, right: str) -> str:
    if left is None or left == right:
        return right
    if "string" in {left, right}:
        return "string"
    if "datetime" in {left, right}:
        return "datetime" if left == right else "string"
    if "float" in {left, right}:
        return "float"
    return "integer"


def combined_types(sources: list[Source]) -> OrderedDict[str, str]:
    types: OrderedDict[str, str] = OrderedDict()
    for source in sources:
        for name, logical_type in source_types(source.path).items():
            types[name] = promote_type(types.get(name), logical_type)
    return types


def arrow_type(logical_type: str) -> pa.DataType:
    return {
        "string": pa.string(),
        "float": pa.float64(),
        "integer": pa.int64(),
        "datetime": pa.timestamp("ns"),
    }[logical_type]


def source_chunks(source: Source, chunk_size: int) -> Iterator[pd.DataFrame]:
    if source.path.suffix.lower() == ".dta":
        yield from pd.read_stata(
            source.path,
            chunksize=chunk_size,
            convert_categoricals=False,
            preserve_dtypes=False,
        )
    else:
        parquet_file = pq.ParquetFile(source.path)
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()


def load_deflator(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Deflator file is missing or empty: {path}")
    frame = pd.read_excel(path, sheet_name="deflator")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    frame = frame.rename(columns={"trim": "trimestre", "uf": "id_uf"})
    required = {"ano", "trimestre", "id_uf", "co2"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Deflator lacks required variables: {sorted(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required)).copy()
    frame[["ano", "trimestre", "id_uf"]] = frame[
        ["ano", "trimestre", "id_uf"]
    ].astype("int64")

    target = (
        frame.loc[(frame["ano"] == 2025) & (frame["trimestre"] == 3), ["id_uf", "co2"]]
        .drop_duplicates("id_uf")
        .rename(columns={"co2": "co2_2025q3"})
    )
    if len(target) != 27:
        raise ValueError(f"Expected 27 UF-specific 2025-Q3 targets; found {len(target)}")

    result = frame[["ano", "trimestre", "id_uf", "co2"]].merge(
        target, on="id_uf", how="left", validate="many_to_one"
    )
    result["deflator_wage_2025q3"] = result["co2"] / result["co2_2025q3"]
    if result.duplicated(["ano", "trimestre", "id_uf"]).any():
        raise ValueError("Deflator contains duplicate year-quarter-UF keys")
    return result


def wage_columns(column_types: OrderedDict[str, str]) -> list[str]:
    columns = []
    for column, logical_type in column_types.items():
        lowered = column.lower()
        if logical_type in {"integer", "float"} and (
            lowered.startswith("renda_") or lowered.startswith("valor_dinheiro_")
        ) and not lowered.startswith("log_"):
            columns.append(column)
    return columns


def harmonize_chunk(
    chunk: pd.DataFrame,
    source: Source,
    column_types: OrderedDict[str, str],
    wages: list[str],
    deflator: pd.DataFrame,
    scope: Scope,
) -> pd.DataFrame:
    chunk = canonicalize_columns(chunk)
    if "ano" not in chunk:
        chunk["ano"] = source.year
    observed_year = pd.to_numeric(chunk["ano"], errors="coerce")
    bad_year = observed_year.notna() & observed_year.ne(source.year)
    if bad_year.any():
        raise ValueError(f"{source.path.name} contains observations outside {source.year}")

    if scope == "higher-ed":
        education = next(
            (name for name in ("nivel_instrucao", "VD3004") if name in chunk), None
        )
        if education is None:
            raise ValueError(f"No education variable in {source.path.name}")
        chunk = chunk.loc[pd.to_numeric(chunk[education], errors="coerce").eq(7)].copy()

    for column in column_types:
        if column not in chunk:
            chunk[column] = pd.NA
    chunk = chunk[list(column_types)]

    keys = ["ano", "trimestre", "id_uf"]
    missing_keys = [key for key in keys if key not in chunk]
    if missing_keys:
        raise ValueError(f"Missing deflator merge keys in {source.path.name}: {missing_keys}")
    for key in keys:
        chunk[key] = pd.to_numeric(chunk[key], errors="coerce").astype("Int64")

    chunk = chunk.merge(deflator, on=keys, how="left", validate="many_to_one")
    unmatched = chunk["deflator_wage_2025q3"].isna()
    if unmatched.any():
        examples = chunk.loc[unmatched, keys].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"Unmatched deflator keys in {source.path.name}: {examples}")

    for column in wages:
        nominal = pd.to_numeric(chunk[column], errors="coerce")
        chunk[f"{column}_real_2025q3"] = nominal * chunk["deflator_wage_2025q3"]
    chunk["source_file"] = source.path.name
    return chunk


def cast_to_schema(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    prepared: dict[str, pd.Series] = {}
    for field in schema:
        series = frame[field.name]
        if pa.types.is_string(field.type):
            prepared[field.name] = series.astype("string")
        elif pa.types.is_integer(field.type):
            prepared[field.name] = pd.to_numeric(series, errors="coerce").astype("Int64")
        elif pa.types.is_floating(field.type):
            prepared[field.name] = pd.to_numeric(series, errors="coerce").astype("Float64")
        elif pa.types.is_timestamp(field.type):
            prepared[field.name] = pd.to_datetime(series, errors="coerce")
        else:  # pragma: no cover - schema is constructed above
            prepared[field.name] = series
    return pa.Table.from_pandas(pd.DataFrame(prepared), schema=schema, preserve_index=False)


def output_path(cleaned_dir: Path, scope: Scope, sources: list[Source]) -> Path:
    years = [source.year for source in sources]
    if scope == "higher-ed":
        stem = f"PNADC_harmonized_VD3004_7_{min(years)}_{max(years)}"
    else:
        stem = f"PNADC_harmonized_full_{min(years)}_{max(years)}"
    return cleaned_dir / f"{stem}.parquet"


def run_scope(
    cleaned_dir: Path,
    deflator: pd.DataFrame,
    scope: Scope,
    chunk_size: int,
    overwrite: bool,
    higher_ed_years: tuple[int, ...] = HIGHER_ED_YEARS,
) -> Path:
    sources = discover_sources(cleaned_dir, scope, higher_ed_years)
    print(f"\nScope: {scope}")
    for source in sources:
        print(f"  input {source.year}: {source.path.name}")

    column_types = combined_types(sources)
    wages = wage_columns(column_types)
    derived_types = OrderedDict(column_types)
    derived_types["co2"] = "float"
    derived_types["co2_2025q3"] = "float"
    derived_types["deflator_wage_2025q3"] = "float"
    for wage in wages:
        derived_types[f"{wage}_real_2025q3"] = "float"
    derived_types["source_file"] = "string"
    schema = pa.schema(
        [pa.field(name, arrow_type(logical_type)) for name, logical_type in derived_types.items()]
    )

    destination = output_path(cleaned_dir, scope, sources)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Use --overwrite to replace it."
        )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    writer = pq.ParquetWriter(
        temporary,
        schema,
        compression="zstd",
        use_dictionary=True,
    )
    total_rows = 0
    try:
        for source in sources:
            source_rows = 0
            for chunk_number, chunk in enumerate(source_chunks(source, chunk_size), 1):
                chunk = harmonize_chunk(
                    chunk, source, column_types, wages, deflator, scope
                )
                if chunk.empty:
                    continue
                table = cast_to_schema(chunk, schema)
                writer.write_table(table)
                source_rows += len(chunk)
                total_rows += len(chunk)
                print(
                    f"  {source.year}: chunk {chunk_number:,}; "
                    f"source rows {source_rows:,}; total {total_rows:,}"
                )
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()

    print(f"Wrote {total_rows:,} rows and {len(schema):,} columns: {destination}")
    print(f"Deflated variables ({len(wages)}): {', '.join(wages)}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=("both", "higher-ed", "full"),
        default="both",
        help="Dataset collection(s) to harmonize.",
    )
    parser.add_argument("--pnadc-dir", type=Path, default=default_pnadc_dir())
    parser.add_argument(
        "--deflator",
        type=Path,
        default=None,
        help="Defaults to PNADC/Notas Técnicas/deflator_PNADC_2025.xls.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--higher-ed-years",
        nargs="+",
        type=int,
        default=list(HIGHER_ED_YEARS),
        help="Expected higher-education waves; missing years are an error.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    pnadc_dir = args.pnadc_dir.expanduser().resolve()
    cleaned_dir = pnadc_dir / "Cleaned Data"
    deflator_path = (
        args.deflator.expanduser().resolve()
        if args.deflator is not None
        else pnadc_dir / "Notas Técnicas" / "deflator_PNADC_2025.xls"
    )
    print(f"PNADC directory: {pnadc_dir}")
    print(f"Deflator: {deflator_path}")
    deflator = load_deflator(deflator_path)

    scopes: tuple[Scope, ...]
    scopes = ("higher-ed", "full") if args.scope == "both" else (args.scope,)
    for scope in scopes:
        run_scope(
            cleaned_dir,
            deflator,
            scope,
            args.chunk_size,
            args.overwrite,
            tuple(sorted(set(args.higher_ed_years))),
        )


if __name__ == "__main__":
    main()
