"""Merge and harmonize the cleaned PNADC waves.

Two independent scopes are supported:

``higher-ed``
    All available ``PNADC_limpo_VD3004_7_<year>`` files for 2016-2019 and
    2022-2025.

``full``
    The two full cleaned files, ``PNADC_limpo_2023.dta`` and
    ``PNADC_limpo_2024.dta``.

The script standardizes raw IBGE variable-name capitalization, aligns columns,
adds a source-file field, and deflates all readable wage/income variables to
2025-Q3 prices.  Nominal variables are retained; harmonized variables receive
the suffix ``_real_2025q3``.  Following ``10_fillins_pnadc_fulltime.do``, the
factor is CO2(year, quarter, UF) / CO2(2025 Q3, UF).

Each scope is loaded and merged entirely in memory, then exported as one
compressed Parquet file.  This is intended for a high-memory compute node.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "This script requires pandas, pyarrow, and xlrd. Install them with "
        "`python -m pip install pandas pyarrow xlrd`."
    ) from exc


Scope = Literal["higher-ed", "full"]
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
                + ". Expected filtered VD3004 == 7 files for 2023/24 as well "
                "as the archived waves."
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


def logical_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return "datetime"
    if pd.api.types.is_string_dtype(series.dtype) or isinstance(
        series.dtype, pd.CategoricalDtype
    ):
        return "string"
    if pd.api.types.is_float_dtype(series.dtype):
        return "float"
    if pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_bool_dtype(
        series.dtype
    ):
        return "integer"
    return "string"


def combined_types(frames: list[pd.DataFrame]) -> OrderedDict[str, str]:
    types: OrderedDict[str, str] = OrderedDict()
    for frame in frames:
        for name in frame.columns:
            types[name] = promote_type(types.get(name), logical_type(frame[name]))
    return types


def read_source(source: Source) -> pd.DataFrame:
    print(f"  loading {source.year}: {source.path.name}")
    if source.path.suffix.lower() == ".dta":
        frame = pd.read_stata(
            source.path,
            convert_categoricals=False,
            preserve_dtypes=False,
        )
    else:
        frame = pd.read_parquet(source.path)
    print(f"    loaded {len(frame):,} rows and {len(frame.columns):,} columns")
    return frame


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


def prepare_source(
    frame: pd.DataFrame,
    source: Source,
    scope: Scope,
) -> pd.DataFrame:
    frame = canonicalize_columns(frame)
    if "ano" not in frame:
        frame["ano"] = source.year
    observed_year = pd.to_numeric(frame["ano"], errors="coerce")
    bad_year = observed_year.notna() & observed_year.ne(source.year)
    if bad_year.any():
        raise ValueError(f"{source.path.name} contains observations outside {source.year}")

    if scope == "higher-ed":
        education = next(
            (name for name in ("nivel_instrucao", "VD3004") if name in frame), None
        )
        if education is None:
            raise ValueError(f"No education variable in {source.path.name}")
        frame = frame.loc[
            pd.to_numeric(frame[education], errors="coerce").eq(7)
        ].copy()
    frame["source_file"] = source.path.name
    return frame


def cast_columns(
    frame: pd.DataFrame, column_types: OrderedDict[str, str]
) -> pd.DataFrame:
    for column, dtype in column_types.items():
        if dtype == "string":
            frame[column] = frame[column].astype("string")
        elif dtype == "integer":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
        elif dtype == "float":
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
        elif dtype == "datetime":
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def add_deflated_wages(
    frame: pd.DataFrame,
    column_types: OrderedDict[str, str],
    deflator: pd.DataFrame,
    source_names: str,
) -> tuple[pd.DataFrame, list[str]]:
    wages = wage_columns(column_types)

    keys = ["ano", "trimestre", "id_uf"]
    missing_keys = [key for key in keys if key not in frame]
    if missing_keys:
        raise ValueError(f"Missing deflator merge keys: {missing_keys}")
    for key in keys:
        frame[key] = pd.to_numeric(frame[key], errors="coerce").astype("Int64")

    frame = frame.merge(deflator, on=keys, how="left", validate="many_to_one")
    unmatched = frame["deflator_wage_2025q3"].isna()
    if unmatched.any():
        examples = frame.loc[unmatched, keys].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"Unmatched deflator keys in {source_names}: {examples}")

    for column in wages:
        nominal = pd.to_numeric(frame[column], errors="coerce")
        frame[f"{column}_real_2025q3"] = nominal * frame["deflator_wage_2025q3"]
    return frame, wages


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
    overwrite: bool,
    higher_ed_years: tuple[int, ...] = HIGHER_ED_YEARS,
) -> Path:
    sources = discover_sources(cleaned_dir, scope, higher_ed_years)
    print(f"\nScope: {scope}")
    for source in sources:
        print(f"  input {source.year}: {source.path.name}")

    frames = [prepare_source(read_source(source), source, scope) for source in sources]
    column_types = combined_types(frames)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    del frames
    combined = cast_columns(combined, column_types)
    combined, wages = add_deflated_wages(
        combined,
        column_types,
        deflator,
        ", ".join(source.path.name for source in sources),
    )

    destination = output_path(cleaned_dir, scope, sources)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Use --overwrite to replace it."
        )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    try:
        print(f"  writing {len(combined):,} merged rows to {destination.name}")
        combined.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Wrote {len(combined):,} rows and {len(combined.columns):,} columns: {destination}")
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
        help="Defaults to PNADC/Raw Data/deflator_PNADC_2025.xls.",
    )
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
    pnadc_dir = args.pnadc_dir.expanduser().resolve()
    cleaned_dir = pnadc_dir / "Cleaned Data"
    deflator_path = (
        args.deflator.expanduser().resolve()
        if args.deflator is not None
        else pnadc_dir / "Raw Data" / "deflator_PNADC_2025.xls"
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
            args.overwrite,
            tuple(sorted(set(args.higher_ed_years))),
        )


if __name__ == "__main__":
    main()
