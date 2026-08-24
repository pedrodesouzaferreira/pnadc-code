"""Export a lightweight CSV subset of the harmonized PNADC panel.

Reads the harmonized file(s) produced by 2_harmonize.py -- Parquet if
present, otherwise the CSV copy -- and writes a much smaller CSV with only
the identifying, demographic, employment, and job-search variables needed
for everyday analysis, small enough to run on a laptop. For each nominal
wage/income variable in that subset, both the nominal column and its
``_real_2025q3`` deflated counterpart are included.

The subset also carries the raw occupation/position variables (V4010,
V4025, posicao_trab_principal, servidor_publico_estatutario,
carteira_assinada) needed to reproduce the ``define_public_sector``
redefinition of the public sector used by 10_fillins_pnadc_fulltime.do --
the harmonized ``empregado_setor_pub`` flag alone is a different, broader
definition -- and the deflator itself (co2, co2_2025q3,
deflator_wage_2025q3) so the applied factor can be checked directly.

Usage
-----
    python 2a_harmonize_makelighter.py --sample higher-ed
    python 2a_harmonize_makelighter.py --sample full
    python 2a_harmonize_makelighter.py --sample both

Requires a harmonized Parquet or CSV file already produced by 2_harmonize.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from harmonized_data import HARMONIZED_FILES, SAMPLES

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "Cleaned Data"

BASE_COLUMNS = [
    "ano", "trimestre", "id_uf", "id_pessoa", "id_domicilio", "peso", "idade",
    "V3004", "servidor_publico_estatutario", "carteira_assinada",
    # Occupation/position variables needed to reproduce the
    # define_public_sector redefinition in 10_fillins_pnadc_fulltime.do.
    "V4010", "V4025", "posicao_trab_principal",
    "valor_dinheiro_principal", "valor_dinheiro_efetivo_principal",
    "horas_habituais_principal", "horas_efetivas_principal",
    "tempo_nesse_trabalho", "tomou_providencia_busca",
    "metodo_busca_emprego_v1", "metodo_busca_emprego",
    "gostaria_ter_trabalhado", "motivo_nao_buscou_v1", "motivo_nao_buscou",
    "tempo_desempregado", "renda_habitual_principal", "renda_efetiva_principal",
    "horas_habituais_todos_faixa", "horas_efetivas_todos_faixa",
    "ocupado", "desocupado", "na_pea", "servidor_publico",
    "buscando_via_concurso", "formal", "informal",
    "empregado_setor_priv", "empregado_setor_pub", "conta_propria",
    "empregador", "trab_domestico", "trab_familiar_aux",
    "faixa_etaria", "idade_trabalho", "log_renda", "faixa_salarial",
    "renda_habitual_principal_winsor", "log_renda_hab_princ_winsor",
]

# Nominal wage/income columns in BASE_COLUMNS that 2_harmonize.py also
# deflates; both versions are exported.
WAGE_COLUMNS = [
    "valor_dinheiro_principal",
    "valor_dinheiro_efetivo_principal",
    "renda_habitual_principal",
    "renda_efetiva_principal",
    "renda_habitual_principal_winsor",
]

# The deflator itself (see 2_harmonize.py's load_deflator/add_deflated_wages),
# carried through so the applied factor can be checked directly.
DEFLATOR_COLUMNS = ["co2", "co2_2025q3", "deflator_wage_2025q3"]


def requested_columns() -> list[str]:
    deflated = [f"{column}_real_2025q3" for column in WAGE_COLUMNS]
    return BASE_COLUMNS + deflated + DEFLATOR_COLUMNS


def resolve_harmonized_input(cleaned_dir: Path, sample: str) -> tuple[Path, str]:
    """Locate the harmonized input for `sample`, preferring Parquet over CSV."""
    if sample not in HARMONIZED_FILES:
        raise ValueError(f"Unknown sample {sample!r}; choose from {SAMPLES}")
    stem = Path(HARMONIZED_FILES[sample]).stem
    for suffix, kind in ((".parquet", "parquet"), (".csv", "csv")):
        path = cleaned_dir / f"{stem}{suffix}"
        if path.exists() and path.stat().st_size > 0:
            return path, kind
    raise FileNotFoundError(
        f"No harmonized {sample!r} input (.parquet or .csv) in {cleaned_dir}. "
        f"Run `python 2_harmonize.py --sample {sample}` first."
    )


def available_columns(path: Path, kind: str) -> set[str]:
    if kind == "parquet":
        return set(pq.ParquetFile(path).schema_arrow.names)
    return set(pd.read_csv(path, nrows=0).columns)


def read_selected(path: Path, kind: str, columns: list[str]) -> pd.DataFrame:
    print(f"Loading harmonized input: {path.name}")
    if kind == "parquet":
        frame = pd.read_parquet(path, columns=columns)
    else:
        frame = pd.read_csv(path, usecols=columns, low_memory=False)
    print(f"Loaded {len(frame):,} rows and {len(frame.columns):,} columns")
    return frame


def light_output_path(cleaned_dir: Path, sample: str) -> Path:
    stem = Path(HARMONIZED_FILES[sample]).stem
    return cleaned_dir / f"{stem}_light.csv"


def export_sample(cleaned_dir: Path, sample: str, overwrite: bool) -> Path:
    path, kind = resolve_harmonized_input(cleaned_dir, sample)
    available = available_columns(path, kind)
    wanted = requested_columns()
    missing = [column for column in wanted if column not in available]
    if missing:
        print(f"  ({sample}) not in {path.name}, skipping: {', '.join(missing)}")
    present = [column for column in wanted if column in available]

    frame = read_selected(path, kind, present)

    destination = light_output_path(cleaned_dir, sample)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Use --overwrite to replace it."
        )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Wrote {len(frame):,} rows and {len(frame.columns):,} columns to {destination}")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sample", choices=("both",) + SAMPLES, default="both")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = SAMPLES if args.sample == "both" else (args.sample,)
    for sample in samples:
        print(f"\nSample: {sample}")
        export_sample(CLEANED_DIR, sample, args.overwrite)


if __name__ == "__main__":
    main()
