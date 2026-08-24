"""Export a lightweight CSV subset of the harmonized PNADC panel.

Reads the harmonized Parquet file(s) produced by 2_harmonize.py and writes a
much smaller CSV with only the identifying, demographic, employment, and
job-search variables needed for everyday analysis -- small enough to run on
a laptop. For each nominal wage/income variable in that subset, both the
nominal column and its ``_real_2025q3`` deflated counterpart are included.

Usage
-----
    python 2a_harmonize_makelighter.py --sample higher-ed
    python 2a_harmonize_makelighter.py --sample full
    python 2a_harmonize_makelighter.py --sample both

Requires the harmonized Parquet file(s) already produced by 2_harmonize.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pyarrow.parquet as pq

from harmonized_data import SAMPLES, harmonized_path, load_harmonized

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "Cleaned Data"

BASE_COLUMNS = [
    "ano", "trimestre", "id_uf", "id_pessoa", "id_domicilio", "peso", "idade",
    "V3004", "servidor_publico_estatutario", "carteira_assinada",
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


def requested_columns() -> list[str]:
    deflated = [f"{column}_real_2025q3" for column in WAGE_COLUMNS]
    return BASE_COLUMNS + deflated


def light_output_path(cleaned_dir: Path, sample: str) -> Path:
    stem = harmonized_path(cleaned_dir, sample).stem
    return cleaned_dir / f"{stem}_light.csv"


def export_sample(cleaned_dir: Path, sample: str, overwrite: bool) -> Path:
    path = harmonized_path(cleaned_dir, sample)
    available = set(pq.ParquetFile(path).schema_arrow.names)
    wanted = requested_columns()
    missing = [column for column in wanted if column not in available]
    if missing:
        print(f"  ({sample}) not in {path.name}, skipping: {', '.join(missing)}")

    frame = load_harmonized(cleaned_dir, sample, columns=wanted)

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
