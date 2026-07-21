import argparse
from pathlib import Path

import pandas as pd

# VD3004 is the variable that indicates complete tertiary education
# it's meant to make the download of the microdados more manageable, 
# since the full dataset is very large.

MICRODADOS_TABLE = "basedosdados.br_ibge_pnadc.microdados"
DEFAULT_VD3004 = 7


def default_pnadc_dir():
    """Assumes this file lives in PNADC/Code."""
    return Path(__file__).resolve().parent.parent


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_query(year, vd3004_value=DEFAULT_VD3004, sample_share=None, sample_modulus=10):
    where = [
        f"ano = {year}",
        f"SAFE_CAST(VD3004 AS INT64) = {vd3004_value}",
    ]
    if sample_share is not None:
        where.append(
            "MOD(ABS(FARM_FINGERPRINT(CONCAT(CAST(id_domicilio AS STRING), '_', CAST(V2003 AS STRING)))), "
            f"{sample_modulus}) = {sample_share}"
        )
    return f"SELECT * FROM `{MICRODADOS_TABLE}` WHERE " + " AND ".join(where)


def output_path(pnadc_dir, year, vd3004_value=DEFAULT_VD3004, sample_share=None):
    suffix = f"VD3004_{vd3004_value}_{year}"
    if sample_share is not None:
        suffix = f"VD3004_{vd3004_value}_sample_{year}"
    return pnadc_dir / "Raw Data" / f"PNADC_microdados_{suffix}.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Download PNADC microdados for one year, restricted to VD3004 == 7."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--vd3004-value", type=int, default=DEFAULT_VD3004)
    parser.add_argument("--sample-share", type=int, default=None)
    parser.add_argument("--sample-modulus", type=int, default=10)
    parser.add_argument("--billing-project-id", default="educacaoideias")
    parser.add_argument(
        "--pnadc-dir",
        type=Path,
        default=default_pnadc_dir(),
        help="PNADC project directory. Defaults to the parent of this script's Code directory.",
    )
    args = parser.parse_args()

    try:
        import basedosdados as bd
    except ImportError as exc:
        raise SystemExit("basedosdados is required for downloads. Install it in the active Python environment.") from exc

    pnadc_dir = args.pnadc_dir.expanduser().resolve()
    raw_dir = ensure_dir(pnadc_dir / "Raw Data")
    query = build_query(
        args.year,
        vd3004_value=args.vd3004_value,
        sample_share=args.sample_share,
        sample_modulus=args.sample_modulus,
    )

    print(f"PNADC directory: {pnadc_dir}")
    print(f"Raw data directory: {raw_dir}")
    df = bd.read_sql(query=query, billing_project_id=args.billing_project_id)
    df = pd.DataFrame(df)

    path = output_path(pnadc_dir, args.year, vd3004_value=args.vd3004_value, sample_share=args.sample_share)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
