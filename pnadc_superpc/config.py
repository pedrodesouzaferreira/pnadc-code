from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
CODE_SUPERPC_DIR = PACKAGE_DIR.parent
PNADC_DIR = CODE_SUPERPC_DIR.parent
RAW_DIR = PNADC_DIR / "Raw Data"
CLEANED_DIR = PNADC_DIR / "Cleaned Data"
OUTPUT_DIR = PNADC_DIR / "Output"
SUPERPC_OUTPUT_DIR = OUTPUT_DIR / "SuperPC"


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs(tag=None):
    base = SUPERPC_OUTPUT_DIR if not tag else SUPERPC_OUTPUT_DIR / tag
    tables = ensure_dir(base / "Tables")
    figures = ensure_dir(base / "Figures")
    return tables, figures


def preferred_raw_path(dataset, year=None, use_sample=False):
    candidates = []
    if dataset == "dictionary":
        candidates = [RAW_DIR / "PNADC_dicionario.csv"]
    elif dataset == "microdados":
        if use_sample:
            candidates.extend(
                [
                    RAW_DIR / f"PNADC_microdados_sample_{year}.csv",
                    RAW_DIR / "PNADC_microdados_sample.csv",
                ]
            )
        if year is not None:
            candidates.append(RAW_DIR / f"PNADC_microdados_{year}.csv")
    elif dataset == "educacao":
        if use_sample:
            candidates.extend(
                [
                    RAW_DIR / f"PNADC_educacao_sample_{year}.csv",
                    RAW_DIR / "PNADC_educacao_sample.csv",
                ]
            )
        if year is not None:
            candidates.extend(
                [
                    RAW_DIR / f"PNADC_educacao_{year}.csv",
                    RAW_DIR / "PNADC_educacao.csv",
                ]
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if candidates:
        return candidates[0]
    raise ValueError(f"Unknown dataset: {dataset}")


def cleaned_path(dataset, year, use_sample=False):
    if dataset == "microdados":
        name = f"PNADC_limpo_{year}.csv"
        if use_sample:
            name = f"PNADC_limpo_sample_{year}.csv"
        return CLEANED_DIR / name
    if dataset == "educacao":
        name = f"PNADC_educacao_limpo_{year}.csv"
        if use_sample:
            name = f"PNADC_educacao_limpo_sample_{year}.csv"
        return CLEANED_DIR / name
    raise ValueError(f"Unknown dataset: {dataset}")
