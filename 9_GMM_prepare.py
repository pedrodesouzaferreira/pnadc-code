from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "Cleaned Data"
OUTPUT_PATH = CLEANED_DIR / "PNADC_prepared_for_GMM_v4.parquet"
YEARS = [2023, 2024]
HOURS_MIN = 30  # habitual weekly hours threshold for full-time classification
COLUMNS_TO_KEEP = [
    "id_pessoa",
    "ano",
    "trimestre",
    "renda_habitual_principal",
    "horas_habituais_principal",
    "empregado_setor_pub",
    "empregado_setor_priv",
    "desocupado",
    "conta_propria",
    "empregador",
    "trab_domestico",
    "trab_familiar_aux",
]


def load_year(year):
    path = CLEANED_DIR / f"PNADC_limpo_{year}.csv"
    return pd.read_csv(path, usecols=COLUMNS_TO_KEEP, low_memory=False)


def build_sector(df):
    public = pd.to_numeric(df["empregado_setor_pub"], errors="coerce").fillna(0).eq(1)
    private = (
        pd.to_numeric(df["empregado_setor_priv"], errors="coerce").fillna(0).eq(1)
        | pd.to_numeric(df["conta_propria"], errors="coerce").fillna(0).eq(1)
        | pd.to_numeric(df["empregador"], errors="coerce").fillna(0).eq(1)
        | pd.to_numeric(df["trab_domestico"], errors="coerce").fillna(0).eq(1)
    )
    return np.select(
        [public, private],
        ["public", "private"],
        default="unemployed",
    )


def main():
    frames = [load_year(year) for year in YEARS]
    df = pd.concat(frames, ignore_index=True)

    for column in ["ano", "trimestre", "renda_habitual_principal", "horas_habituais_principal"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in [
        "empregado_setor_pub",
        "empregado_setor_priv",
        "desocupado",
        "conta_propria",
        "empregador",
        "trab_domestico",
        "trab_familiar_aux",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype("int8")

    df["empregado_setor_priv"] = (
        df["empregado_setor_priv"]
        | df["conta_propria"]
        | df["empregador"]
        | df["trab_domestico"]
    ).astype("int8")
    df["desocupado"] = (df["desocupado"] | df["trab_familiar_aux"]).astype("int8")

    base_year = int(df["ano"].min())
    df["time"] = (df["ano"] - base_year) * 4 + df["trimestre"]
    df = df.sort_values(["id_pessoa", "time"]).reset_index(drop=True)

    df["sector_t"] = build_sector(df)
    df["wage_t"] = df["renda_habitual_principal"].where(df["renda_habitual_principal"] > 0)

    # V4: restrict employed observations to full-time workers (>= HOURS_MIN hours/week).
    # Unemployed workers are always retained regardless of hours.
    # Part-time employed observations have sector_t and wage_t set to NaN so they do
    # not form valid transition endpoints; the consecutive-pair filter then drops them.
    employed_mask = df["sector_t"].isin(["public", "private"])
    not_full_time = employed_mask & ~(df["horas_habituais_principal"] >= HOURS_MIN)
    df.loc[not_full_time, "sector_t"] = np.nan
    df.loc[not_full_time, "wage_t"] = np.nan

    n_total_employed = int(employed_mask.sum())
    n_excluded = int(not_full_time.sum())
    print(f"Employed observations before hours filter: {n_total_employed:,}")
    print(f"Excluded (part-time or missing hours):     {n_excluded:,}  ({100*n_excluded/n_total_employed:.1f}%)")
    print(f"Retained full-time employed observations:  {n_total_employed - n_excluded:,}")

    next_time = df.groupby("id_pessoa")["time"].shift(-1)
    df["next_gap"] = next_time - df["time"]
    df["consecutive_t1"] = df["next_gap"].eq(1)

    next_year = df.groupby("id_pessoa")["ano"].shift(-1)
    next_quarter = df.groupby("id_pessoa")["trimestre"].shift(-1)
    next_sector = df.groupby("id_pessoa")["sector_t"].shift(-1)
    next_wage = df.groupby("id_pessoa")["wage_t"].shift(-1)

    df["ano_t1"] = next_year.where(df["consecutive_t1"])
    df["trimestre_t1"] = next_quarter.where(df["consecutive_t1"])
    df["sector_t1"] = next_sector.where(df["consecutive_t1"])
    df["wage_t1"] = next_wage.where(df["consecutive_t1"])
    df["valid_transition_pair"] = df["consecutive_t1"] & df["sector_t1"].notna()
    df["transition"] = np.select(
        [
            (df["sector_t"] == "private") & (df["sector_t1"] == "public"),
            (df["sector_t"] == "public") & (df["sector_t1"] == "private"),
            (df["sector_t"] == "private") & (df["sector_t1"] == "unemployed"),
            (df["sector_t"] == "public") & (df["sector_t1"] == "unemployed"),
            (df["sector_t"] == "unemployed") & (df["sector_t1"] == "private"),
            (df["sector_t"] == "unemployed") & (df["sector_t1"] == "public"),
            df["sector_t1"].notna(),
        ],
        [
            "private_to_public",
            "public_to_private",
            "private_to_unemployed",
            "public_to_unemployed",
            "unemployed_to_private",
            "unemployed_to_public",
            "other",
        ],
        default=None,
    )

    out_cols = [
        "id_pessoa",
        "ano",
        "trimestre",
        "ano_t1",
        "trimestre_t1",
        "time",
        "next_gap",
        "consecutive_t1",
        "valid_transition_pair",
        "sector_t",
        "sector_t1",
        "wage_t",
        "wage_t1",
        "transition",
    ]
    prepared = df[out_cols].copy()
    prepared.to_parquet(OUTPUT_PATH, index=False)

    print(f"Rows loaded:            {len(df):,}")
    print(f"Consecutive t+1 pairs:  {int(df['consecutive_t1'].sum()):,}")
    print(f"Valid transition pairs: {int(prepared['valid_transition_pair'].sum()):,}")
    print(f"Output written to:      {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
