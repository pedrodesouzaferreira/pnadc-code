import pandas as pd
import basedosdados as bd

folder = '/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC'
print('Folder:', folder)

# ---------------------------------------------------------------------------
# Sampling notes
# ---------------------------------------------------------------------------
# The educacao table shares id_domicilio and V2003 with microdados.
# We use the EXACT same FARM_FINGERPRINT hash on (id_domicilio, V2003) as
# in 0_import_sample.py so the 10% sample is the same set of individuals
# across both datasets — enabling clean merges.
#
# Merge key within a quarter: id_pessoa (= id_domicilio + V2003, per-quarter)
# Longitudinal key:           id_domicilio + V2003 (across quarters)
# ---------------------------------------------------------------------------

YEAR         = 2023   # ← change this to download a different year (must match 0_import_sample.py)
SAMPLE_SHARE = 0      # must match 0_import_sample.py

# ── 10% individual sample (same persons as microdados sample) ─────────────────
print('Downloading 10% educacao sample...')
df_sample = bd.read_sql(
    query=f"""
    SELECT *
    FROM `basedosdados.br_ibge_pnadc.educacao`
    WHERE ano = {YEAR}
      AND MOD(
            ABS(FARM_FINGERPRINT(
              CONCAT(
                CAST(id_domicilio AS STRING), '_',
                CAST(V2003        AS STRING)
              )
            )),
            10
          ) = {SAMPLE_SHARE}
    """,
    billing_project_id='educacaoideias'
)
df_sample = pd.DataFrame(df_sample)
df_sample.to_csv(f'{folder}/Raw Data/PNADC_educacao_sample_{YEAR}.csv', index=False)
print(f'Saved sample. Shape: {df_sample.shape}')

# ── Full year download ────────────────────────────────────────────────────────
print(f'Downloading full educacao {YEAR}...')
df_full = bd.read_sql(
    query=f"""
    SELECT *
    FROM `basedosdados.br_ibge_pnadc.educacao`
    WHERE ano = {YEAR}
    """,
    billing_project_id='educacaoideias'
)
df_full = pd.DataFrame(df_full)
df_full.to_csv(f'{folder}/Raw Data/PNADC_educacao_{YEAR}.csv', index=False)
print(f'Saved full. Shape: {df_full.shape}')
print(f'Columns: {df_full.columns.tolist()}')
del df_full   # free memory before downloading sample
