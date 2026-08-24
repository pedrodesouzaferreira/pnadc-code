import pandas as pd
import basedosdados as bd

folder = '/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC'
print('Folder: ', folder)

# ---------------------------------------------------------------------------
# Sampling strategy
# ---------------------------------------------------------------------------
# PNADC is a rotating panel: the same individual (id_domicilio + v2003) can
# appear in up to 5 quarters. We want a 10% sample of INDIVIDUALS, keeping
# all available quarterly observations for each sampled person.
#
# We use FARM_FINGERPRINT (BigQuery's fast hash function) on the individual
# identifier and keep those where hash mod 10 == 0. This gives ~10% of
# individuals deterministically (same result on every run).
# ---------------------------------------------------------------------------

YEAR         = 2023   # ← change this to download a different year
SAMPLE_SHARE = 0      # change to e.g. 1, 2... to get a different 10% slice

query = f"""
SELECT *
FROM `basedosdados.br_ibge_pnadc.microdados`
WHERE ano = {YEAR}
  AND MOD(
        ABS(FARM_FINGERPRINT(
          CONCAT(
            CAST(id_domicilio AS STRING), '_',
            CAST(v2003        AS STRING)
          )
        )),
        10
      ) = {SAMPLE_SHARE}
"""

df = bd.read_sql(
    query=query,
    billing_project_id='educacaoideias'
)

df = pd.DataFrame(df)

# Save first — before any diagnostic code that might raise an error
df.to_csv(f'{folder}/Raw Data/PNADC_microdados_sample_{YEAR}.csv', index=False)
print('Saved.')
print(f'Observations : {len(df):,}')
print(f'Columns      : {df.columns.tolist()}')

# Diagnostic: count unique individuals.
# basedosdados may return column names in a different case than the SQL source.
person_col = next((c for c in df.columns if c.lower() == 'v2003'), None)
hh_col     = next((c for c in df.columns if c.lower() == 'id_domicilio'), None)

if person_col and hh_col:
    n_individuals = df.groupby([hh_col, person_col]).ngroups
    print(f'Individuals  : {n_individuals:,}')
    print(f'Avg quarters per individual: {len(df) / n_individuals:.2f}')
else:
    print(f'Could not find person ID columns. Available: {df.columns.tolist()}')
