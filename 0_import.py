""""
pip install click==8.0.3
pip install pandas==1.3.5
pip install pyarrow==6.0.0
pip install shapely==1.8.5
NB: basedosdados clashes with geopandas -- so keep one or the other
"""


import pandas as pd
import basedosdados as bd

# Install basedosdados: pip install basedosdados
folder = '/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC'
print('Folder: ', folder)

YEAR = 2022   # ← change this to download a different year

df = bd.read_sql(
        query=f"""
        SELECT *
        FROM `basedosdados.br_ibge_pnadc.microdados`
        WHERE ano = {YEAR}
        """,
        billing_project_id='educacaoideias'
    )
df = pd.DataFrame(df)
df.to_csv(f'{folder}/Raw Data/PNADC_microdados_{YEAR}.csv', index=False)
