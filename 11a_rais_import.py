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
folder = '/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/RAIS_Workers'
print('Folder: ', folder)


# Para carregar o dado direto no pandas
#df = bd.read_table(dataset_id='br_me_rais',
#table_id='dicionario',
#billing_project_id='educacaoideias')
#df = pd.DataFrame(df)

# Salvando csv
#df.to_csv(f'{folder}/Raw Data/dicionario.csv', index=False)

# Para carregar o dado direto no pandas
#df = bd.read_sql(
#    query="SELECT * FROM `basedosdados.br_me_rais.microdados_vinculos` WHERE ano = 2022",
#    billing_project_id='educacaoideias'
#)

# Salvando csv
#df.to_csv(f'{folder}/Raw Data/microdados_vinculos_2022.csv', index=False)
#df.to_pickle(f'{folder}/Raw Data/microdados_vinculos_2022.pkl')



# Importando estado por estado (sigla_uf)
for uf in ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']:
#for uf in ['SP', 'TO']:
    print(f'UF: {uf}')
    df = bd.read_sql(
        query=f"SELECT * FROM `basedosdados.br_me_rais.microdados_vinculos` WHERE ano = 2024 AND sigla_uf = '{uf}'",
        billing_project_id='educacaoideias'
    )
    df = pd.DataFrame(df)
    df.to_csv(f'{folder}/Raw Data/microdados_vinculos_2024_{uf}.csv', index=False)
    #df.to_pickle(f'{folder}/Raw Data/microdados_vinculos_2022_{uf}.pkl')
    print(f'UF: {uf} - Done!')

"""
# Importando todo mundo mas só 2022
#for uf in ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'TO', 'SP']:
for uf in [ 'TO']:
    print(f'UF: {uf}')
    df = bd.read_sql(
        query=f"SELECT * FROM `basedosdados.br_me_rais.microdados_vinculos` WHERE ano = 2022 AND sigla_uf = '{uf}'",
        billing_project_id='educacaoideias'
    )
    df = pd.DataFrame(df)
    df.to_csv(f'{folder}/Raw Data/microdados_vinculos_2022_{uf}.csv', index=False)
    #df.to_pickle(f'{folder}/Raw Data/microdados_vinculos_2022_{uf}.pkl')
    print(f'UF: {uf} - Done!')
"""

""""
# Importando 2022 com random sampling de 1% e pondo uma seed para replicabilidade
df = bd.read_sql(
    query="SELECT * FROM (SELECT *, ROW_NUMBER() OVER () AS row_num FROM `basedosdados.br_me_rais.microdados_vinculos` WHERE ano = 2022) WHERE ABS(MOD(FARM_FINGERPRINT(CONCAT(CAST(row_num AS STRING), \"42\")), 10000)) < 100;",
    billing_project_id='educacaoideias'
)
df = pd.DataFrame(df)
df.to_csv(f'{folder}/Raw Data/microdados_vinculos_2022_sample_1perc.csv', index=False)
"""

""""
# Importando 2022 com random sampling de 10% e pondo uma seed para replicabilidade
df = bd.read_sql(
    query="SELECT * FROM (SELECT *, ROW_NUMBER() OVER () AS row_num FROM `basedosdados.br_me_rais.microdados_vinculos` WHERE ano = 2022) WHERE ABS(MOD(FARM_FINGERPRINT(CONCAT(CAST(row_num AS STRING), \"42\")), 10000)) < 1000;",
    billing_project_id='educacaoideias'
)
df = pd.DataFrame(df)
df.to_csv(f'{folder}/Raw Data/microdados_vinculos_2022_sample_10perc.csv', index=False)
"""

"""
# Importando 2022 com random sampling de 20% e pondo uma seed para replicabilidade
df = bd.read_sql(
    query="SELECT * FROM (SELECT *, ROW_NUMBER() OVER () AS row_num FROM `basedosdados.br_me_rais.microdados_vinculos` WHERE ano = 2022) WHERE ABS(MOD(FARM_FINGERPRINT(CONCAT(CAST(row_num AS STRING), \"42\")), 10000)) < 2000;",
    billing_project_id='educacaoideias'
)
df = pd.DataFrame(df)
df.to_csv(f'{folder}/Raw Data/microdados_vinculos_2022_sample_20perc.csv', index=False)
"""
