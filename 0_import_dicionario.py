import pandas as pd
import basedosdados as bd

folder = '/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC'
print('Folder: ', folder)

# ---------------------------------------------------------------------------
# The basedosdados convention is to store code → label mappings in a
# `dicionario` table within the same dataset. It has the structure:
#   id_tabela      : which table the variable belongs to
#   nome_coluna    : variable name (e.g. 'v2007', 'vd4002')
#   chave          : the numeric code stored in the data
#   cobertura_temporal : years the code is valid
#   valor          : the human-readable label for that code
# ---------------------------------------------------------------------------

df_dic = bd.read_sql(
    query="SELECT * FROM `basedosdados.br_ibge_pnadc.dicionario`",
    billing_project_id='educacaoideias'
)

df_dic = pd.DataFrame(df_dic)
print(f'Dictionary rows : {len(df_dic):,}')
print(f'Variables covered: {df_dic["nome_coluna"].nunique()}')
print(df_dic.head(10).to_string())

df_dic.to_csv(f'{folder}/Raw Data/PNADC_dicionario.csv', index=False)
print('Saved.')
