import pandas as pd
import numpy as np

folder = '/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC'

YEAR = 2023   # ← change this to match the year downloaded by 0_import_sample.py

# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD LABEL MAPPINGS FROM THE DOWNLOADED DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────
# Dictionary: id_tabela | nome_coluna | chave | cobertura_temporal | valor
# We keep only rows with a non-empty numeric chave (discrete code variables).
# basedosdados returns column names in UPPERCASE for IBGE variables (V2007 etc.)
# and lowercase for derived/geographic fields (id_domicilio, sigla_uf, etc.).
# We build a case-insensitive lookup to handle both.

dic = pd.read_csv(f'{folder}/Raw Data/PNADC_dicionario.csv', dtype=str)
dic = dic[dic['id_tabela'] == 'microdados'].copy()
dic = dic[dic['chave'].notna() & (dic['chave'].str.strip() != '')].copy()
dic['col_lower']  = dic['nome_coluna'].str.strip().str.lower()
dic['chave_num']  = pd.to_numeric(dic['chave'].str.strip(), errors='coerce')

label_map = {}   # { lowercase_colname : { int_code : label } }
for col, grp in dic.groupby('col_lower'):
    mapping = {int(r['chave_num']): r['valor'].strip()
               for _, r in grp.iterrows() if pd.notna(r['chave_num'])}
    if mapping:
        label_map[col] = mapping

print(f'Label mappings built for {len(label_map)} variables.')

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(f'{folder}/Raw Data/PNADC_microdados_{YEAR}.csv', low_memory=False)
print(f'Loaded data: {df.shape[0]:,} rows × {df.shape[1]} columns')

# Case-insensitive column lookup:  lowercase_name → actual column name in df
col_lookup = {c.lower(): c for c in df.columns}

# ─────────────────────────────────────────────────────────────────────────────
# 3. APPLY LABEL COLUMNS (add *_label columns from dictionary)
#    Uses case-insensitive matching so it works whether data is upper or lower.
# ─────────────────────────────────────────────────────────────────────────────

for col_lower, mapping in label_map.items():
    if col_lower in col_lookup:
        actual = col_lookup[col_lower]
        df[actual + '_label'] = (
            pd.to_numeric(df[actual], errors='coerce')
              .map(mapping)
              .astype('category')
        )

# ─────────────────────────────────────────────────────────────────────────────
# 4. RENAME COLUMNS TO READABLE NAMES
#    Keys here are the EXACT column names returned by basedosdados:
#      - UPPERCASE for original IBGE variables (V2007, VD4002 …)
#      - lowercase for basedosdados-derived fields (id_domicilio, sigla_uf …)
#    Corrections vs. previous version:
#      - VD4031 = hours worked (NOT income) → horas_habituais_todos
#      - VD4016/VD4017/VD4019/VD4020 = income (FLOAT64)
#      - habitual/efetivo = derived total income variables
# ─────────────────────────────────────────────────────────────────────────────

rename_map = {
    # ── Panel / identifiers ──────────────────────────────────────────────────
    'V2003'      : 'num_ordem',
    'V1016'      : 'num_entrevista',        # 1–5, tells you which panel visit
    'V1028'      : 'peso',
    'posest'     : 'estrato_pesquisa',

    # ── Geography ────────────────────────────────────────────────────────────
    'V1022'      : 'zona',                  # urban / rural
    'V1023'      : 'tipo_municipio',        # capital / RM+RIDE / interior

    # ── Demographics ─────────────────────────────────────────────────────────
    'V2005'      : 'condicao_domicilio',
    'V2007'      : 'sexo',
    'V2008'      : 'dia_nasc',
    'V20081'     : 'mes_nasc',
    'V20082'     : 'ano_nasc',
    'V2009'      : 'idade',
    'V2010'      : 'raca_cor',

    # ── Education ────────────────────────────────────────────────────────────
    'V3001'      : 'sabe_ler_escrever',
    'V3002'      : 'frequenta_escola',
    'V3002A'     : 'rede_escola',
    'VD3004'     : 'nivel_instrucao',
    'VD3005'     : 'anos_estudo',           # continuous years (INT64)
    'VD3006'     : 'anos_estudo_faixa',

    # ── Household ────────────────────────────────────────────────────────────
    'V2001'      : 'num_moradores',
    'VD2002'     : 'relacao_domicilio',
    'VD2003'     : 'num_componentes_familia',
    'VD2004'     : 'tipo_familia',

    # ── Labor force status ───────────────────────────────────────────────────
    'VD4001'     : 'condicao_forca_trabalho',
    'VD4002'     : 'condicao_ocupacao',
    'VD4003'     : 'forca_trabalho_potencial',
    'VD4004'     : 'subocupado_horas_efetivas',
    'VD4004A'    : 'subocupado_horas_habituais',
    'VD4005'     : 'desalentado',

    # ── Employment type / position ───────────────────────────────────────────
    'VD4007'     : 'posicao_empr_geral',
    'VD4008'     : 'categoria_emprego',
    'VD4009'     : 'posicao_emprego',       # detailed (with/without carteira etc.)
    'VD4010'     : 'setor_atividade',       # economic sector
    'VD4011'     : 'grupo_ocupacao',        # occupation group (ISCO-08 adapted)
    'VD4012'     : 'contribui_previdencia',

    # ── Main job characteristics ─────────────────────────────────────────────
    'V4012'      : 'posicao_trab_principal',
    'V4014'      : 'area_trab_principal',   # urban / rural work location
    'V4028'      : 'servidor_publico_estatutario',   # ★ key public servant flag
    'V4029'      : 'carteira_assinada',
    'V4032'      : 'contribui_prev_trab_principal',
    'V4039'      : 'horas_habituais_principal',      # usual weekly hours
    'V4039C'     : 'horas_efetivas_principal',       # actual hours in ref. week
    'V4040'      : 'tempo_nesse_trabalho',

    # ── Secondary job ────────────────────────────────────────────────────────
    'V4047'      : 'servidor_publico_secundario',    # ★ public servant (sec. job)
    'V4048'      : 'carteira_assinada_secundario',
    'V4056'      : 'horas_habituais_secundario',
    'V4056C'     : 'horas_efetivas_secundario',

    # ── Hours worked (derived) ────────────────────────────────────────────────
    # NOTE: VD4031–VD4035 are HOURS, not income
    'VD4013'     : 'horas_habituais_todos_faixa',
    'VD4014'     : 'horas_efetivas_todos_faixa',
    'VD4031'     : 'horas_habituais_todos',          # continuous hours, all jobs
    'VD4032'     : 'horas_efetivas_principal_h',     # effective hours, main job
    'VD4033'     : 'horas_efetivas_secundario_h',
    'VD4035'     : 'horas_efetivas_todos',
    'VD4036'     : 'horas_habituais_principal_faixa',
    'VD4037'     : 'horas_efetivas_principal_faixa',

    # ── Income (continuous FLOAT64) ──────────────────────────────────────────
    'VD4015'     : 'tipo_renda_principal',
    'VD4016'     : 'renda_habitual_principal',   # ★ habitual income, main job
    'VD4017'     : 'renda_efetiva_principal',    # effective income, main job
    'VD4018'     : 'tipo_renda_todos',
    'VD4019'     : 'renda_habitual_todos',       # ★ habitual income, all jobs
    'VD4020'     : 'renda_efetiva_todos',        # effective income, all jobs
    'habitual'   : 'renda_habitual_total',       # basedosdados derived total
    'efetivo'    : 'renda_efetiva_total',

    # ── Income bracket variables (explicit amounts in V403312 etc.) ───────────
    'V403312'    : 'valor_dinheiro_principal',   # cash amount, main job
    'V403412'    : 'valor_dinheiro_efetivo_principal',

    # ── Job search (unemployed) ───────────────────────────────────────────────
    'V4071'      : 'tomou_providencia_busca',    # took any action to find work
    'V4072'      : 'metodo_busca_emprego_v1',    # method (older, pre-2021 wave)
    'V4072A'     : 'metodo_busca_emprego',       # ★ method (current: concurso etc.)
    'V4073'      : 'gostaria_ter_trabalhado',
    'V4074'      : 'motivo_nao_buscou_v1',
    'V4074A'     : 'motivo_nao_buscou',
    'V4076'      : 'tempo_desempregado',

    # ── Out of labor force ────────────────────────────────────────────────────
    'V4077'      : 'poderia_comecar_trabalhar',
    'V4078'      : 'motivo_nao_trabalhou_v1',
    'V4078A'     : 'motivo_nao_trabalhou',
    'VD4023'     : 'motivo_fora_forca_trabalho',
    'VD4030'     : 'motivo_fora_forca_trabalho2',
}

# Rename _label columns to match their readable prefix
label_rename = {
    orig + '_label': readable + '_label'
    for orig, readable in rename_map.items()
    if orig + '_label' in df.columns
}

df.rename(columns={**rename_map, **label_rename}, errors='ignore', inplace=True)

# Refresh col_lookup after rename
col_lookup = {c.lower(): c for c in df.columns}

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIX TYPES & MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────

# Income variables (no coded missing in FLOAT64 — NaN already means N/A)
income_cols = [
    'renda_habitual_principal', 'renda_efetiva_principal',
    'renda_habitual_todos',     'renda_efetiva_todos',
    'renda_habitual_total',     'renda_efetiva_total',
    'valor_dinheiro_principal', 'valor_dinheiro_efetivo_principal',
]
for col in income_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Hours (INT64 in BigQuery — should already be numeric, but coerce to be safe)
hours_cols = [
    'horas_habituais_todos', 'horas_efetivas_todos',
    'horas_habituais_principal', 'horas_efetivas_principal',
    'horas_efetivas_principal_h', 'horas_efetivas_todos',
]
for col in hours_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

for col in ['idade', 'peso', 'anos_estudo', 'num_entrevista']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ─────────────────────────────────────────────────────────────────────────────
# 6. DERIVED VARIABLES
# ─────────────────────────────────────────────────────────────────────────────

# ── Labor market status ──────────────────────────────────────────────────────
# condicao_ocupacao: STRING from basedosdados — check actual values at runtime
# The variable may be stored as "1"/"2" (codes) or as labels depending on the
# basedosdados version. We check both.
def parse_flag(series, code_val, label_substring):
    """Returns 1 where value == code_val OR contains label_substring."""
    numeric = pd.to_numeric(series, errors='coerce')
    by_code = (numeric == code_val)
    by_label = series.astype(str).str.contains(label_substring, case=False, na=False)
    return (by_code | by_label).astype('Int8')

if 'condicao_ocupacao' in df.columns:
    df['ocupado']    = parse_flag(df['condicao_ocupacao'], 1, 'Ocupado')
    df['desocupado'] = parse_flag(df['condicao_ocupacao'], 2, 'Desocupado')

if 'condicao_forca_trabalho' in df.columns:
    df['na_pea'] = parse_flag(df['condicao_forca_trabalho'], 1, 'Força de trabalho')

# ── Public servant flags ─────────────────────────────────────────────────────
# V4028 code 1 = Sim (estatutário). Use parse_flag so numeric 1.0 is handled.
if 'servidor_publico_estatutario' in df.columns:
    df['servidor_publico'] = parse_flag(df['servidor_publico_estatutario'], 1, 'Sim')

# ── Concurso seeker (job-search method = concurso) ────────────────────────────
# V4072A code 5 = "Fez ou inscreveu-se em concurso". Data is stored as numeric
# codes, so check code 5 (and fall back to text match on the label column).
if 'metodo_busca_emprego' in df.columns:
    num_busca = pd.to_numeric(df['metodo_busca_emprego'], errors='coerce')
    flag_busca = (num_busca == 5)
    label_col = 'metodo_busca_emprego_label'
    if label_col in df.columns:
        flag_busca = flag_busca | df[label_col].astype(str).str.contains('concurso', case=False, na=False)
    df['buscando_via_concurso'] = flag_busca.astype('Int8')

# Also check older variable (V4072 — codes may differ; use text match on label)
if 'metodo_busca_emprego_v1' in df.columns:
    label_v1 = 'metodo_busca_emprego_v1_label'
    if label_v1 in df.columns:
        flag2 = df[label_v1].astype(str).str.contains('concurso', case=False, na=False)
    else:
        num_v1 = pd.to_numeric(df['metodo_busca_emprego_v1'], errors='coerce')
        flag2 = num_v1.isin([5])   # assume same code; text fallback above preferred
    if 'buscando_via_concurso' in df.columns:
        df['buscando_via_concurso'] = (df['buscando_via_concurso'].astype(bool) | flag2).astype('Int8')
    else:
        df['buscando_via_concurso'] = flag2.astype('Int8')

# ── Formality ────────────────────────────────────────────────────────────────
# posicao_emprego (VD4009) codes:
#   1=priv c/cart  2=priv s/cart  3=pub c/cart  4=pub s/cart  5=militar/estat
#   6=empregador  7=conta-própria  8=dom c/cart  9=dom s/cart  10=fam. aux.
if 'posicao_emprego' in df.columns:
    pos = pd.to_numeric(df['posicao_emprego'], errors='coerce')
    df['formal']               = pos.isin([1, 3, 5, 8]).astype('Int8')
    df['informal']             = pos.isin([2, 4, 6, 7, 9, 10]).astype('Int8')
    df['empregado_setor_priv'] = pos.isin([1, 2]).astype('Int8')
    df['empregado_setor_pub']  = pos.isin([3, 4, 5]).astype('Int8')
    df['conta_propria']        = (pos == 7).astype('Int8')
    df['empregador']           = (pos == 6).astype('Int8')
    df['trab_domestico']       = pos.isin([8, 9]).astype('Int8')

# ── Age groups ───────────────────────────────────────────────────────────────
if 'idade' in df.columns:
    bins   = [0, 14, 24, 34, 44, 54, 64, np.inf]
    labels = ['0–14', '15–24', '25–34', '35–44', '45–54', '55–64', '65+']
    df['faixa_etaria']   = pd.cut(df['idade'], bins=bins, labels=labels, right=False)
    df['idade_trabalho'] = (df['idade'] >= 14).astype('Int8')

# ── Wages ────────────────────────────────────────────────────────────────────
income_col = next(
    (c for c in ['renda_habitual_principal', 'renda_habitual_todos',
                 'renda_habitual_total']
     if c in df.columns and df[c].notna().any()),
    None
)
if income_col:
    MW_2024 = 1_412
    df['log_renda'] = np.log(df[income_col].where(df[income_col] > 0))
    bins_w   = [0, MW_2024 * 0.5, MW_2024, MW_2024 * 2, MW_2024 * 3,
                MW_2024 * 5, np.inf]
    labels_w = ['< 0,5 SM', '0,5–1 SM', '1–2 SM', '2–3 SM', '3–5 SM', '> 5 SM']
    df['faixa_salarial'] = pd.cut(df[income_col], bins=bins_w,
                                   labels=labels_w, right=False)

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────────────────────────────────────

output = f'{folder}/Cleaned Data/PNADC_limpo_{YEAR}.csv'
df.to_csv(output, index=False)
print(f'Saved → {output}')
print(f'Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns')
print('\nColumn list:')
print(df.columns.tolist())
