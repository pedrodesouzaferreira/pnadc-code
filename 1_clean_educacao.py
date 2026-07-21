import pandas as pd
import numpy as np

folder = '/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC'

YEAR = 2023   # ← change this to match the year downloaded by 0_import_educacao.py

# ─────────────────────────────────────────────────────────────────────────────
# 1. BUILD LABEL MAPPINGS
# ─────────────────────────────────────────────────────────────────────────────
# We reuse the microdados dictionary for the variables shared between tables
# (demographics, geography, weights). Educacao-specific variables (V3034,
# V3034A-C, V3002A, etc.) are stored as text labels in basedosdados (STRING
# type, already human-readable — no numeric codes to map).

dic = pd.read_csv(f'{folder}/Raw Data/PNADC_dicionario.csv', dtype=str)
dic = dic[dic['id_tabela'].isin(['microdados', 'educacao'])].copy()
dic = dic[dic['chave'].notna() & (dic['chave'].str.strip() != '')].copy()
dic['col_lower'] = dic['nome_coluna'].str.strip().str.lower()
dic['chave_num'] = pd.to_numeric(dic['chave'].str.strip(), errors='coerce')

label_map = {}
for col, grp in dic.groupby('col_lower'):
    mapping = {int(r['chave_num']): r['valor'].strip()
               for _, r in grp.iterrows() if pd.notna(r['chave_num'])}
    if mapping:
        label_map[col] = mapping

print(f'Label mappings built for {len(label_map)} variables.')

# ─────────────────────────────────────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(f'{folder}/Raw Data/PNADC_educacao_{YEAR}.csv', low_memory=False)
print(f'Loaded data: {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'Columns: {df.columns.tolist()}')

col_lookup = {c.lower(): c for c in df.columns}

# ─────────────────────────────────────────────────────────────────────────────
# 3. APPLY LABEL COLUMNS (shared variables with microdados dictionary)
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
#    Column names follow the same pattern as microdados:
#    UPPERCASE for IBGE variables, lowercase for basedosdados-derived fields.
# ─────────────────────────────────────────────────────────────────────────────

rename_map = {
    # ── Panel / identifiers ──────────────────────────────────────────────────
    'V2003'   : 'num_ordem',
    'V1016'   : 'num_entrevista',
    'V1028'   : 'peso',
    'posest'  : 'estrato_pesquisa',

    # ── Geography ────────────────────────────────────────────────────────────
    'V1022'   : 'zona',
    'V1023'   : 'tipo_municipio',

    # ── Demographics ─────────────────────────────────────────────────────────
    'V2001'   : 'num_moradores',
    'V3001'   : 'sabe_ler_escrever',

    # ── School attendance (current) ──────────────────────────────────────────
    'V3002'   : 'frequenta_escola',
    'V3002A'  : 'rede_escola',               # public / private
    'V3003A'  : 'curso_atual',               # type of course being taken
    'V3004'   : 'duracao_curso_atual',
    'V3004A'  : 'graduacao_tecnologica',
    'V3005A'  : 'organizacao_curso_atual',   # semester / year / module
    'V3006'   : 'ano_serie_atual',
    'V3006A'  : 'etapa_fund_atual',
    'V3006B'  : 'modalidade_curso_atual',    # presencial / EAD / híbrido
    'V3006C'  : 'turno_curso',               # morning / evening / night
    'V3007'   : 'ja_concluiu_graduacao',

    # ── Previous schooling ───────────────────────────────────────────────────
    'V3008'   : 'frequentou_escola_antes',
    'V3009A'  : 'curso_mais_elevado_anterior',
    'V3010'   : 'duracao_curso_anterior',
    'V3010A'  : 'graduacao_tec_anterior',
    'V3012'   : 'concluiu_1a_serie_anterior',
    'V3013'   : 'ultimo_ano_concluido_anterior',
    'V3014'   : 'concluiu_curso_anterior',
    'V3017'   : 'modalidade_curso_anterior',

    # ── Technical / vocational courses ──────────────────────────────────────
    'V3019A'  : 'frequenta_curso_tecnico',
    'V3020B'  : 'tipo_curso_tecnico',        # integrated / concomitant / subsequent
    'V3020C'  : 'modalidade_curso_tecnico',  # presencial / EAD
    'V3021A'  : 'frequentou_curso_tecnico_antes',
    'V3023A'  : 'concluiu_curso_tecnico_anterior',
    'V3024'   : 'frequenta_pre_vestibular',
    'V3025'   : 'frequenta_extensao_superior',
    'V3026'   : 'frequenta_qualificacao_prof',
    'V3026A'  : 'tipo_qualificacao_prof',
    'V3028'   : 'frequentou_extensao_antes',
    'V3029'   : 'frequentou_qualificacao_antes',
    'V3032'   : 'concluiu_ultima_qualificacao',

    # ── KEY: reason for NOT attending school ─────────────────────────────────
    'V3033'   : 'motivo_nao_frequenta_antes',   # older version of the question
    'V3033A'  : 'motivo_nao_frequenta_v2',
    'V3033B'  : 'com_quem_fica_crianca',

    # ★ The main variable you care about:
    'V3034'   : 'motivo_nao_frequenta_escola',      # older version (pre-2022ish)
    'V3034A'  : 'idade_parou_escola',
    'V3034B'  : 'motivo_deixou_escola',             # why they left school
    'V3034C'  : 'motivo_nao_frequenta_escola_atual',# ★ CURRENT reason (includes concurso)
}

# Build case-insensitive rename using col_lookup (raw data may be lowercase)
actual_rename = {}
for orig, readable in rename_map.items():
    actual = col_lookup.get(orig.lower())   # e.g. 'v3034c' → actual column name
    if actual:
        actual_rename[actual] = readable

label_rename = {
    actual + '_label': readable + '_label'
    for actual, readable in actual_rename.items()
    if actual + '_label' in df.columns
}

df.rename(columns={**actual_rename, **label_rename}, errors='ignore', inplace=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5. DERIVED VARIABLES
# ─────────────────────────────────────────────────────────────────────────────

# ── Concurseiro flag: not attending school, reason = studying for concurso ────
# V3034C code 8 = "Estudando para concurso ou por conta própria para vestibular/ENEM"
concurso_col = 'motivo_nao_frequenta_escola_atual'   # V3034C
if concurso_col in df.columns:
    num_v3034c = pd.to_numeric(df[concurso_col], errors='coerce')
    flag_conc = (num_v3034c == 8)
    label_col = concurso_col + '_label'
    if label_col in df.columns:
        flag_conc = flag_conc | df[label_col].astype(str).str.contains('concurso', case=False, na=False)
    df['estudando_concurso'] = flag_conc.astype('Int8')
    print(f'\nV3034C unique values:\n{df[concurso_col].value_counts(dropna=False).head(20)}')

# Also check older version (V3034)
old_col = 'motivo_nao_frequenta_escola'
if old_col in df.columns:
    num_old = pd.to_numeric(df[old_col], errors='coerce')
    flag_old = (num_old == 8)
    old_label = old_col + '_label'
    if old_label in df.columns:
        flag_old = flag_old | df[old_label].astype(str).str.contains('concurso', case=False, na=False)
    if 'estudando_concurso' in df.columns:
        df['estudando_concurso'] = (df['estudando_concurso'].astype(bool) | flag_old).astype('Int8')
    else:
        df['estudando_concurso'] = flag_old.astype('Int8')

# ── Pre-vestibular (vestibular prep) ─────────────────────────────────────────
if 'frequenta_pre_vestibular' in df.columns:
    df['esta_no_pre_vestibular'] = (
        df['frequenta_pre_vestibular'].astype(str).str.contains('Sim|^1$', case=False, na=False)
    ).astype('Int8')

# ── Currently attending school ────────────────────────────────────────────────
if 'frequenta_escola' in df.columns:
    df['na_escola'] = (
        df['frequenta_escola'].astype(str).str.contains('Sim|^1$', case=False, na=False)
    ).astype('Int8')

# ─────────────────────────────────────────────────────────────────────────────
# 6. SAVE
# ─────────────────────────────────────────────────────────────────────────────

output = f'{folder}/Cleaned Data/PNADC_educacao_limpo_{YEAR}.csv'
df.to_csv(output, index=False)
print(f'\nSaved → {output}')
print(f'Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns')
if 'estudando_concurso' in df.columns:
    n = df['estudando_concurso'].sum()
    print(f'Concurseiros (V3034C) in sample: {n:,}')
