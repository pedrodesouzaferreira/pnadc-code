"""
4_analyze.py  –  Concursos públicos & public employment analysis
================================================================
Outputs:
  Output/Tables/  → LaTeX tables (.tex)
  Output/Figures/ → PNG figures

Run order: 0_import_sample.py → 0_import_educacao.py →
           1_clean.py → 1_clean_educacao.py → 4_analyze.py
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')                        # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent

YEAR    = 2023   # ← change this to analyse a different year

T_DIR   = f'{ROOT}/Output/Tables'
F_DIR   = f'{ROOT}/Output/Figures'
os.makedirs(T_DIR, exist_ok=True)
os.makedirs(F_DIR, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def save_table(df, name, index=True):
    path = f'{T_DIR}/{name}.tex'
    latex_str = df.to_latex(
        index=index,
        float_format=lambda x: f'{x:,.1f}',
        na_rep='--',
        escape=True,
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    print(f'  [TABLE] {name}.tex  ({df.shape[0]} rows)')

def save_fig(name, dpi=150):
    path = f'{F_DIR}/{name}.png'
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'  [FIG]   {name}.png')

MW_2024 = 1_412   # minimum wage 2024

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
BLUE, RED, GREEN, GREY = '#2B7BB9', '#E05C5C', '#3AAA5C', '#888888'

pd.set_option('display.float_format', '{:,.1f}'.format)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & MERGE
# ─────────────────────────────────────────────────────────────────────────────
print('Loading data...')
mdf = pd.read_csv(f'{ROOT}/Cleaned Data/PNADC_limpo_{YEAR}.csv',           low_memory=False)
edf = pd.read_csv(f'{ROOT}/Cleaned Data/PNADC_educacao_limpo_{YEAR}.csv',  low_memory=False)

for df in [mdf, edf]:
    df['num_ordem'] = pd.to_numeric(df.get('num_ordem', df.get('V2003', np.nan)), errors='coerce')
    df['trimestre'] = pd.to_numeric(df['trimestre'], errors='coerce')
    df['ano']       = pd.to_numeric(df['ano'],       errors='coerce')

edu_keep = ['id_domicilio', 'num_ordem', 'ano', 'trimestre',
            'motivo_nao_frequenta_escola_atual',
            'motivo_nao_frequenta_escola',
            'motivo_deixou_escola',
            'frequenta_escola', 'na_escola',
            'estudando_concurso',
            'frequenta_pre_vestibular', 'esta_no_pre_vestibular',
            'frequenta_qualificacao_prof',
            'curso_atual', 'modalidade_curso_atual']
edu_keep = [c for c in edu_keep if c in edf.columns]

merged = mdf.merge(edf[edu_keep],
                   on=['id_domicilio', 'num_ordem', 'ano', 'trimestre'],
                   how='left', suffixes=('', '_edu'))

print(f'Merged: {merged.shape[0]:,} rows × {merged.shape[1]} cols')

# ── Build panel lags ──────────────────────────────────────────────────────────
panel_id = ['id_domicilio', 'num_ordem']
merged = merged.sort_values(panel_id + ['ano', 'trimestre']).reset_index(drop=True)

lag_cols = ['posicao_emprego', 'setor_atividade', 'condicao_ocupacao',
            'servidor_publico', 'buscando_via_concurso', 'estudando_concurso',
            'metodo_busca_emprego', 'renda_habitual_principal',
            'renda_habitual_todos', 'ocupado', 'desocupado',
            'informal', 'formal', 'empregado_setor_pub', 'conta_propria']

for col in lag_cols:
    if col in merged.columns:
        merged[f'{col}_t1'] = merged.groupby(panel_id)[col].shift(1)
        merged[f'{col}_t2'] = merged.groupby(panel_id)[col].shift(2)
        merged[f'{col}_f1'] = merged.groupby(panel_id)[col].shift(-1)

# ── Build label maps for decoded display ─────────────────────────────────────
# For every raw+label pair in merged, build {numeric_code → text_label}
# Also create lagged versions of key label columns for trajectory tables
label_maps = {}
for col in list(merged.columns):
    if col.endswith('_label'):
        base = col[:-6]
        if base in merged.columns:
            pairs = (merged[[base, col]]
                     .dropna()
                     .drop_duplicates())
            m = {}
            for _, row in pairs.iterrows():
                try:
                    m[float(row[base])] = str(row[col])
                except (ValueError, TypeError):
                    pass
            if m:
                label_maps[base] = m

# Create label lag columns for key categorical variables
label_lag_bases = ['posicao_emprego', 'setor_atividade', 'condicao_ocupacao']
for base in label_lag_bases:
    lbl = base + '_label'
    if lbl in merged.columns:
        for sfx in ('_t1', '_f1'):
            merged[lbl + sfx] = merged.groupby(panel_id)[lbl].shift(
                1 if sfx == '_t1' else -1)

def decode(series, col_name):
    """Map numeric codes in a series to text labels, using label_maps."""
    base = col_name
    for sfx in ('_t1', '_t2', '_f1'):
        if col_name.endswith(sfx):
            base = col_name[:-len(sfx)]
            break
    m = label_maps.get(base, {})
    if not m:
        return series
    numeric = pd.to_numeric(series, errors='coerce')
    mapped  = numeric.map(m)
    return mapped.where(mapped.notna(), series.astype(str).replace('nan', np.nan))

def best_col(df, col):
    """Return the label column name if it exists in df, else the original."""
    lbl = col + '_label'
    return lbl if lbl in df.columns else col

# ──────────────────────────────────────────────────────────────────────────────
# A.  WHO IS STUDYING FOR CONCURSOS?
# ──────────────────────────────────────────────────────────────────────────────
print('\n── Section A: Concurseiros ──────────────────────────────────')

for label, mask_col, tag in [
        ('V3034C – Estudando p/ concurso', 'estudando_concurso', 'def1'),
        ('V4072A – Buscando emprego via concurso', 'buscando_via_concurso', 'def2'),
]:
    if mask_col not in merged.columns:
        continue
    grp = merged[merged[mask_col] == 1].copy()
    print(f'\n{label}: {len(grp):,} person-quarter obs')

    # --- V3034C full distribution (only for def1) ---
    if tag == 'def1' and 'motivo_nao_frequenta_escola_atual' in merged.columns:
        raw_col = 'motivo_nao_frequenta_escola_atual'
        dist_raw = merged[raw_col].value_counts(dropna=False)
        dist = (dist_raw
                .reset_index()
                .rename(columns={'index': 'motivo', raw_col: 'n'}))
        dist.columns = ['codigo', 'n']
        dist['motivo'] = decode(dist['codigo'], raw_col)
        dist['pct'] = (dist['n'] / dist['n'].sum() * 100).round(1)
        save_table(dist[['motivo', 'n', 'pct']].fillna('N/A'), 'A0_V3034C_distribuicao', index=False)

        # Figure A0 – reasons for not attending school
        top = dist.dropna(subset=['motivo']).head(12).copy()
        top['motivo_short'] = top['motivo'].astype(str).str[:55]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(top['motivo_short'][::-1], top['pct'][::-1], color=BLUE)
        ax.set_xlabel('% das respostas')
        ax.set_title(f'Motivo principal para não frequentar escola\n(V3034C, PNADC {YEAR})')
        plt.tight_layout()
        save_fig('A0_V3034C_distribuicao')

    # --- Demographic profile table (using label columns) ---
    profile = {}
    for col_name, display in [('sexo', 'Sexo'), ('raca_cor', 'Raça/Cor'),
                               ('faixa_etaria', 'Faixa etária'),
                               ('nivel_instrucao', 'Nível instrução'),
                               ('zona', 'Zona')]:
        use = best_col(grp, col_name)
        if use in grp.columns:
            s = grp[use].value_counts(normalize=True).mul(100).round(1)
            for k, v in s.items():
                profile[f'{display}: {k}'] = v
    if profile:
        profile_df = pd.DataFrame.from_dict(profile, orient='index', columns=['pct_%'])
        save_table(profile_df, f'A1_{tag}_perfil_demografico')

    # --- Age stats ---
    if 'idade' in grp.columns:
        age_stats = grp['idade'].agg(['mean', 'median', 'std', 'min', 'max']).round(1)
        save_table(age_stats.to_frame('idade'), f'A2_{tag}_idade_stats', index=True)

    # --- Labor status (use label column) ---
    use_cond = best_col(grp, 'condicao_ocupacao')
    if use_cond in grp.columns:
        lab = grp[use_cond].value_counts().reset_index()
        lab.columns = ['status', 'n']
        lab['pct'] = (lab['n'] / lab['n'].sum() * 100).round(1)
        save_table(lab, f'A3_{tag}_status_laboral', index=False)

    # --- Income ---
    inc_col = next((c for c in ['renda_habitual_principal', 'renda_habitual_todos']
                    if c in grp.columns and grp[c].notna().any()), None)
    if inc_col:
        r_grp  = grp[inc_col].dropna()
        r_all  = merged[inc_col].dropna()
        inc_df = pd.DataFrame({
            'grupo': ['Concurseiros', 'Todos ocupados'],
            'media':   [r_grp.mean(),   r_all.mean()],
            'mediana': [r_grp.median(), r_all.median()],
            'p25':     [r_grp.quantile(.25), r_all.quantile(.25)],
            'p75':     [r_grp.quantile(.75), r_all.quantile(.75)],
            'n':       [len(r_grp),     len(r_all)],
        })
        save_table(inc_df, f'A4_{tag}_renda', index=False)

    # --- Figure: education level vs. all workers (use label column) ---
    use_ni_grp = best_col(grp, 'nivel_instrucao')
    use_ni_all = best_col(merged, 'nivel_instrucao')
    if use_ni_grp in grp.columns and use_ni_all in merged.columns:
        ed_grp = grp[use_ni_grp].value_counts(normalize=True).mul(100).rename('Concurseiros')
        ed_all = merged[use_ni_all].value_counts(normalize=True).mul(100).rename('Todos')
        ed = pd.concat([ed_grp, ed_all], axis=1).fillna(0)
        # Shorten labels for display
        ed.index = ed.index.astype(str).str[:40]
        fig, ax = plt.subplots(figsize=(12, 5))
        ed.plot(kind='barh', ax=ax, color=[BLUE, GREY])
        ax.set_xlabel('% do grupo')
        ax.set_title(f'Nível de instrução – {label}')
        ax.legend(loc='lower right')
        plt.tight_layout()
        save_fig(f'A5_{tag}_nivel_instrucao')

    # --- Figure: age distribution ---
    if 'idade' in grp.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(merged['idade'].dropna(), bins=range(14, 75), color=GREY,
                alpha=0.5, density=True, label='Todos')
        ax.hist(grp['idade'].dropna(),   bins=range(14, 75), color=BLUE,
                alpha=0.7, density=True, label=label)
        ax.set_xlabel('Idade')
        ax.set_ylabel('Densidade')
        ax.set_title(f'Distribuição de idade – {label}')
        ax.legend()
        plt.tight_layout()
        save_fig(f'A6_{tag}_distribuicao_idade')

# ──────────────────────────────────────────────────────────────────────────────
# B.  CONCURSO-SEEKER TRAJECTORIES
# ──────────────────────────────────────────────────────────────────────────────
print('\n── Section B: Trajectories ──────────────────────────────────')

if 'buscando_via_concurso' in merged.columns:
    seekers = merged[merged['buscando_via_concurso'] == 1].copy()
    n_seek  = len(seekers)
    print(f'  Concurso seekers (person-quarter): {n_seek:,}')

    # --- B1: What were they doing before (t-1)? ---
    for col, label in [('posicao_emprego_label_t1',  'Posição no emprego (t-1)'),
                        ('condicao_ocupacao_label_t1', 'Cond. ocupação (t-1)'),
                        ('setor_atividade_label_t1',   'Setor atividade (t-1)')]:
        # Fall back to numeric+decode if label lag not available
        fallback = col.replace('_label_t1', '_t1')
        use = col if col in seekers.columns else fallback
        if use in seekers.columns:
            s = (seekers[use]
                 .pipe(lambda x: decode(x, use) if use == fallback else x)
                 .value_counts(dropna=False)
                 .reset_index())
            s.columns = [label, 'n']
            s['pct'] = (s['n'] / n_seek * 100).round(1)
            save_table(s, f'B1_{col}', index=False)

    # Prev income
    if 'renda_habitual_principal_t1' in seekers.columns:
        r = seekers['renda_habitual_principal_t1'].dropna()
        inc = pd.DataFrame({'stat': ['mean', 'median', 'p25', 'p75', 'n'],
                            'valor': [r.mean(), r.median(), r.quantile(.25),
                                      r.quantile(.75), len(r)]})
        save_table(inc, 'B2_renda_antes_concurso', index=False)

    # --- B2: Transition matrix (what happened after, t+1) ---
    def classify_next(row):
        if pd.isna(row.get('ocupado_f1')):
            return 'Não observado (attrition)'
        if row.get('servidor_publico_f1') == 1:
            return 'Virou servidor público'
        if row.get('empregado_setor_pub_f1') == 1:
            return 'Empregado setor público (sem estatutário)'
        if row.get('buscando_via_concurso_f1') == 1:
            return 'Ainda buscando via concurso'
        if row.get('ocupado_f1') == 1 and row.get('formal_f1') == 1:
            return 'Empregado setor formal privado'
        if row.get('ocupado_f1') == 1 and row.get('informal_f1') == 1:
            return 'Empregado setor informal'
        if row.get('ocupado_f1') == 1:
            return 'Empregado (sem info detalhe)'
        if row.get('desocupado_f1') == 1:
            return 'Desocupado (outro método)'
        return 'Fora da força de trabalho'

    seekers['proximo_status'] = seekers.apply(classify_next, axis=1)
    trans = (seekers['proximo_status']
             .value_counts()
             .reset_index()
             .rename(columns={'index': 'status_t1', 'proximo_status': 'n'}))
    trans.columns = ['status_t1', 'n']
    trans['pct'] = (trans['n'] / n_seek * 100).round(1)
    save_table(trans, 'B3_transicoes_pos_concurso', index=False)

    # Figure B1: transitions bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [GREEN if 'servidor' in s else BLUE if 'concurso' in s else GREY
              for s in trans['status_t1']]
    ax.barh(trans['status_t1'][::-1], trans['pct'][::-1], color=colors[::-1])
    ax.set_xlabel('% dos concurseiros')
    ax.set_title('O que aconteceu no trimestre seguinte?\n(Buscadores via concurso, V4072A)')
    plt.tight_layout()
    save_fig('B1_transicoes_concurseiros')

    # --- B3: Profile of those who succeeded (became public servants) ---
    if 'servidor_publico_f1' in seekers.columns:
        winners = seekers[seekers['servidor_publico_f1'] == 1].copy()
        losers  = seekers[seekers['proximo_status'] == 'Ainda buscando via concurso'].copy()
        print(f'  Became public servant: {len(winners):,}  |  Still seeking: {len(losers):,}')

        rows = []
        for grp_name, grp_df in [('Aprovados (virou servidor)', winners),
                                   ('Continuou buscando',        losers)]:
            r = {
                'grupo': grp_name,
                'n': len(grp_df),
            }
            for col in ['idade', 'anos_estudo']:
                if col in grp_df.columns:
                    r[f'{col}_media'] = grp_df[col].mean()
            if 'renda_habitual_principal_f1' in grp_df.columns:
                r['renda_nova_media'] = grp_df['renda_habitual_principal_f1'].mean()
            if 'renda_habitual_principal_t1' in grp_df.columns:
                r['renda_anterior_media'] = grp_df['renda_habitual_principal_t1'].mean()
            rows.append(r)
        if rows:
            save_table(pd.DataFrame(rows), 'B4_aprovados_vs_continuando', index=False)

        # Figure B2: income before vs after for winners
        if all(c in winners.columns for c in
               ['renda_habitual_principal_t1', 'renda_habitual_principal_f1']):
            before = winners['renda_habitual_principal_t1'].dropna()
            after  = winners['renda_habitual_principal_f1'].dropna()
            if len(before) > 2 and len(after) > 2:
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(before, bins=20, alpha=0.6, color=GREY,  label='Renda antes (t-1)', density=True)
                ax.hist(after,  bins=20, alpha=0.6, color=GREEN, label='Renda depois (t+1)', density=True)
                ax.axvline(before.median(), color=GREY, linestyle='--', label=f'Mediana antes: R${before.median():,.0f}')
                ax.axvline(after.median(),  color=GREEN, linestyle='--', label=f'Mediana depois: R${after.median():,.0f}')
                ax.set_xlabel('Renda mensal (R$)')
                ax.set_title(f'Renda antes e depois de entrar no serviço público\n(aprovados, {YEAR})')
                ax.legend(fontsize=8)
                plt.tight_layout()
                save_fig('B2_renda_antes_depois_aprovados')

    # --- B4: Previous-quarter status for those who gave up ---
    gave_up = seekers[seekers['proximo_status'].isin(
        ['Empregado setor formal privado', 'Empregado setor informal',
         'Desocupado (outro método)', 'Fora da força de trabalho']
    )].copy()
    if len(gave_up) > 0:
        dest_col = 'posicao_emprego_label_f1'
        fallback  = 'posicao_emprego_f1'
        use = dest_col if dest_col in gave_up.columns else fallback
        t = (gave_up[use]
             .pipe(lambda x: decode(x, use) if use == fallback else x)
             .value_counts(dropna=False)
             .reset_index())
        t.columns = ['destino', 'n']
        t['pct'] = (t['n'] / len(gave_up) * 100).round(1)
        save_table(t, 'B5_destino_quem_desistiu', index=False)

else:
    print('  buscando_via_concurso not found — skip section B.')

# ──────────────────────────────────────────────────────────────────────────────
# C.  CURRENT PUBLIC SERVANTS
# ──────────────────────────────────────────────────────────────────────────────
print('\n── Section C: Public servants ───────────────────────────────')

if 'servidor_publico' not in merged.columns:
    print('  servidor_publico not found — skip section C.')
else:
    pub  = merged[merged['servidor_publico'] == 1].copy()
    priv = merged[(merged['servidor_publico'] != 1) & (merged['ocupado'] == 1)].copy()
    print(f'  Public: {len(pub):,}  |  Private: {len(priv):,}')

    # --- C1: Demographic comparison table ---
    rows = []
    for grp_name, grp_df in [('Servidor público', pub), ('Trabalhador privado', priv)]:
        r = {'grupo': grp_name, 'n': len(grp_df)}
        for col in ['idade', 'anos_estudo']:
            if col in grp_df.columns:
                r[f'{col}_media'] = round(grp_df[col].mean(), 1)
        for col in ['sexo', 'raca_cor', 'nivel_instrucao']:
            use = best_col(grp_df, col)
            if use in grp_df.columns:
                top = grp_df[use].value_counts(normalize=True).mul(100).head(3)
                for k, v in top.items():
                    r[f'{col}_{k}_%'] = round(v, 1)
        inc_col = next((c for c in ['renda_habitual_principal', 'renda_habitual_todos']
                        if c in grp_df.columns and grp_df[c].notna().any()), None)
        if inc_col:
            r['renda_media']   = round(grp_df[inc_col].mean(), 0)
            r['renda_mediana'] = round(grp_df[inc_col].median(), 0)
        rows.append(r)
    save_table(pd.DataFrame(rows), 'C1_perfil_publico_vs_privado', index=False)

    # --- Figure C1: income by education – public vs private (use label column) ---
    inc_col = next((c for c in ['renda_habitual_principal', 'renda_habitual_todos']
                    if c in merged.columns and merged[c].notna().any()), None)
    ni_col = best_col(merged, 'nivel_instrucao')
    if inc_col and ni_col in merged.columns:
        both = merged[merged['ocupado'] == 1].copy()
        both = both[both[inc_col].notna() & both[ni_col].notna()]
        both['Setor'] = both['servidor_publico'].map({1: 'Público', 0: 'Privado'}).fillna('Privado')

        med = (both.groupby([ni_col, 'Setor'])[inc_col]
               .median().reset_index()
               .rename(columns={ni_col: 'nivel_instrucao', inc_col: 'renda_mediana'}))
        # Shorten labels for table
        med_tbl = med.copy()
        med_tbl['nivel_instrucao'] = med_tbl['nivel_instrucao'].astype(str).str[:50]
        save_table(med_tbl, 'C2_renda_por_instrucao_setor', index=False)

        piv = med.pivot(index='nivel_instrucao', columns='Setor', values='renda_mediana').fillna(0)
        piv.index = piv.index.astype(str).str[:40]
        fig, ax = plt.subplots(figsize=(11, 5))
        piv.plot(kind='bar', ax=ax, color=[BLUE, RED], width=0.7)
        ax.set_xlabel('')
        ax.set_ylabel('Renda mediana (R$)')
        ax.set_title(f'Renda mediana por nível de instrução\nSetor público vs. privado (PNADC {YEAR})')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x:,.0f}'))
        ax.tick_params(axis='x', rotation=40)
        ax.legend(title='Setor')
        plt.tight_layout()
        save_fig('C1_renda_instrucao_publico_privado')

    # --- Figure C2: age distribution public vs private ---
    if 'idade' in merged.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(priv['idade'].dropna(), bins=range(14, 75), density=True,
                alpha=0.5, color=RED,  label='Privado')
        ax.hist(pub['idade'].dropna(),  bins=range(14, 75), density=True,
                alpha=0.6, color=BLUE, label='Público')
        ax.set_xlabel('Idade')
        ax.set_ylabel('Densidade')
        ax.set_title('Distribuição de idade: público vs. privado')
        ax.legend()
        plt.tight_layout()
        save_fig('C2_idade_publico_privado')

    # --- Figure C3: income box/violin by sector and hours ---
    if inc_col and 'horas_habituais_principal' in merged.columns:
        h = merged[merged['ocupado'] == 1].copy()
        h = h[h[inc_col].notna() & h['horas_habituais_principal'].notna()]
        h['Setor'] = h['servidor_publico'].map({1: 'Público', 0: 'Privado'}).fillna('Privado')
        h_stats = h.groupby('Setor').agg(
            renda_media=(inc_col, 'mean'),
            renda_mediana=(inc_col, 'median'),
            horas_media=('horas_habituais_principal', 'mean'),
            n=('id_domicilio', 'count')
        ).reset_index()
        save_table(h_stats, 'C3_renda_horas_setor', index=False)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, var, label_y in [(axes[0], inc_col,                   'Renda habitual (R$)'),
                                  (axes[1], 'horas_habituais_principal', 'Horas/semana')]:
            plot_data = [h[h['Setor'] == s][var].dropna() for s in ['Público', 'Privado']]
            ax.boxplot(plot_data, tick_labels=['Público', 'Privado'],
                       medianprops={'color': 'black', 'linewidth': 2})
            ax.set_ylabel(label_y)
        axes[0].set_title('Renda')
        axes[1].set_title('Horas trabalhadas')
        fig.suptitle('Servidor público vs. trabalhador privado', fontsize=12)
        plt.tight_layout()
        save_fig('C3_renda_horas_publico_privado')

    # --- C2: Where did public servants come from? (t-1) ---
    for lag_col, name in [('posicao_emprego_label_t1',  'C4_origem_servidores_posicao_t1'),
                           ('condicao_ocupacao_label_t1', 'C5_origem_servidores_cond_ocupacao_t1')]:
        fallback = lag_col.replace('_label_t1', '_t1')
        use = lag_col if lag_col in pub.columns else fallback
        if use in pub.columns:
            s = (pub[use]
                 .pipe(lambda x: decode(x, use) if use == fallback else x)
                 .value_counts(dropna=False)
                 .reset_index())
            s.columns = ['status_anterior', 'n']
            s['pct'] = (s['n'] / len(pub) * 100).round(1)
            save_table(s, name, index=False)

    # --- C3: Newly recruited public servants ---
    if 'servidor_publico_t1' in pub.columns:
        new_pub = pub[(pub['servidor_publico_t1'] != 1) & pub['servidor_publico_t1'].notna()]
        print(f'  Newly became public servant: {len(new_pub):,}')
        if len(new_pub) > 0:
            use = 'posicao_emprego_label_t1' if 'posicao_emprego_label_t1' in new_pub.columns else 'posicao_emprego_t1'
            t = (new_pub[use]
                 .pipe(lambda x: decode(x, use) if 'label' not in use else x)
                 .value_counts(dropna=False)
                 .reset_index())
            t.columns = ['origem', 'n']
            t['pct'] = (t['n'] / len(new_pub) * 100).round(1)
            save_table(t, 'C6_novos_servidores_origem', index=False)

            if 'buscando_via_concurso_t1' in new_pub.columns:
                n_conc = (new_pub['buscando_via_concurso_t1'] == 1).sum()
                via_conc = pd.DataFrame([{
                    'grupo': 'Novos servidores',
                    'n_total': len(new_pub),
                    'vieram_via_concurso': int(n_conc),
                    'pct_via_concurso': round(100 * n_conc / len(new_pub), 1)
                }])
                save_table(via_conc, 'C7_novos_servidores_via_concurso', index=False)
                print(f'  Of new public servants, {n_conc} ({100*n_conc/len(new_pub):.1f}%) '
                      f'were concurso seekers at t-1')

    # --- Figure C4: who becomes public servant? flow bar chart ---
    if 'posicao_emprego_label_t1' in pub.columns or 'posicao_emprego_t1' in pub.columns:
        use = 'posicao_emprego_label_t1' if 'posicao_emprego_label_t1' in pub.columns else 'posicao_emprego_t1'
        orig_s = (pub[use]
                  .pipe(lambda x: decode(x, use) if 'label' not in use else x)
                  .value_counts(dropna=False)
                  .head(8))
        if len(orig_s) > 0:
            orig_s.index = orig_s.index.astype(str).str[:45]
            fig, ax = plt.subplots(figsize=(10, 4))
            orig_s.iloc[::-1].plot(kind='barh', ax=ax, color=BLUE)
            ax.set_xlabel('Número de servidores públicos')
            ax.set_title('De onde vieram os servidores públicos?\n(posição no emprego no trimestre anterior, t-1)')
            plt.tight_layout()
            save_fig('C4_origem_servidores')

    # --- C4: Income premium by education (use label column for groupby) ---
    ni_col = best_col(merged, 'nivel_instrucao')
    if inc_col and ni_col in merged.columns:
        prem = (merged[merged['ocupado'] == 1]
                .groupby([ni_col, 'servidor_publico'])[inc_col]
                .median().unstack('servidor_publico'))
        prem.columns = ['Privado', 'Público'] if 0 in prem.columns else prem.columns
        if 'Público' in prem.columns and 'Privado' in prem.columns:
            prem['Premio_%'] = ((prem['Público'] / prem['Privado']) - 1).mul(100).round(1)
        prem_tbl = prem.copy()
        prem_tbl.index = prem_tbl.index.astype(str).str[:50]
        save_table(prem_tbl, 'C8_premio_salarial_publico_por_instrucao')

        if 'Premio_%' in prem.columns:
            prem_fig = prem['Premio_%'].dropna().copy()
            prem_fig.index = prem_fig.index.astype(str).str[:40]
            fig, ax = plt.subplots(figsize=(10, 4))
            prem_fig.plot(kind='bar', ax=ax, color=BLUE)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_ylabel('Prêmio salarial público (%)')
            ax.set_title('Prêmio salarial do setor público por nível de instrução\n'
                         f'(mediana pública / mediana privada − 1, {YEAR})')
            ax.tick_params(axis='x', rotation=40)
            plt.tight_layout()
            save_fig('C5_premio_salarial_por_instrucao')

# ──────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('Done.')
print(f'  Tables → {T_DIR}')
print(f'  Figures → {F_DIR}')
