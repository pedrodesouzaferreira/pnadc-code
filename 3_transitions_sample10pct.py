"""
3_transitions_sample10pct.py  –  Labor-market transition analysis, PNADC 2024
              10 % random sample of INDIVIDUALS (all their quarters retained)
==============================================================================
Identical logic to 3_transitions.py, but uses a two-pass strategy so the
full dataset is NEVER loaded into memory:

  Pass 1 – reads only the two ID columns (id_domicilio + num_ordem) to
            enumerate all unique individuals, then draws a 10 % random sample.
            Memory cost: ~2 columns × N rows (very cheap).

  Pass 2 – reads the CSV in chunks of CHUNK_SIZE rows.  Each chunk is
            filtered against the sampled-individual set before being kept.
            At most one chunk lives in memory at a time.

This is useful for:
  • Quick iteration / debugging without loading the full dataset
  • Checking that results are stable across random subsamples

Sampling details
----------------
  Unit of sampling : individual  (id_domicilio + num_ordem)
  Fraction         : 10 %  (SAMPLE_FRAC = 0.10)
  Seed             : SEED = 42  (change for a different draw)
  All quarters for a sampled individual are retained, so the rotating-panel
  structure is preserved within the subsample.

Outputs
-------
  Same tables and figures as 3_transitions.py, but written to
  Output/Tables/sample10pct/  and  Output/Figures/sample10pct/
  so they never overwrite the full-sample results.

Run order: 0_import.py → 1_clean.py (YEAR=2024) → 3_transitions_sample10pct.py
           (falls back to raw CSV if cleaned file is absent, just like the
           original script)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT  = '/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC'
YEAR  = 2024
MW    = 1_412   # 2024 minimum wage (R$)

# ── Sampling parameters ───────────────────────────────────────────────────────
SAMPLE_FRAC = 0.10   # fraction of individuals to keep
SEED        = 42     # random seed for reproducibility

T_DIR = f'{ROOT}/Output/Tables/sample10pct'
F_DIR = f'{ROOT}/Output/Figures/sample10pct'
os.makedirs(T_DIR, exist_ok=True)
os.makedirs(F_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
STATE_COLORS = {
    'Fora da PEA'   : '#888888',
    'Desocupado'    : '#E05C5C',
    'Conta-própria' : '#F4A623',
    'Setor Privado' : '#2B7BB9',
    'Setor Público' : '#3AAA5C',
}
STATES = list(STATE_COLORS.keys())

# ── Helpers ───────────────────────────────────────────────────────────────────
def save_table(df, name, index=True):
    path = f'{T_DIR}/{name}.tex'
    latex = df.to_latex(
        index=index,
        float_format=lambda x: f'{x:,.2f}',
        na_rep='--',
        escape=True,
    )
    with open(path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f'  [TABLE] {name}.tex  ({df.shape[0]} rows)')

def save_fig(name, dpi=150):
    path = f'{F_DIR}/{name}.png'
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'  [FIG]   {name}.png')

def fmt_brl(x, _=None):
    return f'R${x:,.0f}'

# ─────────────────────────────────────────────────────────────────────────────
# 1 + 2.  TWO-PASS LOAD + INDIVIDUAL SAMPLE
#
#  Pass 1 – read only the two ID columns to draw the 10 % individual sample.
#            The full dataset is NOT loaded; only ~2 columns are in memory.
#  Pass 2 – stream the CSV in chunks of CHUNK_SIZE rows.  Each chunk is
#            filtered to sampled individuals immediately and then discarded,
#            so at most one chunk is in memory at a time.
# ─────────────────────────────────────────────────────────────────────────────
PANEL_ID   = ['id_domicilio', 'num_ordem']
CHUNK_SIZE = 200_000   # rows per chunk during pass 2; lower = less RAM

cleaned_path = f'{ROOT}/Cleaned Data/PNADC_limpo_{YEAR}.csv'
raw_path     = f'{ROOT}/Raw Data/PNADC_microdados_{YEAR}.csv'

USE_CLEANED = os.path.exists(cleaned_path)
data_path   = cleaned_path if USE_CLEANED else raw_path

# Columns to keep (cleaned names)
keep_cleaned = [
    'ano', 'trimestre', 'id_domicilio', 'num_ordem', 'peso',
    'condicao_forca_trabalho', 'condicao_ocupacao', 'posicao_emprego',
    'renda_habitual_principal', 'renda_habitual_todos',
]
# Equivalent raw names (if cleaned file is absent)
raw_cols_needed = [
    'ano', 'trimestre', 'id_domicilio',
    'V2003',   # num_ordem
    'V1028',   # peso
    'VD4001',  # condicao_forca_trabalho
    'VD4002',  # condicao_ocupacao
    'VD4009',  # posicao_emprego
    'VD4016',  # renda_habitual_principal
    'VD4019',  # renda_habitual_todos
]
raw_rename = {
    'V1028': 'peso', 'V2003': 'num_ordem',
    'VD4001': 'condicao_forca_trabalho',
    'VD4002': 'condicao_ocupacao',
    'VD4009': 'posicao_emprego',
    'VD4016': 'renda_habitual_principal',
    'VD4019': 'renda_habitual_todos',
}

# Detect which columns actually exist in the file
header = pd.read_csv(data_path, nrows=0).columns.tolist()
if USE_CLEANED:
    use_cols = [c for c in keep_cleaned if c in header]
    id_cols_file = ['id_domicilio', 'num_ordem']
else:
    use_cols = [c for c in raw_cols_needed if c in header]
    id_cols_file = [c for c in ['id_domicilio', 'V2003'] if c in header]

# ── Pass 1: read only ID columns, draw the sample ────────────────────────────
print(f'Pass 1 – reading ID columns from: {data_path}')
ids_only = pd.read_csv(data_path, usecols=id_cols_file, low_memory=False)
if not USE_CLEANED and 'V2003' in ids_only.columns:
    ids_only.rename(columns={'V2003': 'num_ordem'}, inplace=True)

individuals = ids_only.drop_duplicates()
n_total     = len(individuals)
sampled_ids = individuals.sample(frac=SAMPLE_FRAC, random_state=SEED)
n_sampled   = len(sampled_ids)
del ids_only   # free memory immediately

# Build a fast lookup set: {(id_domicilio, num_ordem), ...}
sampled_set = set(zip(sampled_ids['id_domicilio'], sampled_ids['num_ordem']))

print(f'\n── 10 % individual sample ───────────────────────────────────')
print(f'  Total individuals  : {n_total:,}')
print(f'  Sampled individuals: {n_sampled:,}  ({100*n_sampled/n_total:.1f} %)')
print(f'────────────────────────────────────────────────────────────\n')

# ── Pass 2: stream the CSV in chunks, drop non-sampled rows immediately ───────
print(f'Pass 2 – streaming data in chunks of {CHUNK_SIZE:,} rows...')
chunks = []
for i, chunk in enumerate(pd.read_csv(data_path, usecols=use_cols,
                                       chunksize=CHUNK_SIZE, low_memory=False)):
    if not USE_CLEANED:
        chunk.rename(columns={k: v for k, v in raw_rename.items()
                               if k in chunk.columns}, inplace=True)
    # Set-membership filter — O(1) per row, no row-by-row Python loop
    keys = list(zip(chunk['id_domicilio'], chunk['num_ordem']))
    keep = [k in sampled_set for k in keys]
    filtered = chunk[keep]
    if len(filtered):
        chunks.append(filtered)
    if (i + 1) % 10 == 0:
        print(f'  ... processed {(i+1)*CHUNK_SIZE:,} rows so far')

df = pd.concat(chunks, ignore_index=True)
del chunks, sampled_set   # free memory

print(f'Rows after sampling: {df.shape[0]:,}  ×  {df.shape[1]} columns')

# ── Coerce numeric types ──────────────────────────────────────────────────────
for col in ['trimestre', 'ano', 'peso',
            'posicao_emprego', 'condicao_forca_trabalho', 'condicao_ocupacao',
            'renda_habitual_principal', 'renda_habitual_todos']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ─────────────────────────────────────────────────────────────────────────────
# 3. VECTORIZED STATE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
print('Classifying labor market states (vectorized)...')

forca = df['condicao_forca_trabalho']
ocup  = df['condicao_ocupacao']
pos   = df['posicao_emprego']

conditions = [
    forca == 2,
    (forca == 1) & (ocup == 2),
    (forca == 1) & (ocup == 1) & pos.isin([3, 4, 5]),
    (forca == 1) & (ocup == 1) & (pos == 7),
    (forca == 1) & (ocup == 1) & pos.isin([1, 2, 6, 8, 9, 10]),
]
choices = ['Fora da PEA', 'Desocupado', 'Setor Público',
           'Conta-própria', 'Setor Privado']

df['estado_lm'] = np.select(conditions, choices, default='')
df['estado_lm'] = df['estado_lm'].replace('', np.nan)

print(df['estado_lm'].value_counts(dropna=False))

# ─────────────────────────────────────────────────────────────────────────────
# 4. BUILD TRANSITION PAIRS  (consecutive quarters only)
# ─────────────────────────────────────────────────────────────────────────────
print('\nBuilding transition pairs...')

df = df.sort_values(PANEL_ID + ['ano', 'trimestre']).reset_index(drop=True)

INC = next((c for c in ['renda_habitual_principal', 'renda_habitual_todos']
            if c in df.columns and df[c].notna().any()), None)
print(f'Income column: {INC}')

grp = df.groupby(PANEL_ID)
df['estado_t1']    = grp['estado_lm'].shift(-1)
df['trimestre_t1'] = grp['trimestre'].shift(-1)
df['ano_t1']       = grp['ano'].shift(-1)
if INC:
    df[f'{INC}_t1'] = grp[INC].shift(-1)

transitions = df[
    (df['ano']       == YEAR) &
    (df['ano_t1']    == YEAR) &
    ((df['trimestre_t1'] - df['trimestre']) == 1) &
    df['estado_lm'].notna() &
    df['estado_t1'].notna()
].copy()

print(f'Consecutive-quarter transition pairs: {len(transitions):,}')

# ─────────────────────────────────────────────────────────────────────────────
# 5. TRANSITION MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def build_transition_matrix(df_trans, weight_col=None):
    """Compute count and probability transition matrices (5×5)."""
    if weight_col and weight_col in df_trans.columns:
        w = pd.to_numeric(df_trans[weight_col], errors='coerce').fillna(1)
        counts = (df_trans.assign(_w=w)
                  .groupby(['estado_lm', 'estado_t1'])['_w']
                  .sum().unstack(fill_value=0))
    else:
        counts = (df_trans.groupby(['estado_lm', 'estado_t1'])
                  .size().unstack(fill_value=0))

    counts = counts.reindex(index=STATES, columns=STATES, fill_value=0)
    probs  = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return counts, probs


cnt_all, prb_all = build_transition_matrix(transitions)
cnt_wt,  prb_wt  = build_transition_matrix(transitions, weight_col='peso')

print('\nFull-year transition probability matrix (sample):')
print(prb_all.round(3))

save_table(cnt_all.astype(int), 'D1_transicoes_contagem_anual')
save_table(prb_all.mul(100).round(2), 'D2_transicoes_probabilidade_anual')
save_table(cnt_wt.round(0).astype(int), 'D3_transicoes_pesado_anual')
save_table(prb_wt.mul(100).round(2), 'D4_transicoes_probabilidade_pesado_anual')

quarter_matrices = {}
for q in [1, 2, 3]:
    sub = transitions[transitions['trimestre'] == q]
    cnt_q, prb_q = build_transition_matrix(sub)
    quarter_matrices[q] = (cnt_q, prb_q)
    save_table(prb_q.mul(100).round(2), f'D5_transicoes_prob_Q{q}_Q{q+1}')
    print(f'  Q{q}→Q{q+1}: {len(sub):,} pairs')

# ─────────────────────────────────────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── Figure D1: Heatmap – full year ────────────────────────────────────────────
print('\nPlotting Figure D1...')
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(prb_all.mul(100), annot=True, fmt='.1f', cmap='Blues',
            linewidths=0.5, linecolor='grey',
            annot_kws={'size': 9}, ax=ax,
            cbar_kws={'label': '% de transições'})
ax.set_xlabel('Estado em $t+1$', fontsize=11)
ax.set_ylabel('Estado em $t$', fontsize=11)
ax.set_title(f'Probabilidades de transição entre estados do mercado de trabalho\n'
             f'PNADC {YEAR} – 10% amostra – todos os trimestres (linhas somam 100%)',
             fontsize=11)
ax.tick_params(axis='x', rotation=30)
ax.tick_params(axis='y', rotation=0)
plt.tight_layout()
save_fig('D1_heatmap_transicoes_anual')

# ── Figure D2: Per-quarter heatmaps ──────────────────────────────────────────
print('Plotting Figure D2...')
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
for i, q in enumerate([1, 2, 3]):
    _, prb_q = quarter_matrices[q]
    sns.heatmap(prb_q.mul(100), annot=True, fmt='.1f', cmap='Blues',
                linewidths=0.5, linecolor='grey',
                annot_kws={'size': 8}, ax=axes[i],
                vmin=0, vmax=100,
                cbar=(i == 2),
                cbar_kws={'label': '%'} if i == 2 else {})
    axes[i].set_title(f'Q{q} → Q{q+1}', fontsize=12)
    axes[i].set_xlabel('Estado em $t+1$', fontsize=9)
    if i == 0:
        axes[i].set_ylabel('Estado em $t$', fontsize=9)
    axes[i].tick_params(axis='x', rotation=30)
    axes[i].tick_params(axis='y', rotation=0)
fig.suptitle(f'Probabilidades de transição por trimestre – 10% amostra (PNADC {YEAR})',
             fontsize=13)
plt.tight_layout()
save_fig('D2_heatmap_transicoes_por_trimestre')


# ── Helper: draw Sankey (Bézier flows) ───────────────────────────────────────
def _bezier_flow(ax, x1, y1_top, y1_bot, x2, y2_top, y2_bot, color, alpha=0.4):
    """Draw one cubic-Bézier flow band between two vertical bars."""
    cx = (x1 + x2) / 2
    verts = [
        (x1, y1_top),
        (cx, y1_top),
        (cx, y2_top),
        (x2, y2_top),
        (x2, y2_bot),
        (cx, y2_bot),
        (cx, y1_bot),
        (x1, y1_bot),
        (x1, y1_top),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    patch = mpatches.PathPatch(
        Path(verts, codes),
        facecolor=mcolors.to_rgba(color, alpha),
        edgecolor='none', zorder=1)
    ax.add_patch(patch)


def draw_sankey(count_matrix, title='', ax=None, bar_width=0.06, gap=0.025):
    """Sankey diagram with Bézier flows coloured by origin state."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    scale = count_matrix.values.sum()
    if scale == 0:
        return
    row_totals = count_matrix.sum(axis=1)
    col_totals = count_matrix.sum(axis=0)
    LEFT_X  = 0.1
    RIGHT_X = 0.85

    def bar_positions(totals):
        info = {}
        y = 1.0
        for s in STATES:
            h = totals[s] / scale
            info[s] = (y - h, y)
            y -= h + gap
        return info

    left_bars  = bar_positions(row_totals)
    right_bars = bar_positions(col_totals)

    for side, xpos, bars in [('left', LEFT_X, left_bars),
                               ('right', RIGHT_X, right_bars)]:
        for s in STATES:
            bot, top = bars[s]
            rect = mpatches.FancyBboxPatch(
                (xpos, bot), bar_width, top - bot,
                boxstyle='square,pad=0',
                facecolor=STATE_COLORS[s], edgecolor='white',
                linewidth=0.5, zorder=3)
            ax.add_patch(rect)
            mid = (bot + top) / 2
            pct = (top - bot) * 100
            if side == 'left':
                ax.text(xpos - 0.01, mid,
                        f'{s}\n({pct:.0f}%)',
                        ha='right', va='center', fontsize=7.5,
                        color=STATE_COLORS[s], fontweight='bold')
            else:
                ax.text(xpos + bar_width + 0.01, mid,
                        f'{s}\n({pct:.0f}%)',
                        ha='left', va='center', fontsize=7.5,
                        color=STATE_COLORS[s], fontweight='bold')

    left_cursor  = {s: left_bars[s][1]  for s in STATES}
    right_cursor = {s: right_bars[s][1] for s in STATES}

    for orig in STATES:
        for dest in STATES:
            cnt = count_matrix.loc[orig, dest]
            if cnt == 0:
                continue
            h = cnt / scale
            y1_top = left_cursor[orig];   y1_bot = y1_top - h
            y2_top = right_cursor[dest];  y2_bot = y2_top - h
            left_cursor[orig]  -= h
            right_cursor[dest] -= h
            _bezier_flow(ax,
                         LEFT_X + bar_width, y1_top, y1_bot,
                         RIGHT_X,            y2_top, y2_bot,
                         STATE_COLORS[orig])

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.08, 1.1)
    ax.axis('off')
    ax.set_title(title, fontsize=11, pad=12)


# ── Figure D3: Sankey – full year ─────────────────────────────────────────────
print('Plotting Figure D3 (Sankey full year)...')
fig, ax = plt.subplots(figsize=(11, 8))
draw_sankey(cnt_all,
            title=(f'Fluxo entre estados do mercado de trabalho\n'
                   f'PNADC {YEAR} – 10% amostra – todos os trimestres'),
            ax=ax)
plt.tight_layout()
save_fig('D3_sankey_anual')

# ── Figure D4: Sankey – per quarter ──────────────────────────────────────────
print('Plotting Figure D4 (Sankey per quarter)...')
fig, axes = plt.subplots(1, 3, figsize=(26, 8))
for i, q in enumerate([1, 2, 3]):
    cnt_q, _ = quarter_matrices[q]
    draw_sankey(cnt_q, title=f'Q{q} → Q{q+1}', ax=axes[i])
fig.suptitle(f'Fluxo entre estados do mercado de trabalho por trimestre – 10% amostra (PNADC {YEAR})',
             fontsize=13, y=1.01)
plt.tight_layout()
save_fig('D4_sankey_por_trimestre')


# ── Figure D5: Stacked bars – destination distribution per origin ─────────────
print('Plotting Figure D5...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (mat, lbl) in zip(axes, [(prb_all, 'Sem ponderação'),
                                   (prb_wt,  'Com ponderação amostral')]):
    bottom = np.zeros(len(STATES))
    x = np.arange(len(STATES))
    for dest in STATES:
        vals = mat[dest].values * 100
        ax.bar(x, vals, bottom=bottom, color=STATE_COLORS[dest], label=dest, width=0.65)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(STATES, rotation=28, ha='right', fontsize=8.5)
    ax.set_ylabel('% do estado de origem')
    ax.set_ylim(0, 108)
    ax.set_title(f'Distribuição de destinos por estado de origem\n{lbl}')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.legend(title='Destino (t+1)', fontsize=8, loc='upper right',
          bbox_to_anchor=(1.4, 1.0))
plt.tight_layout()
save_fig('D5_barra_empilhada_destinos')


# ── Figure D6: Absolute frequencies per quarter ───────────────────────────────
print('Plotting Figure D6...')
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for i, q in enumerate([1, 2, 3]):
    cnt_q, _ = quarter_matrices[q]
    bottom = np.zeros(len(STATES))
    x = np.arange(len(STATES))
    for dest in STATES:
        vals = cnt_q[dest].values.astype(float)
        axes[i].bar(x, vals / 1_000, bottom=bottom / 1_000,
                    color=STATE_COLORS[dest], label=dest, width=0.65)
        bottom += vals
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(STATES, rotation=28, ha='right', fontsize=8)
    axes[i].set_ylabel('Mil observações')
    axes[i].set_title(f'Q{q} → Q{q+1}')
    if i == 2:
        axes[i].legend(title='Destino', fontsize=7, loc='upper right',
                       bbox_to_anchor=(1.35, 1.0))
fig.suptitle(f'Frequência absoluta de transições por trimestre – 10% amostra (PNADC {YEAR})',
             fontsize=12)
plt.tight_layout()
save_fig('D6_frequencia_transicoes_trimestre')


# ─────────────────────────────────────────────────────────────────────────────
# 7. INDIVIDUAL PANEL TRAJECTORIES  (alluvial plot, Q1–Q4)
# ─────────────────────────────────────────────────────────────────────────────
print('\nBuilding individual trajectories for alluvial plot...')

panel_year = df[(df['ano'] == YEAR) & df['estado_lm'].notna()].copy()

traj_wide = (panel_year
             .pivot_table(index=PANEL_ID, columns='trimestre',
                          values='estado_lm', aggfunc='first')
             .rename(columns={1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
             .reindex(columns=['Q1', 'Q2', 'Q3', 'Q4']))

obs_ids = (panel_year[PANEL_ID + ['trimestre']]
           .drop_duplicates())
consec_check = obs_ids.merge(
    obs_ids.rename(columns={'trimestre': 'trimestre_nxt'}),
    on=PANEL_ID)
consec_check = consec_check[consec_check['trimestre_nxt'] == consec_check['trimestre'] + 1]
keep_ids = (consec_check[PANEL_ID]
            .drop_duplicates()
            .set_index(PANEL_ID)
            .index)
traj_panel = traj_wide[traj_wide.index.isin(keep_ids)]

print(f'Individuals with ≥1 consecutive quarter: {len(traj_panel):,}')
print('Distribution by number of quarters observed:')
for n in range(1, 5):
    print(f'  {n} quarters: {(traj_panel.notna().sum(axis=1) == n).sum():,}')


def build_flow(col_a, col_b, df_w):
    sub = df_w[[col_a, col_b]].dropna()
    return sub.groupby([col_a, col_b]).size().reset_index(name='n')

flows_12 = build_flow('Q1', 'Q2', traj_panel)
flows_23 = build_flow('Q2', 'Q3', traj_panel)
flows_34 = build_flow('Q3', 'Q4', traj_panel)


# ── Figure D7: Alluvial plot Q1–Q4 ───────────────────────────────────────────
print('Plotting Figure D7 (alluvial trajectories)...')

def alluvial_plot(traj_df, flows_list, quarter_names, title=''):
    n_q = len(quarter_names)
    fig, ax = plt.subplots(figsize=(4.5 * n_q, 8))

    BAR_W = 0.08
    GAP   = 0.025
    x_pos = {q: i for i, q in enumerate(quarter_names)}

    def compute_bars(q):
        counts = traj_df[q].value_counts(dropna=True)
        total  = counts.sum()
        y, info = 1.0, {}
        for s in STATES:
            h = counts.get(s, 0) / total if total > 0 else 0
            info[s] = (y - h, y)
            y -= h + GAP
        return info

    bar_info = {q: compute_bars(q) for q in quarter_names}

    for q in quarter_names:
        xi = x_pos[q]
        for s in STATES:
            bot, top = bar_info[q][s]
            if top - bot < 0.001:
                continue
            rect = mpatches.FancyBboxPatch(
                (xi - BAR_W/2, bot), BAR_W, top - bot,
                boxstyle='square,pad=0',
                facecolor=STATE_COLORS[s], edgecolor='white',
                linewidth=0.5, zorder=3)
            ax.add_patch(rect)
            if top - bot > 0.025:
                ax.text(xi, (bot+top)/2, f'{(top-bot)*100:.0f}%',
                        ha='center', va='center', fontsize=6.5,
                        color='white', fontweight='bold', zorder=4)
        ax.text(xi, 1.05, q, ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    for (q_from, q_to), flows in zip(
            zip(quarter_names[:-1], quarter_names[1:]),
            flows_list):

        xi_from = x_pos[q_from] + BAR_W/2
        xi_to   = x_pos[q_to]   - BAR_W/2

        total = traj_df[q_from].count()
        left_cursor  = {s: bar_info[q_from][s][1] for s in STATES}
        right_cursor = {s: bar_info[q_to][s][1]   for s in STATES}

        for _, row in flows.iterrows():
            orig, dest, cnt = row[q_from], row[q_to], row['n']
            if orig not in STATES or dest not in STATES or cnt == 0:
                continue
            h = cnt / total
            y1_top = left_cursor[orig];   y1_bot = y1_top - h
            y2_top = right_cursor[dest];  y2_bot = y2_top - h
            left_cursor[orig]  -= h
            right_cursor[dest] -= h
            _bezier_flow(ax, xi_from, y1_top, y1_bot,
                         xi_to, y2_top, y2_bot,
                         STATE_COLORS[orig], alpha=0.4)

    handles = [mpatches.Patch(facecolor=STATE_COLORS[s], label=s) for s in STATES]
    ax.legend(handles=handles, loc='lower center', ncol=5,
              bbox_to_anchor=(0.5, -0.06), fontsize=9, frameon=True)

    ax.set_xlim(-0.3, n_q - 0.7)
    ax.set_ylim(-0.12, 1.12)
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=15)
    return fig


fig = alluvial_plot(
    traj_panel,
    [flows_12, flows_23, flows_34],
    ['Q1', 'Q2', 'Q3', 'Q4'],
    title=(f'Trajetórias individuais no mercado de trabalho\n'
           f'PNADC {YEAR} – 10% amostra – indivíduos com ≥1 par consecutivo')
)
save_fig('D7_alluvial_trajetorias_individuais')


# ── Figure D8: State composition per quarter (stacked bar) ───────────────────
print('Plotting Figure D8...')
qtr_shares = []
for qi, ql in [(1,'Q1'),(2,'Q2'),(3,'Q3'),(4,'Q4')]:
    sub = panel_year[panel_year['trimestre'] == qi]['estado_lm']
    tot = len(sub)
    for s in STATES:
        qtr_shares.append({'Trimestre': ql, 'Estado': s,
                           'n': (sub == s).sum(),
                           'pct': 100*(sub == s).sum()/tot if tot else 0})
qtr_df = pd.DataFrame(qtr_shares)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(4)
bottoms = np.zeros(4)
for s in STATES:
    vals = [qtr_df[(qtr_df.Trimestre == q) & (qtr_df.Estado == s)]['pct'].values[0]
            for q in ['Q1','Q2','Q3','Q4']]
    ax.bar(x, vals, bottom=bottoms, color=STATE_COLORS[s], label=s, width=0.6)
    bottoms += np.array(vals)
ax.set_xticks(x); ax.set_xticklabels(['Q1','Q2','Q3','Q4'])
ax.set_ylabel('% da amostra')
ax.set_title(f'Composição do mercado de trabalho por trimestre\nPNADC {YEAR} – 10% amostra')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
save_fig('D8_composicao_por_trimestre')

comp_wide = qtr_df.pivot_table(index='Estado', columns='Trimestre', values='pct').reindex(STATES)
comp_wide.columns.name = None
save_table(comp_wide.round(1), 'D6_composicao_estados_por_trimestre')

# ─────────────────────────────────────────────────────────────────────────────
# 8. WAGE ANALYSIS FOR TRANSITIONS
# ─────────────────────────────────────────────────────────────────────────────
if INC and f'{INC}_t1' in transitions.columns:
    print('\nBuilding wage analysis...')
    renda_t1_col = f'{INC}_t1'

    wage_df = transitions[
        transitions[INC].notna() &
        transitions[renda_t1_col].notna() &
        (transitions[INC] > 0) &
        (transitions[renda_t1_col] > 0)
    ].copy()

    wage_df['delta_R']   = wage_df[renda_t1_col] - wage_df[INC]
    wage_df['delta_pct'] = 100 * wage_df['delta_R'] / wage_df[INC]
    wage_df['ganhou']    = (wage_df['delta_R'] > 0).astype(int)
    wage_df['perdeu']    = (wage_df['delta_R'] < 0).astype(int)
    wage_df['igual']     = (wage_df['delta_R'] == 0).astype(int)

    CORE = [
        ('Setor Privado', 'Setor Privado', 'Privado→Privado'),
        ('Setor Privado', 'Setor Público', 'Privado→Público'),
        ('Setor Público', 'Setor Privado', 'Público→Privado'),
        ('Setor Público', 'Setor Público', 'Público→Público'),
        ('Conta-própria', 'Setor Privado', 'CP→Privado'),
        ('Conta-própria', 'Setor Público', 'CP→Público'),
        ('Setor Privado', 'Conta-própria', 'Privado→CP'),
        ('Desocupado',    'Setor Privado', 'Desocup.→Privado'),
        ('Desocupado',    'Setor Público', 'Desocup.→Público'),
    ]

    wage_rows = []
    for orig, dest, lbl in CORE:
        sub = wage_df[(wage_df['estado_lm'] == orig) & (wage_df['estado_t1'] == dest)]
        if len(sub) < 5:
            continue
        wage_rows.append({
            'Transicao'          : lbl,
            'N'                  : len(sub),
            'Renda t media'      : round(sub[INC].mean(), 0),
            'Renda t+1 media'    : round(sub[renda_t1_col].mean(), 0),
            'Renda t mediana'    : round(sub[INC].median(), 0),
            'Renda t+1 mediana'  : round(sub[renda_t1_col].median(), 0),
            'Var RS media'       : round(sub['delta_R'].mean(), 0),
            'Var RS mediana'     : round(sub['delta_R'].median(), 0),
            'Var pct media'      : round(sub['delta_pct'].mean(), 1),
            'Ganhou pct'         : round(sub['ganhou'].mean() * 100, 1),
            'Perdeu pct'         : round(sub['perdeu'].mean() * 100, 1),
            'Igual pct'          : round(sub['igual'].mean() * 100, 1),
        })

    if wage_rows:
        wage_tbl = pd.DataFrame(wage_rows).set_index('Transicao')
        save_table(wage_tbl, 'D7_ganho_salarial_transicoes')
        print(f'  Wage table: {len(wage_tbl)} transition types')

        # ── Figure D9: Gain/loss bar chart ────────────────────────────────────
        print('Plotting Figure D9...')
        wt = pd.DataFrame(wage_rows).set_index('Transicao')
        fig, ax = plt.subplots(figsize=(10, 5))
        x  = np.arange(len(wt))
        w  = 0.25
        ax.bar(x - w,   wt['Ganhou pct'].values, width=w,
               color=STATE_COLORS['Setor Público'], label='Ganhou')
        ax.bar(x,       wt['Perdeu pct'].values, width=w,
               color=STATE_COLORS['Desocupado'],   label='Perdeu')
        ax.bar(x + w,   wt['Igual pct'].values,  width=w,
               color=STATE_COLORS['Fora da PEA'],  label='Mesmo salário')
        ax.set_xticks(x)
        ax.set_xticklabels(wt.index, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('% dos que fizeram a transição')
        ax.set_title(f'Proporção que ganhou / perdeu / manteve salário na transição\n'
                     f'PNADC {YEAR} – 10% amostra')
        ax.legend()
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        plt.tight_layout()
        save_fig('D9_proporcao_ganha_perde')

        # ── Figure D10: Average wage change by transition type ────────────────
        print('Plotting Figure D10...')
        wt_sorted = wt.sort_values('Var RS mediana')
        colors_bar = [STATE_COLORS['Setor Público'] if v >= 0
                      else STATE_COLORS['Desocupado']
                      for v in wt_sorted['Var RS mediana'].values]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(wt_sorted.index, wt_sorted['Var RS mediana'].values,
                color=colors_bar, edgecolor='white')
        ax.axvline(0, color='black', linewidth=1.0, linestyle='--')
        ax.set_xlabel('Variação salarial mediana (R$)')
        ax.set_title(f'Variação salarial mediana por tipo de transição\n'
                     f'PNADC {YEAR} – 10% amostra')
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_brl))
        plt.tight_layout()
        save_fig('D10_variacao_salarial_mediana')

    # ── Figure D11: Wage distributions for key transitions ───────────────────
    print('Plotting Figure D11...')
    key_trans = [
        ('Setor Privado', 'Setor Privado', 'Privado→Privado', STATE_COLORS['Setor Privado']),
        ('Setor Privado', 'Setor Público', 'Privado→Público', STATE_COLORS['Setor Público']),
        ('Setor Público', 'Setor Privado', 'Público→Privado', STATE_COLORS['Desocupado']),
    ]
    valid = [(o, d, lbl, c) for o, d, lbl, c in key_trans
             if len(wage_df[(wage_df.estado_lm==o)&(wage_df.estado_t1==d)]) >= 5]

    if valid:
        n_p = len(valid)
        fig, axes = plt.subplots(2, n_p, figsize=(5*n_p, 9))
        if n_p == 1:
            axes = axes.reshape(2, 1)
        for j, (orig, dest, lbl, col) in enumerate(valid):
            sub = wage_df[(wage_df.estado_lm == orig) & (wage_df.estado_t1 == dest)]
            ax_top = axes[0, j]
            clip_top = sub[[INC, renda_t1_col]].stack().quantile(0.99)
            ax_top.hist(sub[INC].clip(upper=clip_top), bins=30,
                        color='#AABBCC', alpha=0.7, density=True, label='Renda em $t$')
            ax_top.hist(sub[renda_t1_col].clip(upper=clip_top), bins=30,
                        color=col, alpha=0.6, density=True, label='Renda em $t+1$')
            ax_top.axvline(sub[INC].median(), color='#446688', linestyle='--',
                           linewidth=1.2, label=f'Med. t: {fmt_brl(sub[INC].median())}')
            ax_top.axvline(sub[renda_t1_col].median(), color=col, linestyle='-.',
                           linewidth=1.2,
                           label=f'Med. t+1: {fmt_brl(sub[renda_t1_col].median())}')
            ax_top.set_title(f'{lbl}  (n={len(sub):,})', fontsize=10)
            ax_top.set_xlabel('Renda (R$)')
            ax_top.set_ylabel('Densidade')
            ax_top.legend(fontsize=7)
            ax_top.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_brl))

            ax_bot = axes[1, j]
            delta = sub['delta_pct'].clip(-100, 200)
            ax_bot.hist(delta, bins=30, color=col, alpha=0.75, edgecolor='white')
            ax_bot.axvline(0, color='black', linestyle='--', linewidth=1.2)
            ax_bot.axvline(delta.median(), color='red', linestyle='-.',
                           linewidth=1.2, label=f'Mediana: {delta.median():.1f}%')
            pct_gain = (delta > 0).mean() * 100
            ax_bot.set_title(f'{pct_gain:.0f}% ganham salário na transição')
            ax_bot.set_xlabel('Variação salarial (%)')
            ax_bot.set_ylabel('Contagem')
            ax_bot.legend(fontsize=8)
            ax_bot.xaxis.set_major_formatter(mticker.PercentFormatter())

        fig.suptitle(f'Renda e variação salarial nas transições – PNADC {YEAR} – 10% amostra',
                     fontsize=12)
        plt.tight_layout()
        save_fig('D11_renda_variacao_transicoes')

else:
    print('  Income or lead-income variable not found — skipping wage analysis.')

# ─────────────────────────────────────────────────────────────────────────────
# 9. SUMMARY TABLE (long format)
# ─────────────────────────────────────────────────────────────────────────────
summary = []
tot_pairs = cnt_all.values.sum()
for orig in STATES:
    n_orig = int(cnt_all.loc[orig].sum())
    for dest in STATES:
        cnt = int(cnt_all.loc[orig, dest])
        if cnt == 0:
            continue
        summary.append({
            'Estado_t'   : orig,
            'Estado_t+1' : dest,
            'N'          : cnt,
            'Prob_linha' : round(100 * cnt / n_orig, 2) if n_orig else 0,
            'Prob_total' : round(100 * cnt / tot_pairs, 3) if tot_pairs else 0,
        })
save_table(pd.DataFrame(summary), 'D8_transicoes_long', index=False)


# ─────────────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('Done.')
print(f'  Sample: {n_sampled:,} individuals ({100*n_sampled/n_total:.1f} % of {n_total:,})')
print(f'  Seed  : {SEED}')
print(f'  Tables → {T_DIR}')
print(f'  Figures → {F_DIR}')
