import warnings
import textwrap

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.path import Path

from .config import cleaned_path, ensure_output_dirs, preferred_raw_path
from .io_utils import downsample_individuals


matplotlib.use("Agg")
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

PANEL_ID = ["id_domicilio", "num_ordem"]
STATE_COLORS = {
    "Fora da PEA": "#888888",
    "Desocupado": "#E05C5C",
    "Conta-propria": "#F4A623",
    "Setor Privado": "#2B7BB9",
    "Setor Publico": "#3AAA5C",
}
STATES = list(STATE_COLORS.keys())
STATE_LABELS_EN = {
    "Fora da PEA": "Out of labor force",
    "Desocupado": "Unemployed",
    "Conta-propria": "Self-employed",
    "Setor Privado": "Private sector",
    "Setor Publico": "Public sector",
}
STATE_LABELS_PT = {value: key for key, value in STATE_LABELS_EN.items()}
PANEL_COLS = ["id_domicilio", "num_ordem"]

TEXT_REPLACEMENTS_EN = {
    "Estado em t": "State at t",
    "Estado em t+1": "State at t+1",
    "% de transicoes": "% of transitions",
    "Probabilidades de transicao entre estados do mercado de trabalho": "Transition probabilities across labor-market states",
    "PNADC": "PNADC",
    "todos os trimestres": "all quarters",
    "Probabilidades de transicao por trimestre": "Transition probabilities by quarter",
    "Fluxo entre estados do mercado de trabalho": "Flow across labor-market states",
    "por trimestre": "by quarter",
    "Sem ponderacao": "Unweighted",
    "Com ponderacao amostral": "Weighted",
    "Distribuicao de destinos por estado de origem": "Destination distribution by origin state",
    "% do estado de origem": "% within origin state",
    "Destino (t+1)": "Destination (t+1)",
    "Mil observacoes": "Thousands of observations",
    "Frequencia absoluta de transicoes por trimestre": "Absolute transition frequency by quarter",
    "Trajetorias individuais no mercado de trabalho": "Individual labor-market trajectories",
    "individuos com >=1 par consecutivo": "individuals with >=1 consecutive pair",
    "Composicao do mercado de trabalho por trimestre": "Labor-market composition by quarter",
    "% da amostra": "% of sample",
    "Destino": "Destination",
}
TABLE_COLS_EN = {
    "Transicao": "Transition",
    "Log renda t media": "Mean log earnings at t",
    "Log renda t+1 media": "Mean log earnings at t+1",
    "Var log mediana": "Median log change",
    "Var log media": "Mean log change",
    "Mediana t": "Median at t",
    "Mediana t+1": "Median at t+1",
    "Variacao R$ mediana": "Median R$ change",
    "Variacao % mediana": "Median % change",
    "Ganhou %": "Gained %",
    "Perdeu %": "Lost %",
    "Igual %": "No change %",
    "Trimestre": "Quarter",
    "Estado": "State",
}


def n_unique_people(df):
    if df is None or df.empty:
        return 0
    if all(col in df.columns for col in PANEL_COLS):
        return len(df[PANEL_COLS].drop_duplicates())
    return 0


def output_note(full_df, used_df):
    return (
        f"Note: full dataset = {len(full_df):,} observations; "
        f"full panel = {n_unique_people(full_df):,} unique individuals; "
        f"this output = {n_unique_people(used_df):,} unique individuals."
    )


def output_note_pt(full_df, used_df):
    return (
        f"Nota: base completa = {len(full_df):,} observacoes; "
        f"painel completo = {n_unique_people(full_df):,} individuos unicos; "
        f"esta tabela/figura = {n_unique_people(used_df):,} individuos unicos."
    )


def wrap_label(text, width=22):
    return textwrap.fill(str(text), width=width, break_long_words=False, break_on_hyphens=False)


def translate_text_en(text):
    out = str(text)
    for pt, en in sorted(TEXT_REPLACEMENTS_EN.items(), key=lambda item: len(item[0]), reverse=True):
        out = out.replace(pt, en)
    for pt, en in sorted(STATE_LABELS_EN.items(), key=lambda item: len(item[0]), reverse=True):
        out = out.replace(pt, wrap_label(en))
    return out


def translate_text_pt(text):
    out = str(text)
    for pt, en in sorted(TEXT_REPLACEMENTS_EN.items(), key=lambda item: len(item[1]), reverse=True):
        out = out.replace(en, pt)
    for en, pt in sorted(STATE_LABELS_PT.items(), key=lambda item: len(item[0]), reverse=True):
        out = out.replace(wrap_label(en), wrap_label(pt))
        out = out.replace(en, pt)
    return out


def localize_table_en(df):
    out = df.copy()
    out.columns = [TABLE_COLS_EN.get(str(col), str(col)) for col in out.columns]
    out.index = [translate_text_en(idx) for idx in out.index]
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(lambda value: translate_text_en(value) if pd.notna(value) else value)
    return out


def localize_figure_text(fig, to_lang="en"):
    translate = translate_text_en if to_lang == "en" else translate_text_pt
    for ax in fig.axes:
        ax.set_title(translate(ax.get_title()))
        ax.set_xlabel(translate(ax.get_xlabel()))
        ax.set_ylabel(translate(ax.get_ylabel()))
        ax.set_xticklabels([translate(t.get_text()) for t in ax.get_xticklabels()])
        ax.set_yticklabels([translate(t.get_text()) for t in ax.get_yticklabels()])
        for txt in ax.texts:
            txt.set_text(translate(txt.get_text()))
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(translate(legend.get_title().get_text()))
            for txt in legend.get_texts():
                txt.set_text(translate(txt.get_text()))
    for txt in fig.texts:
        txt.set_text(translate(txt.get_text()))


def save_table(df, name, tables_dir, full_df, used_df, index=True, pt_df=None):
    df_en = localize_table_en(df)
    path = tables_dir / f"{name}.tex"
    latex = df_en.to_latex(index=index, float_format=lambda x: f"{x:,.2f}", na_rep="--", escape=True)
    latex = latex.rstrip() + f"\n\n\\begin{{flushleft}}\\footnotesize {output_note(full_df, used_df)}\\end{{flushleft}}\n"
    path.write_text(latex, encoding="utf-8")
    pt_df = pt_df if pt_df is not None else df
    pt_latex = pt_df.to_latex(index=index, float_format=lambda x: f"{x:,.2f}", na_rep="--", escape=True)
    pt_latex = pt_latex.rstrip() + f"\n\n\\begin{{flushleft}}\\footnotesize {output_note_pt(full_df, used_df)}\\end{{flushleft}}\n"
    (tables_dir / f"{name}_pt.tex").write_text(pt_latex, encoding="utf-8")


def save_figure(name, figures_dir, full_df, used_df, extra_note=None, dpi=150):
    fig = plt.gcf()
    localize_figure_text(fig, to_lang="en")
    note_lines = [output_note(full_df, used_df)]
    if extra_note:
        note_lines.extend(textwrap.wrap(extra_note, width=125))
    fig.subplots_adjust(bottom=0.22)
    y = 0.02
    for line in note_lines[:5]:
        fig.text(0.01, y, line, ha="left", va="bottom", fontsize=7)
        y += 0.022
    path = figures_dir / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    localize_figure_text(fig, to_lang="pt")
    pt_lines = [output_note_pt(full_df, used_df)]
    if extra_note:
        pt_lines.extend(textwrap.wrap(f"Nota: {extra_note}", width=125))
    for idx, txt in enumerate(fig.texts[-len(note_lines[:5]):]):
        if idx < len(pt_lines[:5]):
            txt.set_text(pt_lines[idx])
    plt.savefig(figures_dir / f"{name}_pt.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def fmt_brl(x, _=None):
    return f"R${x:,.0f}"


def load_transition_frame(year, use_sample_inputs=False):
    cleaned = cleaned_path("microdados", year, use_sample=use_sample_inputs)
    if cleaned.exists():
        keep_cols = [
            "ano",
            "trimestre",
            "id_domicilio",
            "num_ordem",
            "peso",
            "idade",
            "condicao_forca_trabalho",
            "condicao_ocupacao",
            "posicao_emprego",
            "posicao_emprego_label",
            "renda_habitual_principal",
            "renda_habitual_todos",
            "tempo_nesse_trabalho_label",
            "contribui_previdencia_label",
            "gostaria_ter_trabalhado_label",
            "motivo_nao_buscou_label",
            "motivo_nao_trabalhou_label",
            "motivo_fora_forca_trabalho2_label",
        ]
        available = pd.read_csv(cleaned, nrows=0).columns.tolist()
        use_cols = [col for col in keep_cols if col in available]
        return pd.read_csv(cleaned, usecols=use_cols, low_memory=False)

    raw = preferred_raw_path("microdados", year=year, use_sample=use_sample_inputs)
    raw_cols = ["ano", "trimestre", "id_domicilio", "V2003", "V1028", "VD4001", "VD4002", "VD4009", "VD4016", "VD4019"]
    available = pd.read_csv(raw, nrows=0).columns.tolist()
    use_cols = [col for col in raw_cols if col in available]
    df = pd.read_csv(raw, usecols=use_cols, low_memory=False)
    return df.rename(
        columns={
            "V2003": "num_ordem",
            "V1028": "peso",
            "VD4001": "condicao_forca_trabalho",
            "VD4002": "condicao_ocupacao",
            "VD4009": "posicao_emprego",
            "VD4016": "renda_habitual_principal",
            "VD4019": "renda_habitual_todos",
        }
    )


def load_transition_frames(years, use_sample_inputs=False):
    frames = [load_transition_frame(y, use_sample_inputs=use_sample_inputs) for y in sorted(years)]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def normalize_years(years=None, year=None):
    if years is None:
        if year is None:
            raise ValueError("Specify year or years.")
        years = [year]
    elif isinstance(years, int):
        years = [years]
    years_sorted = sorted({int(y) for y in years})
    if not years_sorted:
        raise ValueError("Specify at least one year.")
    return years_sorted


def build_run_tag(years, sample_frac=None, tag=None):
    if tag is not None:
        return tag
    year_str = "_".join(str(y) for y in years)
    qual = "sample" if sample_frac and sample_frac < 1 else "full"
    return f"{year_str}_{qual}"


def build_year_label(years):
    return "/".join(str(y) for y in years)


def quarter_index(years, quarters):
    return pd.to_numeric(years, errors="coerce") * 4 + pd.to_numeric(quarters, errors="coerce")


def add_period_labels(df, include_year=False):
    out = df.copy()
    year_num = pd.to_numeric(out["ano"], errors="coerce").astype("Int64")
    quarter_num = pd.to_numeric(out["trimestre"], errors="coerce").astype("Int64")
    if include_year:
        out["period_label"] = year_num.astype(str) + "Q" + quarter_num.astype(str)
    else:
        out["period_label"] = "Q" + quarter_num.astype(str)
    out["period_order"] = quarter_index(out["ano"], out["trimestre"])
    return out


def make_panel_grid(n_panels, max_cols=3, width=6.0, height=5.0, sharey=False):
    ncols = min(max_cols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), sharey=sharey)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_panels:]:
        ax.axis("off")
    return fig, axes


def classify_states(df):
    for column in [
        "ano",
        "trimestre",
        "num_ordem",
        "peso",
        "condicao_forca_trabalho",
        "condicao_ocupacao",
        "posicao_emprego",
        "renda_habitual_principal",
        "renda_habitual_todos",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    forca = df["condicao_forca_trabalho"]
    ocup = df["condicao_ocupacao"]
    pos = df["posicao_emprego"]
    conditions = [
        forca == 2,
        (forca == 1) & (ocup == 2),
        (forca == 1) & (ocup == 1) & (pos == 9),
        (forca == 1) & (ocup == 1) & pos.isin([1, 2, 3, 4, 8, 10]),
        (forca == 1) & (ocup == 1) & pos.isin([5, 6, 7]),
    ]
    choices = ["Fora da PEA", "Desocupado", "Conta-propria", "Setor Privado", "Setor Publico"]
    df["estado_lm"] = np.select(conditions, choices, default=None)
    return df


def build_transition_matrix(df_trans, weight_col=None):
    if weight_col and weight_col in df_trans.columns:
        weights = pd.to_numeric(df_trans[weight_col], errors="coerce").fillna(1)
        counts = df_trans.assign(_w=weights).groupby(["estado_lm", "estado_t1"])["_w"].sum().unstack(fill_value=0)
    else:
        counts = df_trans.groupby(["estado_lm", "estado_t1"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=STATES, columns=STATES, fill_value=0)
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return counts, probs


def income_column(df):
    return next(
        (
            column
            for column in ["renda_habitual_principal", "renda_habitual_todos"]
            if column in df.columns and df[column].notna().any()
        ),
        None,
    )


def _bezier_flow(ax, x1, y1_top, y1_bot, x2, y2_top, y2_bot, color, alpha=0.4):
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
    patch = mpatches.PathPatch(Path(verts, codes), facecolor=mcolors.to_rgba(color, alpha), edgecolor="none", zorder=1)
    ax.add_patch(patch)


def draw_sankey(count_matrix, title="", ax=None, bar_width=0.06, gap=0.025):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    scale = count_matrix.values.sum()
    if scale == 0:
        return

    row_totals = count_matrix.sum(axis=1)
    col_totals = count_matrix.sum(axis=0)
    left_x = 0.1
    right_x = 0.85

    def bar_positions(totals):
        info = {}
        y = 1.0
        for state in STATES:
            h = totals[state] / scale
            info[state] = (y - h, y)
            y -= h + gap
        return info

    left_bars = bar_positions(row_totals)
    right_bars = bar_positions(col_totals)

    for side, xpos, bars in [("left", left_x, left_bars), ("right", right_x, right_bars)]:
        for state in STATES:
            bottom, top = bars[state]
            rect = mpatches.FancyBboxPatch(
                (xpos, bottom),
                bar_width,
                top - bottom,
                boxstyle="square,pad=0",
                facecolor=STATE_COLORS[state],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.add_patch(rect)
            mid = (bottom + top) / 2
            pct = (top - bottom) * 100
            if side == "left":
                ax.text(xpos - 0.01, mid, f"{state}\n({pct:.0f}%)", ha="right", va="center", fontsize=7.5, color=STATE_COLORS[state], fontweight="bold")
            else:
                ax.text(xpos + bar_width + 0.01, mid, f"{state}\n({pct:.0f}%)", ha="left", va="center", fontsize=7.5, color=STATE_COLORS[state], fontweight="bold")

    left_cursor = {state: left_bars[state][1] for state in STATES}
    right_cursor = {state: right_bars[state][1] for state in STATES}

    for orig in STATES:
        for dest in STATES:
            cnt = count_matrix.loc[orig, dest]
            if cnt == 0:
                continue
            h = cnt / scale
            y1_top = left_cursor[orig]
            y1_bot = y1_top - h
            y2_top = right_cursor[dest]
            y2_bot = y2_top - h
            left_cursor[orig] -= h
            right_cursor[dest] -= h
            _bezier_flow(ax, left_x + bar_width, y1_top, y1_bot, right_x, y2_top, y2_bot, STATE_COLORS[orig])

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.08, 1.1)
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=12)


def build_flow(col_a, col_b, df_w):
    sub = df_w[[col_a, col_b]].dropna()
    return sub.groupby([col_a, col_b]).size().reset_index(name="n")


def alluvial_plot(traj_df, flows_list, quarter_names, title=""):
    n_q = len(quarter_names)
    fig, ax = plt.subplots(figsize=(4.5 * n_q, 8))
    bar_w = 0.08
    gap = 0.025
    x_pos = {q: i for i, q in enumerate(quarter_names)}

    def compute_bars(q):
        counts = traj_df[q].value_counts(dropna=True)
        total = counts.sum()
        y, info = 1.0, {}
        for s in STATES:
            h = counts.get(s, 0) / total if total > 0 else 0
            info[s] = (y - h, y)
            y -= h + gap
        return info

    bar_info = {q: compute_bars(q) for q in quarter_names}

    for q in quarter_names:
        xi = x_pos[q]
        for s in STATES:
            bottom, top = bar_info[q][s]
            if top - bottom < 0.001:
                continue
            rect = mpatches.FancyBboxPatch((xi - bar_w / 2, bottom), bar_w, top - bottom, boxstyle="square,pad=0", facecolor=STATE_COLORS[s], edgecolor="white", linewidth=0.5, zorder=3)
            ax.add_patch(rect)
            if top - bottom > 0.025:
                ax.text(xi, (bottom + top) / 2, f"{(top-bottom)*100:.0f}%", ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=4)
        ax.text(xi, 1.05, q, ha="center", va="bottom", fontsize=12, fontweight="bold")

    for (q_from, q_to), flows in zip(zip(quarter_names[:-1], quarter_names[1:]), flows_list):
        xi_from = x_pos[q_from] + bar_w / 2
        xi_to = x_pos[q_to] - bar_w / 2
        total = traj_df[q_from].count()
        left_cursor = {s: bar_info[q_from][s][1] for s in STATES}
        right_cursor = {s: bar_info[q_to][s][1] for s in STATES}
        for _, row in flows.iterrows():
            orig, dest, cnt = row[q_from], row[q_to], row["n"]
            if orig not in STATES or dest not in STATES or cnt == 0:
                continue
            h = cnt / total
            y1_top = left_cursor[orig]
            y1_bot = y1_top - h
            y2_top = right_cursor[dest]
            y2_bot = y2_top - h
            left_cursor[orig] -= h
            right_cursor[dest] -= h
            _bezier_flow(ax, xi_from, y1_top, y1_bot, xi_to, y2_top, y2_bot, STATE_COLORS[orig], alpha=0.4)

    handles = [mpatches.Patch(facecolor=STATE_COLORS[s], label=s) for s in STATES]
    ax.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06), fontsize=9, frameon=True)
    ax.set_xlim(-0.3, n_q - 0.7)
    ax.set_ylim(-0.12, 1.12)
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=15)
    return fig


def append_log_versions(wage_table, inc_col, wage_df):
    log_rows = []
    trans_cols = [c for c in ["estado_lm", "estado_t1"] if c in wage_df.columns]
    if len(trans_cols) < 2:
        return None
    wage_df = wage_df.copy()
    wage_df["log_t"] = np.log(wage_df[inc_col])
    wage_df["log_t1"] = np.log(wage_df[f"{inc_col}_t1"])
    wage_df["delta_log"] = wage_df["log_t1"] - wage_df["log_t"]
    for trans in wage_table.index:
        parts = trans.split("->")
        if len(parts) != 2:
            continue
        orig, dest = parts
        mapping = {"CP": "Conta-propria", "Publico": "Setor Publico", "Privado": "Setor Privado", "Desocupado": "Desocupado"}
        orig_full = mapping.get(orig, orig)
        dest_full = mapping.get(dest, dest)
        sub = wage_df[(wage_df["estado_lm"] == orig_full) & (wage_df["estado_t1"] == dest_full)]
        if len(sub) < 5:
            continue
        log_rows.append(
            {
                "Transicao": trans,
                "Log renda t media": round(sub["log_t"].mean(), 3),
                "Log renda t+1 media": round(sub["log_t1"].mean(), 3),
                "Var log mediana": round(sub["delta_log"].median(), 3),
                "Var log media": round(sub["delta_log"].mean(), 3),
            }
        )
    if not log_rows:
        return None
    return pd.DataFrame(log_rows).set_index("Transicao")


def public_exit_analysis_frame(df, years=None, year=None):
    years_sorted = normalize_years(years=years, year=year)
    work = df.sort_values(PANEL_ID + ["ano", "trimestre"]).reset_index(drop=True).copy()
    group = work.groupby(PANEL_ID)

    carry_cols = [
        "idade",
        "estado_lm",
        "posicao_emprego_label",
        "tempo_nesse_trabalho_label",
        "contribui_previdencia_label",
        "gostaria_ter_trabalhado_label",
        "motivo_nao_buscou_label",
        "motivo_nao_trabalhou_label",
        "motivo_fora_forca_trabalho2_label",
    ]
    for column in carry_cols:
        if column in work.columns:
            work[f"{column}_t1"] = group[column].shift(-1)
    work["ano_t1"] = group["ano"].shift(-1)
    work["trimestre_t1"] = group["trimestre"].shift(-1)
    work["period_order"] = quarter_index(work["ano"], work["trimestre"])
    work["period_order_t1"] = quarter_index(work["ano_t1"], work["trimestre_t1"])

    trans = work[
        work["ano"].isin(years_sorted)
        & work["ano_t1"].isin(years_sorted)
        & ((work["period_order_t1"] - work["period_order"]) == 1)
        & work["estado_lm"].notna()
        & work["estado_lm_t1"].notna()
    ].copy()
    public = trans[trans["estado_lm"] == "Setor Publico"].copy()
    outlf = public[public["estado_lm_t1"] == "Fora da PEA"].copy()
    private = public[public["estado_lm_t1"] == "Setor Privado"].copy()

    motivo_fora = outlf.get("motivo_fora_forca_trabalho2_label_t1", pd.Series(index=outlf.index, dtype=object))
    motivo_nao = outlf.get("motivo_nao_trabalhou_label_t1", pd.Series(index=outlf.index, dtype=object))
    reason = motivo_fora.fillna(motivo_nao)
    outlf["exit_reason_pt"] = reason
    outlf["exit_reason_en"] = outlf["exit_reason_pt"].map(translate_text_en)
    outlf["retirement_proxy"] = (
        motivo_fora.astype(str).str.contains("idos", case=False, na=False)
        | motivo_nao.astype(str).str.contains("idos", case=False, na=False)
    )
    return trans, public, outlf, private


def run_public_exit_analysis(years=None, year=None, sample_frac=None, seed=42, tag=None, use_sample_inputs=False):
    years_sorted = normalize_years(years=years, year=year)
    year_label = build_year_label(years_sorted)
    tag = build_run_tag(years_sorted, sample_frac=sample_frac, tag=tag)
    tables_dir, figures_dir = ensure_output_dirs(tag=f"public_exits_{tag}")

    df = load_transition_frames(years_sorted, use_sample_inputs=use_sample_inputs)
    df = downsample_individuals(df, sample_frac, seed, panel_cols=PANEL_ID)
    df = classify_states(df)

    transitions, public, outlf, private = public_exit_analysis_frame(df, years=years_sorted)

    summary = pd.DataFrame(
        [
            {"flow": "Public -> all next-quarter transitions", "n": len(public), "share_pct": 100.0 if len(public) else np.nan},
            {"flow": "Public -> out of labor force", "n": len(outlf), "share_pct": round(len(outlf) / len(public) * 100, 1) if len(public) else np.nan},
            {"flow": "Public -> private sector", "n": len(private), "share_pct": round(len(private) / len(public) * 100, 1) if len(public) else np.nan},
            {"flow": "Out-of-labor-force exits flagged as likely retirement", "n": int(outlf["retirement_proxy"].sum()), "share_pct": round(outlf["retirement_proxy"].mean() * 100, 1) if len(outlf) else np.nan},
        ]
    )
    save_table(summary, "E1_resumo_saidas_setor_publico", tables_dir, df, outlf, index=False)

    if not outlf.empty:
        age = pd.cut(
            outlf["idade_t1"],
            bins=[0, 29, 39, 49, 59, 69, 200],
            labels=["<=29", "30s", "40s", "50s", "60s", "70+"],
        ).value_counts(normalize=True).sort_index().mul(100).rename_axis("age_band").reset_index(name="pct")
        age_pt = age.rename(columns={"age_band": "faixa_etaria", "pct": "pct"})
        save_table(age, "E2_idade_saida_publico_fora_pea", tables_dir, df, outlf, index=False, pt_df=age_pt)

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(age["age_band"], age["pct"], color=STATE_COLORS["Setor Publico"])
        ax.set_xlabel("Age band")
        ax.set_ylabel("Percent of public -> out of labor force")
        ax.set_title(f"Age profile of public-sector exits to out of labor force\nPNADC {year_label}")
        plt.tight_layout()
        save_figure("E1_idade_saida_publico_fora_pea", figures_dir, df, outlf)

        reasons = outlf["exit_reason_en"].value_counts(normalize=True).mul(100).head(8).rename_axis("reason_en").reset_index(name="pct")
        reasons_pt = outlf["exit_reason_pt"].value_counts(normalize=True).mul(100).head(8).rename_axis("motivo_pt").reset_index(name="pct")
        save_table(reasons, "E3_motivos_saida_publico_fora_pea", tables_dir, df, outlf, index=False, pt_df=reasons_pt)

        fig, ax = plt.subplots(figsize=(10, 5.8))
        ax.barh([wrap_label(x, 28) for x in reasons["reason_en"][::-1]], reasons["pct"][::-1], color=STATE_COLORS["Fora da PEA"])
        ax.set_xlabel("Percent of exits")
        ax.set_title(f"Why public-sector workers leave to out of labor force\nPNADC {year_label}")
        plt.tight_layout()
        save_figure(
            "E2_motivos_saida_publico_fora_pea",
            figures_dir,
            df,
            outlf,
            extra_note="Portuguese originals are listed in the companion table E3_motivos_saida_publico_fora_pea_pt.",
        )

    return {"years": years_sorted, "rows": len(df), "public_transitions": len(public), "public_to_outlf": len(outlf), "tag": tag}


def run_transitions(years=None, year=None, sample_frac=None, seed=42, tag=None, use_sample_inputs=False):
    years_sorted = normalize_years(years=years, year=year)
    multi_year = len(years_sorted) > 1
    year_label = build_year_label(years_sorted)
    tag = build_run_tag(years_sorted, sample_frac=sample_frac, tag=tag)
    tables_dir, figures_dir = ensure_output_dirs(tag=f"transitions_{tag}")

    df = load_transition_frames(years_sorted, use_sample_inputs=use_sample_inputs)
    df = downsample_individuals(df, sample_frac, seed, panel_cols=PANEL_ID)
    df = classify_states(df)
    df = add_period_labels(df, include_year=multi_year)
    df = df.sort_values(PANEL_ID + ["ano", "trimestre"]).reset_index(drop=True)

    inc_col = income_column(df)
    group = df.groupby(PANEL_ID)
    df["estado_t1"] = group["estado_lm"].shift(-1)
    df["trimestre_t1"] = group["trimestre"].shift(-1)
    df["ano_t1"] = group["ano"].shift(-1)
    df["period_label_t1"] = group["period_label"].shift(-1)
    df["period_order_t1"] = group["period_order"].shift(-1)
    if inc_col:
        df[f"{inc_col}_t1"] = group[inc_col].shift(-1)

    selected_rows = df[df["ano"].isin(years_sorted)].copy()
    eligible_next_obs = df[df["ano_t1"].isin(years_sorted)].copy()
    consecutive_next_obs = df[
        df["ano_t1"].isin(years_sorted)
        & ((df["period_order_t1"] - df["period_order"]) == 1)
    ].copy()

    transitions = df[
        df["ano_t1"].isin(years_sorted)
        & ((df["period_order_t1"] - df["period_order"]) == 1)
        & df["estado_lm"].notna()
        & df["estado_t1"].notna()
    ].copy()

    cnt_all, prb_all = build_transition_matrix(transitions)
    cnt_wt, prb_wt = build_transition_matrix(transitions, weight_col="peso")

    save_table(cnt_all.astype(int), "D1_transicoes_contagem_anual", tables_dir, df, transitions)
    save_table(prb_all.mul(100).round(2), "D2_transicoes_probabilidade_anual", tables_dir, df, transitions)
    save_table(cnt_wt.round(0).astype(int), "D3_transicoes_pesado_anual", tables_dir, df, transitions)
    save_table(prb_wt.mul(100).round(2), "D4_transicoes_probabilidade_pesado_anual", tables_dir, df, transitions)

    period_pairs = (
        transitions[["period_order", "period_label", "period_order_t1", "period_label_t1"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["period_order", "period_order_t1"])
        .to_dict("records")
    )
    quarter_matrices = []
    for spec in period_pairs:
        sub = transitions[
            (transitions["period_label"] == spec["period_label"])
            & (transitions["period_label_t1"] == spec["period_label_t1"])
        ].copy()
        cnt_q, prb_q = build_transition_matrix(sub)
        quarter_matrices.append({**spec, "count": cnt_q, "prob": prb_q, "data": sub})
        pair_name = f"D5_transicoes_prob_{spec['period_label']}_{spec['period_label_t1']}".replace("/", "_")
        save_table(prb_q.mul(100).round(2), pair_name, tables_dir, df, sub)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(prb_all.mul(100), annot=True, fmt=".1f", cmap="Blues", linewidths=0.5, linecolor="grey", annot_kws={"size": 9}, ax=ax, cbar_kws={"label": "% de transicoes"})
    ax.set_xlabel("Estado em t+1", fontsize=11)
    ax.set_ylabel("Estado em t", fontsize=11)
    ax.set_title(f"Probabilidades de transicao entre estados do mercado de trabalho\nPNADC {year_label} - todos os pares consecutivos", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    save_figure("D1_heatmap_transicoes_anual", figures_dir, df, transitions)

    if quarter_matrices:
        fig, axes = make_panel_grid(len(quarter_matrices), max_cols=3, width=6.0, height=5.5, sharey=True)
        for i, spec in enumerate(quarter_matrices):
            sns.heatmap(
                spec["prob"].mul(100),
                annot=True,
                fmt=".1f",
                cmap="Blues",
                linewidths=0.5,
                linecolor="grey",
                annot_kws={"size": 8},
                ax=axes[i],
                vmin=0,
                vmax=100,
                cbar=(i == len(quarter_matrices) - 1),
                cbar_kws={"label": "%"} if i == len(quarter_matrices) - 1 else {},
            )
            axes[i].set_title(f"{spec['period_label']} -> {spec['period_label_t1']}", fontsize=12)
            axes[i].set_xlabel("Estado em t+1", fontsize=9)
            if i % min(3, len(quarter_matrices)) == 0:
                axes[i].set_ylabel("Estado em t", fontsize=9)
            axes[i].tick_params(axis="x", rotation=30)
            axes[i].tick_params(axis="y", rotation=0)
        fig.suptitle(f"Probabilidades de transicao por par consecutivo (PNADC {year_label})", fontsize=13)
        plt.tight_layout()
        save_figure("D2_heatmap_transicoes_por_trimestre", figures_dir, df, transitions)

    fig, ax = plt.subplots(figsize=(11, 8))
    draw_sankey(cnt_all, title=f"Fluxo entre estados do mercado de trabalho\nPNADC {year_label} - todos os pares consecutivos", ax=ax)
    plt.tight_layout()
    save_figure("D3_sankey_anual", figures_dir, df, transitions)

    if quarter_matrices:
        fig, axes = make_panel_grid(len(quarter_matrices), max_cols=2, width=10.0, height=8.0)
        for ax, spec in zip(axes, quarter_matrices):
            draw_sankey(spec["count"], title=f"{spec['period_label']} -> {spec['period_label_t1']}", ax=ax)
        fig.suptitle(f"Fluxo entre estados do mercado de trabalho por par consecutivo (PNADC {year_label})", fontsize=13, y=1.01)
        plt.tight_layout()
        save_figure("D4_sankey_por_trimestre", figures_dir, df, transitions)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (mat, lbl) in zip(axes, [(prb_all, "Sem ponderacao"), (prb_wt, "Com ponderacao amostral")]):
        bottom = np.zeros(len(STATES))
        x = np.arange(len(STATES))
        for dest in STATES:
            vals = mat[dest].values * 100
            ax.bar(x, vals, bottom=bottom, color=STATE_COLORS[dest], label=dest, width=0.65)
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels(STATES, rotation=28, ha="right", fontsize=8.5)
        ax.set_ylabel("% do estado de origem")
        ax.set_ylim(0, 108)
        ax.set_title(f"Distribuicao de destinos por estado de origem\n{lbl}")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(title="Destino (t+1)", fontsize=8, loc="upper right", bbox_to_anchor=(1.4, 1.0))
    plt.tight_layout()
    save_figure("D5_barra_empilhada_destinos", figures_dir, df, transitions)

    if quarter_matrices:
        fig, axes = make_panel_grid(len(quarter_matrices), max_cols=3, width=6.0, height=4.5)
        for i, spec in enumerate(quarter_matrices):
            bottom = np.zeros(len(STATES))
            x = np.arange(len(STATES))
            for dest in STATES:
                vals = spec["count"][dest].values.astype(float)
                axes[i].bar(x, vals / 1000, bottom=bottom / 1000, color=STATE_COLORS[dest], label=dest, width=0.65)
                bottom += vals
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(STATES, rotation=28, ha="right", fontsize=8)
            axes[i].set_ylabel("Mil observacoes")
            axes[i].set_title(f"{spec['period_label']} -> {spec['period_label_t1']}")
            if i == len(quarter_matrices) - 1:
                axes[i].legend(title="Destino", fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.0))
        fig.suptitle(f"Frequencia absoluta de transicoes por par consecutivo (PNADC {year_label})", fontsize=12)
        plt.tight_layout()
        save_figure("D6_frequencia_transicoes_trimestre", figures_dir, df, transitions)

    panel_selected = df[df["estado_lm"].notna()].copy()
    period_names = (
        panel_selected[["period_order", "period_label"]]
        .drop_duplicates()
        .sort_values("period_order")["period_label"]
        .tolist()
    )
    traj_wide = panel_selected.pivot_table(index=PANEL_ID, columns="period_label", values="estado_lm", aggfunc="first")
    traj_wide = traj_wide.reindex(columns=period_names)
    obs_ids = panel_selected[PANEL_ID + ["period_order"]].drop_duplicates()
    consec_check = obs_ids.merge(obs_ids.rename(columns={"period_order": "period_order_nxt"}), on=PANEL_ID)
    consec_check = consec_check[consec_check["period_order_nxt"] == consec_check["period_order"] + 1]
    keep_ids = consec_check[PANEL_ID].drop_duplicates().set_index(PANEL_ID).index
    traj_panel = traj_wide[traj_wide.index.isin(keep_ids)]

    if len(period_names) > 1 and not traj_panel.empty:
        flows = [build_flow(period_names[i], period_names[i + 1], traj_panel) for i in range(len(period_names) - 1)]
        fig = alluvial_plot(
            traj_panel,
            flows,
            period_names,
            title=f"Trajetorias individuais no mercado de trabalho\nPNADC {year_label} - individuos com >=1 par consecutivo",
        )
        save_figure("D7_alluvial_trajetorias_individuais", figures_dir, df, traj_panel.reset_index())

    qtr_shares = []
    for period_order, period_label in (
        panel_selected[["period_order", "period_label"]]
        .drop_duplicates()
        .sort_values("period_order")
        .itertuples(index=False, name=None)
    ):
        sub = panel_selected[panel_selected["period_label"] == period_label]["estado_lm"]
        tot = len(sub)
        for s in STATES:
            qtr_shares.append({"Trimestre": period_label, "Estado": s, "n": (sub == s).sum(), "pct": 100 * (sub == s).sum() / tot if tot else 0})
    qtr_df = pd.DataFrame(qtr_shares)
    fig_width = max(8, 1.5 * max(1, len(period_names)))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    x = np.arange(len(period_names))
    bottoms = np.zeros(len(period_names))
    for s in STATES:
        vals = [qtr_df[(qtr_df.Trimestre == q) & (qtr_df.Estado == s)]["pct"].values[0] for q in period_names]
        ax.bar(x, vals, bottom=bottoms, color=STATE_COLORS[s], label=s, width=0.6)
        bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(period_names, rotation=35 if multi_year else 0, ha="right" if multi_year else "center")
    ax.set_ylabel("% da amostra")
    ax.set_title(f"Composicao do mercado de trabalho por trimestre\nPNADC {year_label}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    save_figure("D8_composicao_por_trimestre", figures_dir, df, panel_selected)
    comp_wide = qtr_df.pivot_table(index="Estado", columns="Trimestre", values="pct").reindex(STATES)
    comp_wide = comp_wide.reindex(columns=period_names)
    comp_wide.columns.name = None
    save_table(comp_wide.round(1), "D6_composicao_estados_por_trimestre", tables_dir, df, panel_selected)

    if inc_col and f"{inc_col}_t1" in transitions.columns:
        wage_df = transitions[
            transitions[inc_col].notna()
            & transitions[f"{inc_col}_t1"].notna()
            & (transitions[inc_col] > 0)
            & (transitions[f"{inc_col}_t1"] > 0)
        ].copy()
        wage_df["delta_R"] = wage_df[f"{inc_col}_t1"] - wage_df[inc_col]
        wage_df["delta_pct"] = 100 * wage_df["delta_R"] / wage_df[inc_col]
        wage_df["ganhou"] = (wage_df["delta_R"] > 0).astype(int)
        wage_df["perdeu"] = (wage_df["delta_R"] < 0).astype(int)
        wage_df["igual"] = (wage_df["delta_R"] == 0).astype(int)

        core = [
            ("Setor Privado", "Setor Privado", "Privado->Privado"),
            ("Setor Privado", "Setor Publico", "Privado->Publico"),
            ("Setor Publico", "Setor Privado", "Publico->Privado"),
            ("Setor Publico", "Setor Publico", "Publico->Publico"),
            ("Conta-propria", "Setor Privado", "CP->Privado"),
            ("Conta-propria", "Setor Publico", "CP->Publico"),
            ("Setor Privado", "Conta-propria", "Privado->CP"),
            ("Desocupado", "Setor Privado", "Desocupado->Privado"),
            ("Desocupado", "Setor Publico", "Desocupado->Publico"),
        ]
        wage_rows = []
        for orig, dest, lbl in core:
            sub = wage_df[(wage_df["estado_lm"] == orig) & (wage_df["estado_t1"] == dest)]
            if len(sub) < 5:
                continue
            wage_rows.append(
                {
                    "Transicao": lbl,
                    "N": len(sub),
                    "Renda t media": round(sub[inc_col].mean(), 0),
                    "Renda t+1 media": round(sub[f"{inc_col}_t1"].mean(), 0),
                    "Renda t mediana": round(sub[inc_col].median(), 0),
                    "Renda t+1 mediana": round(sub[f"{inc_col}_t1"].median(), 0),
                    "Var RS media": round(sub["delta_R"].mean(), 0),
                    "Var RS mediana": round(sub["delta_R"].median(), 0),
                    "Var pct media": round(sub["delta_pct"].mean(), 1),
                    "Ganhou pct": round(sub["ganhou"].mean() * 100, 1),
                    "Perdeu pct": round(sub["perdeu"].mean() * 100, 1),
                    "Igual pct": round(sub["igual"].mean() * 100, 1),
                }
            )

        if wage_rows:
            wage_tbl = pd.DataFrame(wage_rows).set_index("Transicao")
            save_table(wage_tbl, "D7_ganho_salarial_transicoes", tables_dir, df, wage_df)

            log_tbl = append_log_versions(wage_tbl, inc_col, wage_df)
            if log_tbl is not None:
                save_table(log_tbl, "D7_ganho_salarial_transicoes_log", tables_dir, df, wage_df)

            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(wage_tbl))
            w = 0.25
            ax.bar(x - w, wage_tbl["Ganhou pct"].values, width=w, color=STATE_COLORS["Setor Publico"], label="Ganhou")
            ax.bar(x, wage_tbl["Perdeu pct"].values, width=w, color=STATE_COLORS["Desocupado"], label="Perdeu")
            ax.bar(x + w, wage_tbl["Igual pct"].values, width=w, color=STATE_COLORS["Fora da PEA"], label="Mesmo salario")
            ax.set_xticks(x)
            ax.set_xticklabels(wage_tbl.index, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("% dos que fizeram a transicao")
            ax.set_title(f"Proporcao que ganhou / perdeu / manteve salario na transicao\nPNADC {year_label}")
            ax.legend()
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            plt.tight_layout()
            save_figure("D9_proporcao_ganha_perde", figures_dir, df, wage_df)

            wt_sorted = wage_tbl.sort_values("Var RS mediana")
            colors_bar = [STATE_COLORS["Setor Publico"] if v >= 0 else STATE_COLORS["Desocupado"] for v in wt_sorted["Var RS mediana"].values]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(wt_sorted.index, wt_sorted["Var RS mediana"].values, color=colors_bar, edgecolor="white")
            ax.axvline(0, color="black", linewidth=1.0, linestyle="--")
            ax.set_xlabel("Variacao salarial mediana (R$)")
            ax.set_title(f"Variacao salarial mediana por tipo de transicao\nPNADC {year_label}")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_brl))
            plt.tight_layout()
            save_figure("D10_variacao_salarial_mediana", figures_dir, df, wage_df)

            valid = [(o, d, lbl, c) for o, d, lbl, c in [
                ("Setor Privado", "Setor Privado", "Privado->Privado", STATE_COLORS["Setor Privado"]),
                ("Setor Privado", "Setor Publico", "Privado->Publico", STATE_COLORS["Setor Publico"]),
                ("Setor Publico", "Setor Privado", "Publico->Privado", STATE_COLORS["Desocupado"]),
            ] if len(wage_df[(wage_df.estado_lm == o) & (wage_df.estado_t1 == d)]) >= 5]

            if valid:
                n_p = len(valid)
                fig, axes = plt.subplots(2, n_p, figsize=(5 * n_p, 9))
                if n_p == 1:
                    axes = axes.reshape(2, 1)
                for j, (orig, dest, lbl, color) in enumerate(valid):
                    sub = wage_df[(wage_df.estado_lm == orig) & (wage_df.estado_t1 == dest)]
                    ax_top = axes[0, j]
                    clip_top = sub[[inc_col, f"{inc_col}_t1"]].stack().quantile(0.99)
                    ax_top.hist(sub[inc_col].clip(upper=clip_top), bins=30, color="#AABBCC", alpha=0.7, density=True, label="Renda em t")
                    ax_top.hist(sub[f"{inc_col}_t1"].clip(upper=clip_top), bins=30, color=color, alpha=0.6, density=True, label="Renda em t+1")
                    ax_top.axvline(sub[inc_col].median(), color="#446688", linestyle="--", linewidth=1.2, label=f"Med. t: {fmt_brl(sub[inc_col].median())}")
                    ax_top.axvline(sub[f"{inc_col}_t1"].median(), color=color, linestyle="-.", linewidth=1.2, label=f"Med. t+1: {fmt_brl(sub[f'{inc_col}_t1'].median())}")
                    ax_top.set_title(f"{lbl} (n={len(sub):,})", fontsize=10)
                    ax_top.set_xlabel("Renda (R$)")
                    ax_top.set_ylabel("Densidade")
                    ax_top.legend(fontsize=7)
                    ax_top.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_brl))

                    ax_bot = axes[1, j]
                    delta = sub["delta_pct"].clip(-100, 200)
                    ax_bot.hist(delta, bins=30, color=color, alpha=0.75, edgecolor="white")
                    ax_bot.axvline(0, color="black", linestyle="--", linewidth=1.2)
                    ax_bot.axvline(delta.median(), color="red", linestyle="-.", linewidth=1.2, label=f"Mediana: {delta.median():.1f}%")
                    pct_gain = (delta > 0).mean() * 100
                    ax_bot.set_title(f"{pct_gain:.0f}% ganham salario na transicao")
                    ax_bot.set_xlabel("Variacao salarial (%)")
                    ax_bot.set_ylabel("Contagem")
                    ax_bot.legend(fontsize=8)
                    ax_bot.xaxis.set_major_formatter(mticker.PercentFormatter())

                fig.suptitle(f"Renda e variacao salarial nas transicoes - PNADC {year_label}", fontsize=12)
                plt.tight_layout()
                save_figure("D11_renda_variacao_transicoes", figures_dir, df, wage_df)

                if log_tbl is not None:
                    fig, axes = plt.subplots(1, len(valid), figsize=(5 * len(valid), 4.5))
                    if len(valid) == 1:
                        axes = [axes]
                    for ax, (orig, dest, lbl, color) in zip(axes, valid):
                        sub = wage_df[(wage_df.estado_lm == orig) & (wage_df.estado_t1 == dest)].copy()
                        sub["log_t"] = np.log(sub[inc_col])
                        sub["log_t1"] = np.log(sub[f"{inc_col}_t1"])
                        ax.hist(sub["log_t"], bins=30, color="#AABBCC", alpha=0.7, density=True, label="Log renda t")
                        ax.hist(sub["log_t1"], bins=30, color=color, alpha=0.6, density=True, label="Log renda t+1")
                        ax.set_title(lbl)
                        ax.set_xlabel("Log earnings")
                        ax.legend(fontsize=7)
                    fig.suptitle(f"Log earnings around transitions - PNADC {year_label}", fontsize=12)
                    plt.tight_layout()
                    save_figure("D11_renda_variacao_transicoes_log", figures_dir, df, wage_df)

    summary = []
    tot_pairs = cnt_all.values.sum()
    for orig in STATES:
        n_orig = int(cnt_all.loc[orig].sum())
        for dest in STATES:
            cnt = int(cnt_all.loc[orig, dest])
            if cnt == 0:
                continue
            summary.append(
                {
                    "Estado_t": orig,
                    "Estado_t+1": dest,
                    "N": cnt,
                    "Prob_linha": round(100 * cnt / n_orig, 2) if n_orig else 0,
                    "Prob_total": round(100 * cnt / tot_pairs, 3) if tot_pairs else 0,
                }
            )
    save_table(pd.DataFrame(summary), "D8_transicoes_long", tables_dir, df, transitions, index=False)

    return {
        "years": years_sorted,
        "pairs": len(transitions),
        "rows": len(df),
        "selected_rows": len(selected_rows),
        "year_rows": len(selected_rows),
        "next_obs_in_selected_years": len(eligible_next_obs),
        "same_year_next_obs": len(eligible_next_obs),
        "consecutive_next_obs": len(consecutive_next_obs),
        "tag": tag,
    }
