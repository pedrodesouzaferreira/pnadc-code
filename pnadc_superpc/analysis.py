import textwrap
import unicodedata
import warnings

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.path import Path

from .config import cleaned_path, ensure_output_dirs
from .io_utils import best_column, build_label_maps_from_frame, decode_series, downsample_individuals
from .mappings import ANALYSIS_EDU_COLUMNS


matplotlib.use("Agg")
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

BLUE = "#2B7BB9"
RED = "#E05C5C"
GREEN = "#3AAA5C"
GREY = "#888888"
PANEL_COLS = ["id_domicilio", "num_ordem"]
SIMPLIFIED_STATES = ["Unemployed", "Private", "Public"]
SIMPLIFIED_STATE_COLORS = {
    "Unemployed": RED,
    "Private": BLUE,
    "Public": GREEN,
}
SIMPLIFIED_STATE_LABELS_PT = {
    "Unemployed": "Desocupado",
    "Private": "Setor privado",
    "Public": "Setor publico",
}
WINSORIZED_INCOME_COL = "renda_habitual_principal_winsor"
WINSOR_NOTE = "Winsorized by year at the 1st and 99th percentiles of renda_habitual_principal."
WINSOR_NOTE_PT = "Renda habitual principal winsorizada por ano nos percentis 1 e 99."
FIGURE3_DETAILED_STATES = [
    "Unemployed",
    "Private employee",
    "Domestic worker",
    "Self-employed",
    "Employer",
    "Family auxiliary",
    "Public employee",
]
FIGURE3_DETAILED_LABELS_PT = {
    "Unemployed": "Desocupado",
    "Private employee": "Empregado privado",
    "Domestic worker": "Trabalhador domestico",
    "Self-employed": "Conta propria",
    "Employer": "Empregador",
    "Family auxiliary": "Trabalhador familiar auxiliar",
    "Public employee": "Empregado publico",
}
FIGURE3_COLLAPSED_STATES = ["Unemployed", "Private sector", "Public sector"]
FIGURE3_COLLAPSED_LABELS_PT = {
    "Unemployed": "Desocupado",
    "Private sector": "Setor privado",
    "Public sector": "Setor publico",
}


EXACT_TRANSLATIONS = {
    "Homem": "Men",
    "Mulher": "Women",
    "Branca": "White",
    "Preta": "Black",
    "Parda": "Brown",
    "Amarela": "Asian",
    "Indigena": "Indigenous",
    "Urbana": "Urban",
    "Rural": "Rural",
    "Capital": "Capital city",
    "Interior": "Interior",
    "Sim": "Yes",
    "Nao": "No",
    "Ocupado": "Employed",
    "Desocupado": "Unemployed",
    "Pessoas na forca de trabalho": "In labor force",
    "Pessoas fora da forca de trabalho": "Out of labor force",
    "Sem instrucao e menos de 1 ano de estudo": "No schooling / under 1 year",
    "Fundamental incompleto ou equivalente": "Primary incomplete or equivalent",
    "Fundamental completo ou equivalente": "Primary complete or equivalent",
    "Medio incompleto ou equivalente": "Secondary incomplete or equivalent",
    "Medio completo ou equivalente": "Secondary complete or equivalent",
    "Superior incompleto ou equivalente": "Higher education incomplete",
    "Superior completo": "Higher education complete",
    "Pos-graduacao, mestrado ou doutorado": "Postgraduate / MA / PhD",
}


PHRASE_REPLACEMENTS = [
    ("forca de trabalho", "labor force"),
    ("setor publico", "public sector"),
    ("setor privado", "private sector"),
    ("servidor publico", "public servant"),
    ("posicao no emprego", "employment position"),
    ("condicao de ocupacao", "employment status"),
    ("condicao ocupacao", "employment status"),
    ("setor atividade", "industry"),
    ("concurso", "civil-service exam"),
    ("renda", "earnings"),
    ("nivel de instrucao", "education level"),
    ("idade", "age"),
    ("faixa etaria", "age group"),
    ("trabalhador", "worker"),
    ("empregado", "employee"),
    ("formal", "formal"),
    ("informal", "informal"),
    ("desocupado", "unemployed"),
    ("fora da forca de trabalho", "out of the labor force"),
    ("nao observado", "not observed"),
    ("estudando para", "studying for"),
    ("frequentar escola", "attending school"),
    ("motivo principal para nao frequentar escola", "Main reason for not attending school"),
]


def strip_accents(text):
    return "".join(
        char for char in unicodedata.normalize("NFKD", str(text))
        if not unicodedata.combining(char)
    )


def wrap_label(text, width=24):
    return textwrap.fill(str(text), width=width, break_long_words=False, break_on_hyphens=False)


def english_label(label, max_len=34):
    original = str(label)
    clean = strip_accents(original).strip()
    if clean in EXACT_TRANSLATIONS:
        return EXACT_TRANSLATIONS[clean]

    lower = clean.lower()
    for old, new in PHRASE_REPLACEMENTS:
        lower = lower.replace(old, new)

    translated = lower.title()
    translated = translated.replace(" / ", "/").replace("Civil-Service Exam", "Civil-service exam")
    translated = translated.replace("Ma / Phd", "MA / PhD")
    translated = translated.replace("Nao", "Not").replace(" Ou ", " or ")
    return translated


def translate_labels(labels, max_len=34):
    translated = []
    pairs = []
    for label in labels:
        en = english_label(label, max_len=max_len)
        translated.append(wrap_label(en, width=max_len))
        pairs.append((en, str(label)))
    return translated, pairs


def latex_escape(text):
    out = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


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


PT_TEXT_REPLACEMENTS = {
    "Percent of responses": "Percentual das respostas",
    "Percent of group": "Percentual do grupo",
    "All observed": "Todos observados",
    "Exam-focused": "Grupo de interesse",
    "Exam-searchers": "Concurseiros",
    "Age": "Idade",
    "Density": "Densidade",
    "Quarter": "Trimestre",
    "State at t": "Estado em t",
    "State at t+1": "Estado em t+1",
    "Transition probability (%)": "Probabilidade de transicao (%)",
    "Simplified 3-state transitions for exam-searchers": "Transicoes simplificadas em 3 estados para concurseiros",
    "Simplified 3-state flows for exam-searchers": "Fluxos simplificados em 3 estados para concurseiros",
    "Simplified 3-state transitions in the pooled panel": "Transicoes simplificadas em 3 estados no painel pooled",
    "Simplified 3-state flows in the pooled panel": "Fluxos simplificados em 3 estados no painel pooled",
    "Main reason for not attending school": "Principal motivo para nao frequentar escola",
    "Education profile": "Perfil educacional",
    "Age distribution": "Distribuicao etaria",
    "Studying for civil-service exam": "Estudando para concurso",
    "Job search via civil-service exam": "Busca de emprego via concurso",
    "Mean earnings": "Renda media",
    "Median earnings": "Renda mediana",
    "Monthly earnings": "Renda mensal",
    "Log earnings": "Log da renda",
    "Log monthly earnings": "Log da renda mensal",
    "Log wage distribution by quarter": "Distribuicao do log da renda por trimestre",
    "Transition probabilities between employment statuses": "Probabilidades de transicao entre status ocupacionais",
    "Transition probabilities after the 3-state simplification": "Probabilidades de transicao apos a simplificacao em 3 estados",
    "Employment status at t": "Status ocupacional em t",
    "Employment status at t+1": "Status ocupacional em t+1",
    "Private employee": "Empregado privado",
    "Domestic worker": "Trabalhador domestico",
    "Self-employed": "Conta propria",
    "Employer": "Empregador",
    "Family auxiliary": "Trabalhador familiar auxiliar",
    "Public employee": "Empregado publico",
    "Unemployed": "Desocupado",
    "Private": "Privado",
    "Public": "Publico",
    "Public sector": "Setor publico",
    "Private sector": "Setor privado",
}


def translate_text_pt(text, category_pairs=None):
    out = str(text)
    if category_pairs:
        for en, pt in sorted(category_pairs, key=lambda item: len(item[0]), reverse=True):
            out = out.replace(str(en), wrap_label(str(pt), width=24))
    for en, pt in sorted(PT_TEXT_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        out = out.replace(en, pt)
    return out


PT_COLUMN_REPLACEMENTS = {
    "group": "grupo",
    "mean": "media",
    "median": "mediana",
    "p25": "p25",
    "p75": "p75",
    "n": "n",
    "pct": "pct",
    "period": "periodo",
    "age": "idade",
    "status_en": "status_pt",
    "education_en": "instrucao_pt",
    "destination_en": "destino_pt",
    "origin_en": "origem_pt",
    "reason_en": "motivo_pt",
    "mean_log": "media_log",
    "median_log": "mediana_log",
    "median_earnings": "renda_mediana",
    "median_log_earnings": "log_renda_mediana",
    "Public": "Publico",
    "Private": "Privado",
    "Public_premium_pct": "premio_publico_pct",
}


def localize_table_pt(df):
    out = df.copy()
    out.columns = [PT_COLUMN_REPLACEMENTS.get(str(col), str(col)) for col in out.columns]
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(lambda value: translate_text_pt(value) if pd.notna(value) else value)
    return out


def localize_figure_pt(fig, category_pairs=None):
    for ax in fig.axes:
        ax.set_title(translate_text_pt(ax.get_title(), category_pairs))
        ax.set_xlabel(translate_text_pt(ax.get_xlabel(), category_pairs))
        ax.set_ylabel(translate_text_pt(ax.get_ylabel(), category_pairs))
        ax.set_xticklabels([translate_text_pt(t.get_text(), category_pairs) for t in ax.get_xticklabels()])
        ax.set_yticklabels([translate_text_pt(t.get_text(), category_pairs) for t in ax.get_yticklabels()])
        legend = ax.get_legend()
        if legend is not None:
            legend.set_title(translate_text_pt(legend.get_title().get_text(), category_pairs))
            for txt in legend.get_texts():
                txt.set_text(translate_text_pt(txt.get_text(), category_pairs))


def save_table(df, name, tables_dir, full_df, used_df, index=True, pt_df=None, extra_note=None, extra_note_pt=None):
    path = tables_dir / f"{name}.tex"
    latex = df.to_latex(index=index, float_format=lambda x: f"{x:,.1f}", na_rep="--", escape=True)
    note = latex_escape(output_note(full_df, used_df))
    latex = latex.rstrip() + f"\n\n\\begin{{flushleft}}\\footnotesize {note}\\end{{flushleft}}\n"
    if extra_note:
        latex += f"\\begin{{flushleft}}\\footnotesize {latex_escape(extra_note)}\\end{{flushleft}}\n"
    path.write_text(latex, encoding="utf-8")

    pt_df = pt_df if pt_df is not None else localize_table_pt(df)
    pt_latex = pt_df.to_latex(index=index, float_format=lambda x: f"{x:,.1f}", na_rep="--", escape=True)
    pt_note = latex_escape(output_note_pt(full_df, used_df))
    pt_latex = pt_latex.rstrip() + f"\n\n\\begin{{flushleft}}\\footnotesize {pt_note}\\end{{flushleft}}\n"
    if extra_note or extra_note_pt:
        pt_extra = extra_note_pt if extra_note_pt is not None else extra_note
        pt_latex += f"\\begin{{flushleft}}\\footnotesize {latex_escape(pt_extra)}\\end{{flushleft}}\n"
    pt_path = tables_dir / f"{name}_pt.tex"
    pt_path.write_text(pt_latex, encoding="utf-8")


def save_figure(
    name,
    figures_dir,
    full_df,
    used_df,
    category_pairs=None,
    extra_note=None,
    extra_note_pt=None,
    dpi=150,
    export_pdf=False,
    include_notes=True,
):
    fig = plt.gcf()
    note_lines = [output_note(full_df, used_df)] if include_notes else []

    if include_notes and extra_note:
        note_lines.extend(textwrap.wrap(extra_note, width=125))

    if include_notes and category_pairs:
        mapping_text = "Original Portuguese labels: " + "; ".join(
            f"{en} = {pt}" for en, pt in category_pairs[:12]
        )
        note_lines.extend(textwrap.wrap(mapping_text, width=125))

    if note_lines:
        bottom = 0.18 + 0.035 * min(len(note_lines), 6)
        fig.subplots_adjust(bottom=min(bottom, 0.38))
        y = 0.02
        for line in note_lines[:6]:
            fig.text(0.01, y, line, ha="left", va="bottom", fontsize=7)
            y += 0.022

    path = figures_dir / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if export_pdf:
        pdf_path = figures_dir / f"{name}.pdf"
        plt.savefig(pdf_path, bbox_inches="tight")
    localize_figure_pt(fig, category_pairs=category_pairs)
    if include_notes:
        pt_note_lines = [output_note_pt(full_df, used_df)]
        if extra_note_pt:
            pt_note_lines.extend(textwrap.wrap(extra_note_pt, width=125))
        elif extra_note:
            pt_note_lines.extend(textwrap.wrap(f"Titulo/origem em portugues: {extra_note.split(':', 1)[-1].strip()}", width=125))
        if category_pairs:
            mapping_text_pt = "Rotulos originais em portugues: " + "; ".join(
                f"{en} = {pt}" for en, pt in category_pairs[:12]
            )
            pt_note_lines.extend(textwrap.wrap(mapping_text_pt, width=125))
        for idx, txt in enumerate(fig.texts[-len(note_lines[:6]):]):
            if idx < len(pt_note_lines[:6]):
                txt.set_text(pt_note_lines[idx])
    pt_path = figures_dir / f"{name}_pt.png"
    plt.savefig(pt_path, dpi=dpi, bbox_inches="tight")
    if export_pdf:
        pt_pdf_path = figures_dir / f"{name}_pt.pdf"
        plt.savefig(pt_pdf_path, bbox_inches="tight")
    plt.close()


def _load_merged_year(year, use_sample_inputs=False, age_min=None, age_max=None):
    mdf = pd.read_csv(cleaned_path("microdados", year, use_sample=use_sample_inputs), low_memory=False)
    edf = pd.read_csv(cleaned_path("educacao", year, use_sample=use_sample_inputs), low_memory=False)
    for frame in (mdf, edf):
        if "num_ordem" not in frame.columns and "V2003" in frame.columns:
            frame["num_ordem"] = pd.to_numeric(frame["V2003"], errors="coerce")
        for column in ["ano", "trimestre", "num_ordem"]:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "idade" in mdf.columns and (age_min is not None or age_max is not None):
        mdf["idade"] = pd.to_numeric(mdf["idade"], errors="coerce")
        age_mask = pd.Series(True, index=mdf.index)
        if age_min is not None:
            age_mask &= mdf["idade"] >= age_min
        if age_max is not None:
            age_mask &= mdf["idade"] <= age_max
        mdf = mdf.loc[age_mask].copy()
    keep_cols = [column for column in ANALYSIS_EDU_COLUMNS if column in edf.columns]
    return mdf.merge(
        edf[keep_cols],
        on=["id_domicilio", "num_ordem", "ano", "trimestre"],
        how="left",
        suffixes=("", "_edu"),
    )


def load_merged(years, sample_frac=None, seed=42, use_sample_inputs=False, age_min=None, age_max=None):
    if isinstance(years, int):
        years = [years]
    frames = [
        _load_merged_year(
            y,
            use_sample_inputs=use_sample_inputs,
            age_min=age_min,
            age_max=age_max,
        )
        for y in years
    ]
    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return downsample_individuals(combined, sample_frac, seed, panel_cols=PANEL_COLS)


def add_year_winsorized_income(df, source_col="renda_habitual_principal", output_col=WINSORIZED_INCOME_COL, lower_q=0.01, upper_q=0.99):
    if source_col not in df.columns:
        return df

    out = df.copy()
    out[source_col] = pd.to_numeric(out[source_col], errors="coerce")
    year_col = pd.to_numeric(out["ano"], errors="coerce") if "ano" in out.columns else pd.Series(np.nan, index=out.index)

    if year_col.notna().any():
        bounds = (
            pd.DataFrame({"ano": year_col, source_col: out[source_col]})
            .dropna(subset=["ano"])
            .groupby("ano")[source_col]
            .quantile([lower_q, upper_q])
            .unstack()
            .rename(columns={lower_q: "_winsor_low", upper_q: "_winsor_high"})
        )
        out = out.join(bounds, on="ano")
        out[output_col] = out[source_col].clip(lower=out["_winsor_low"], upper=out["_winsor_high"])
        out = out.drop(columns=["_winsor_low", "_winsor_high"])
    else:
        low = out[source_col].quantile(lower_q)
        high = out[source_col].quantile(upper_q)
        out[output_col] = out[source_col].clip(lower=low, upper=high)
    return out


def add_panel_lags(df):
    df = df.sort_values(PANEL_COLS + ["ano", "trimestre"]).reset_index(drop=True)

    lag_columns = [
        "ano",
        "trimestre",
        "posicao_emprego",
        "setor_atividade",
        "condicao_ocupacao",
        "servidor_publico",
        "buscando_via_concurso",
        "estudando_concurso",
        "metodo_busca_emprego",
        "renda_habitual_principal",
        WINSORIZED_INCOME_COL,
        "renda_habitual_todos",
        "ocupado",
        "desocupado",
        "informal",
        "formal",
        "empregado_setor_priv",
        "empregado_setor_pub",
        "conta_propria",
        "empregador",
        "trab_domestico",
        "trab_familiar_aux",
        "tempo_nesse_trabalho",
    ]
    for column in lag_columns:
        if column not in df.columns:
            continue
        group = df.groupby(PANEL_COLS)[column]
        df[f"{column}_t1"] = group.shift(1)
        df[f"{column}_t2"] = group.shift(2)
        df[f"{column}_f1"] = group.shift(-1)

    for base in ["posicao_emprego", "setor_atividade", "condicao_ocupacao", "tempo_nesse_trabalho"]:
        label_col = base + "_label"
        if label_col not in df.columns:
            continue
        group = df.groupby(PANEL_COLS)[label_col]
        df[label_col + "_t1"] = group.shift(1)
        df[label_col + "_f1"] = group.shift(-1)
    return df


def income_column(df, suffix="", winsorized=False):
    candidates = []
    if winsorized:
        candidates.append(f"{WINSORIZED_INCOME_COL}{suffix}")
    candidates.extend(
        [
            f"renda_habitual_principal{suffix}",
            f"renda_habitual_todos{suffix}",
            f"renda_habitual_total{suffix}",
        ]
    )
    return next(
        (
            column
            for column in candidates
            if column in df.columns and df[column].notna().any()
        ),
        None,
    )


def positive_log(series):
    return np.log(series.where(series > 0))


PERCENTILE_LEVELS = [
    (0.10, "p10"),
    (0.25, "p25"),
    (0.50, "p50"),
    (0.75, "p75"),
    (0.90, "p90"),
    (0.95, "p95"),
    (0.99, "p99"),
    (0.999, "p999"),
]


def add_period_labels(frame):
    out = frame.copy()
    if "ano" not in out.columns or "trimestre" not in out.columns:
        out["period"] = "All"
        out["period_order"] = 0
        return out
    if out["ano"].nunique(dropna=True) > 1:
        out["period"] = out["ano"].astype("Int64").astype(str) + "Q" + out["trimestre"].astype("Int64").astype(str)
    else:
        out["period"] = "Q" + out["trimestre"].astype("Int64").astype(str)
    out["period_order"] = out["ano"].fillna(0).astype(int) * 10 + out["trimestre"].fillna(0).astype(int)
    return out


def percentile_table_by_period(frame, income_col):
    rows = []
    cols = ["period", "n"] + [label for _, label in PERCENTILE_LEVELS]
    if frame.empty:
        return pd.DataFrame(columns=cols)
    for _, grp in frame.sort_values("period_order").groupby("period", sort=False):
        values = pd.to_numeric(grp[income_col], errors="coerce")
        values = values[(values > 0) & values.notna()]
        if values.empty:
            continue
        row = {"period": grp["period"].iloc[0], "n": int(len(values))}
        for q, label in PERCENTILE_LEVELS:
            row[label] = round(values.quantile(q), 0)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def plot_log_distribution_by_period(frame, income_col, sector_label, year_label, max_bins=45):
    plot = frame.copy()
    plot["log_earnings"] = positive_log(plot[income_col])
    plot = plot[plot["log_earnings"].notna()].copy()
    if plot.empty:
        return False

    periods = (
        plot[["period", "period_order"]]
        .drop_duplicates()
        .sort_values("period_order")["period"]
        .tolist()
    )
    if not periods:
        return False

    lower = plot["log_earnings"].quantile(0.005)
    upper = plot["log_earnings"].quantile(0.995)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        lower = plot["log_earnings"].min()
        upper = plot["log_earnings"].max()
    bins = np.linspace(lower, upper, max_bins)
    colors = sns.color_palette("viridis", n_colors=len(periods))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for period, color in zip(periods, colors):
        values = plot.loc[plot["period"] == period, "log_earnings"].dropna()
        if len(values) < 10:
            continue
        hist, edges = np.histogram(values.clip(lower=lower, upper=upper), bins=bins, density=True)
        mids = (edges[:-1] + edges[1:]) / 2
        ax.plot(mids, hist, color=color, linewidth=2, label=period)

    ax.set_xlabel("Log monthly earnings")
    ax.set_ylabel("Density")
    ax.set_title(f"Log wage distribution by quarter\n{sector_label} sector (PNADC {year_label})")
    ax.legend(title="Quarter", fontsize=8, ncol=2 if len(periods) > 4 else 1)
    plt.tight_layout()
    return True


def quarter_distance(curr_year, curr_quarter, prev_year, prev_quarter):
    return (curr_year * 4 + curr_quarter) - (prev_year * 4 + prev_quarter)


def indicator_is_one(frame, column):
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0).eq(1)


def simplified_state(frame, suffix=""):
    public = indicator_is_one(frame, f"empregado_setor_pub{suffix}")
    private = (
        indicator_is_one(frame, f"empregado_setor_priv{suffix}")
        | indicator_is_one(frame, f"conta_propria{suffix}")
        | indicator_is_one(frame, f"empregador{suffix}")
        | indicator_is_one(frame, f"trab_domestico{suffix}")
    )
    unemployed = (
        indicator_is_one(frame, f"desocupado{suffix}")
        | indicator_is_one(frame, f"trab_familiar_aux{suffix}")
    )
    state = np.select(
        [public, private, unemployed],
        ["Public", "Private", "Unemployed"],
        default="Unemployed",
    )
    return pd.Series(state, index=frame.index)


def figure3_detailed_state(frame, suffix=""):
    in_lf = indicator_is_one(frame, f"na_pea{suffix}") if f"na_pea{suffix}" in frame.columns else pd.Series(True, index=frame.index)
    private_employee = indicator_is_one(frame, f"empregado_setor_priv{suffix}")
    domestic_worker = indicator_is_one(frame, f"trab_domestico{suffix}")
    self_employed = indicator_is_one(frame, f"conta_propria{suffix}")
    employer = indicator_is_one(frame, f"empregador{suffix}")
    family_auxiliary = indicator_is_one(frame, f"trab_familiar_aux{suffix}")
    public_employee = indicator_is_one(frame, f"empregado_setor_pub{suffix}")
    unemployed = indicator_is_one(frame, f"desocupado{suffix}")
    state = np.select(
        [
            unemployed,
            private_employee,
            domestic_worker,
            self_employed,
            employer,
            family_auxiliary,
            public_employee,
        ],
        FIGURE3_DETAILED_STATES,
        default=None,
    )
    out = pd.Series(state, index=frame.index, dtype="object")
    out = out.where(in_lf)
    return out


def figure3_collapsed_state(frame, suffix=""):
    in_lf = indicator_is_one(frame, f"na_pea{suffix}") if f"na_pea{suffix}" in frame.columns else pd.Series(True, index=frame.index)
    public = indicator_is_one(frame, f"empregado_setor_pub{suffix}")
    private = (
        indicator_is_one(frame, f"empregado_setor_priv{suffix}")
        | indicator_is_one(frame, f"conta_propria{suffix}")
        | indicator_is_one(frame, f"empregador{suffix}")
        | indicator_is_one(frame, f"trab_domestico{suffix}")
    )
    unemployed = (
        indicator_is_one(frame, f"desocupado{suffix}")
        | indicator_is_one(frame, f"trab_familiar_aux{suffix}")
    )
    state = np.select(
        [unemployed, private, public],
        FIGURE3_COLLAPSED_STATES,
        default=None,
    )
    out = pd.Series(state, index=frame.index, dtype="object")
    out = out.where(in_lf)
    return out


def next_consecutive_observation(frame):
    next_year = pd.to_numeric(frame.get("ano_f1"), errors="coerce")
    next_quarter = pd.to_numeric(frame.get("trimestre_f1"), errors="coerce")
    curr_year = pd.to_numeric(frame.get("ano"), errors="coerce")
    curr_quarter = pd.to_numeric(frame.get("trimestre"), errors="coerce")
    return quarter_distance(next_year, next_quarter, curr_year, curr_quarter) == 1


def simplified_transition_matrices(frame, origin_col, dest_col):
    counts = (
        frame.groupby([origin_col, dest_col])
        .size()
        .unstack(fill_value=0)
        .reindex(index=SIMPLIFIED_STATES, columns=SIMPLIFIED_STATES, fill_value=0)
    )
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return counts, probs


def transition_matrices(frame, origin_col, dest_col, states):
    counts = (
        frame.groupby([origin_col, dest_col])
        .size()
        .unstack(fill_value=0)
        .reindex(index=states, columns=states, fill_value=0)
    )
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return counts, probs


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
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    patch = mpatches.PathPatch(
        Path(verts, codes),
        facecolor=mcolors.to_rgba(color, alpha),
        edgecolor="none",
        zorder=1,
    )
    ax.add_patch(patch)


def draw_simplified_sankey(count_matrix, title="", ax=None, bar_width=0.08, gap=0.05):
    if ax is None:
        _, ax = plt.subplots(figsize=(9.5, 6.5))

    total = count_matrix.values.sum()
    if total == 0:
        return

    row_totals = count_matrix.sum(axis=1)
    col_totals = count_matrix.sum(axis=0)
    left_x = 0.12
    right_x = 0.82

    def bar_positions(totals):
        info = {}
        y = 1.0
        for state in SIMPLIFIED_STATES:
            height = totals[state] / total
            info[state] = (y - height, y)
            y -= height + gap
        return info

    left_bars = bar_positions(row_totals)
    right_bars = bar_positions(col_totals)

    for side, xpos, bars in [("left", left_x, left_bars), ("right", right_x, right_bars)]:
        for state in SIMPLIFIED_STATES:
            bottom, top = bars[state]
            rect = mpatches.FancyBboxPatch(
                (xpos, bottom),
                bar_width,
                top - bottom,
                boxstyle="square,pad=0",
                facecolor=SIMPLIFIED_STATE_COLORS[state],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
            ax.add_patch(rect)
            mid = (bottom + top) / 2
            pct = (top - bottom) * 100
            if side == "left":
                ax.text(xpos - 0.02, mid, f"{state}\n({pct:.0f}%)", ha="right", va="center", fontsize=8)
            else:
                ax.text(xpos + bar_width + 0.02, mid, f"{state}\n({pct:.0f}%)", ha="left", va="center", fontsize=8)

    left_cursor = {state: left_bars[state][1] for state in SIMPLIFIED_STATES}
    right_cursor = {state: right_bars[state][1] for state in SIMPLIFIED_STATES}

    for origin in SIMPLIFIED_STATES:
        for destination in SIMPLIFIED_STATES:
            count = count_matrix.loc[origin, destination]
            if count == 0:
                continue
            height = count / total
            y1_top = left_cursor[origin]
            y1_bot = y1_top - height
            y2_top = right_cursor[destination]
            y2_bot = y2_top - height
            left_cursor[origin] -= height
            right_cursor[destination] -= height
            _bezier_flow(
                ax,
                left_x + bar_width,
                y1_top,
                y1_bot,
                right_x,
                y2_top,
                y2_bot,
                SIMPLIFIED_STATE_COLORS[origin],
            )

    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.05, 1.08)
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)


def new_job_proxy(frame):
    if frame.empty:
        return pd.Series(dtype="bool")
    work = frame.copy()
    work["tempo_num"] = pd.to_numeric(work.get("tempo_nesse_trabalho"), errors="coerce")
    work["tempo_t1_num"] = pd.to_numeric(work.get("tempo_nesse_trabalho_t1"), errors="coerce")
    prev_year = pd.to_numeric(work.get("ano_t1"), errors="coerce")
    prev_quarter = pd.to_numeric(work.get("trimestre_t1"), errors="coerce")
    curr_year = pd.to_numeric(work.get("ano"), errors="coerce")
    curr_quarter = pd.to_numeric(work.get("trimestre"), errors="coerce")

    consecutive_prev = quarter_distance(curr_year, curr_quarter, prev_year, prev_quarter) == 1
    employed_now = pd.to_numeric(work.get("ocupado"), errors="coerce") == 1
    employed_prev = pd.to_numeric(work.get("ocupado_t1"), errors="coerce") == 1
    switched_job = (
        (work["tempo_num"] == 1)
        | (
            work["tempo_num"].notna()
            & work["tempo_t1_num"].notna()
            & (work["tempo_num"] < work["tempo_t1_num"])
        )
    )
    entered_employment = employed_now & (~employed_prev)
    return consecutive_prev & employed_now & (entered_employment | (employed_prev & switched_job))


def maybe_decode(frame, column, label_maps):
    if column in frame.columns:
        return frame[column]
    fallback = column.replace("_label", "")
    if fallback in frame.columns:
        return decode_series(frame[fallback], fallback, label_maps)
    return pd.Series(dtype="object")


def profile_table(frame):
    profile = {}
    for column_name, display in [
        ("sexo", "Sex"),
        ("raca_cor", "Race/color"),
        ("faixa_etaria", "Age group"),
        ("nivel_instrucao", "Education"),
        ("zona", "Zone"),
    ]:
        use_col = best_column(frame, column_name)
        if use_col not in frame.columns:
            continue
        shares = frame[use_col].value_counts(normalize=True).mul(100).round(1)
        for key, value in shares.items():
            profile[f"{display}: {key}"] = value
    if not profile:
        return None
    return pd.DataFrame.from_dict(profile, orient="index", columns=["pct_%"])


def run_analysis(years, sample_frac=None, seed=42, tag=None, use_sample_inputs=False, age_min=None, age_max=None):
    if isinstance(years, int):
        years = [years]
    years_sorted = sorted(years)

    if tag is None:
        year_str = "_".join(str(y) for y in years_sorted)
        age_str = f"_age{age_min}_{age_max}" if (age_min is not None or age_max is not None) else ""
        qual = "sample" if sample_frac and sample_frac < 1 else "full"
        tag = f"{year_str}{age_str}_{qual}"

    year_label = "/".join(str(y) for y in years_sorted)  # used in plot titles
    tables_dir, figures_dir = ensure_output_dirs(tag=f"analysis_{tag}")

    raw = load_merged(
        years,
        sample_frac=sample_frac,
        seed=seed,
        use_sample_inputs=use_sample_inputs,
        age_min=age_min,
        age_max=age_max,
    )
    raw = add_year_winsorized_income(raw)
    merged = add_panel_lags(raw)
    label_maps = build_label_maps_from_frame(merged)

    for label, mask_col, short_tag in [
        ("Studying for civil-service exam", "estudando_concurso", "def1"),
        ("Job search via civil-service exam", "buscando_via_concurso", "def2"),
    ]:
        if mask_col not in merged.columns:
            continue
        grp = merged[merged[mask_col] == 1].copy()
        if grp.empty:
            continue

        if short_tag == "def1" and "motivo_nao_frequenta_escola_atual" in merged.columns:
            raw_col = "motivo_nao_frequenta_escola_atual"
            used = merged[merged[raw_col].notna()].copy()
            dist = used[raw_col].value_counts(dropna=False).rename_axis("code").reset_index(name="n")
            label_col = raw_col + "_label"
            if label_col in used.columns:
                direct_map = (
                    used[[raw_col, label_col]]
                    .dropna()
                    .drop_duplicates(subset=[raw_col])
                    .set_index(raw_col)[label_col]
                    .astype(str)
                    .to_dict()
                )
                dist["reason_pt"] = dist["code"].map(direct_map)
            else:
                dist["reason_pt"] = np.nan
            fallback = decode_series(dist["code"], raw_col, label_maps)
            dist["reason_pt"] = dist["reason_pt"].fillna(fallback)
            dist["reason_en"], pairs = translate_labels(dist["reason_pt"].fillna("N/A").tolist(), max_len=36)
            dist["pct"] = (dist["n"] / dist["n"].sum() * 100).round(1)
            save_table(
                dist[["reason_en", "n", "pct"]],
                "A0_V3034C_distribuicao",
                tables_dir,
                merged,
                used,
                index=False,
                pt_df=dist[["reason_pt", "n", "pct"]].rename(columns={"reason_pt": "motivo_pt"}),
            )

            top = dist.dropna(subset=["reason_pt"]).head(12).copy()
            fig, ax = plt.subplots(figsize=(10.5, 5.5))
            ax.barh(top["reason_en"][::-1], top["pct"][::-1], color=BLUE)
            ax.set_xlabel("Percent of responses")
            ax.set_title(f"Main reason for not attending school\n(V3034C, PNADC {year_label})")
            plt.tight_layout()
            save_figure(
                "A0_V3034C_distribuicao",
                figures_dir,
                merged,
                used,
                category_pairs=pairs,
                extra_note="Original variable title in Portuguese: Motivo principal para nao frequentar escola (V3034C).",
            )
            legacy_png = figures_dir / "A0_def1_distribuicao.png"
            canonical_png = figures_dir / "A0_V3034C_distribuicao.png"
            if canonical_png.exists():
                legacy_png.write_bytes(canonical_png.read_bytes())

        prof = profile_table(grp)
        if prof is not None:
            save_table(prof, f"A1_{short_tag}_perfil", tables_dir, merged, grp)

        if "idade" in grp.columns:
            save_table(grp["idade"].agg(["mean", "median", "std", "min", "max"]).round(1).to_frame("age"), f"A2_{short_tag}_idade", tables_dir, merged, grp)

        use_cond = best_column(grp, "condicao_ocupacao")
        if use_cond in grp.columns:
            labor = grp[use_cond].value_counts(dropna=False).rename_axis("status_pt").reset_index(name="n")
            labor["status_en"], _ = translate_labels(labor["status_pt"].tolist(), max_len=32)
            labor["pct"] = (labor["n"] / labor["n"].sum() * 100).round(1)
            save_table(labor[["status_en", "n", "pct"]], f"A3_{short_tag}_status_laboral", tables_dir, merged, grp, index=False)

        inc_col = income_column(grp)
        if inc_col:
            r_grp = grp[inc_col].dropna()
            r_all = merged[inc_col].dropna()
            used = grp[grp[inc_col].notna()].copy()
            inc_df = pd.DataFrame(
                {
                    "group": ["Exam-focused", "All observed"],
                    "mean": [r_grp.mean(), r_all.mean()],
                    "median": [r_grp.median(), r_all.median()],
                    "p25": [r_grp.quantile(0.25), r_all.quantile(0.25)],
                    "p75": [r_grp.quantile(0.75), r_all.quantile(0.75)],
                    "n": [len(r_grp), len(r_all)],
                }
            )
            save_table(inc_df, f"A4_{short_tag}_renda", tables_dir, merged, used, index=False)

            log_df = inc_df.copy()
            log_df["mean_log"] = [positive_log(r_grp).mean(), positive_log(r_all).mean()]
            log_df["median_log"] = [positive_log(r_grp).median(), positive_log(r_all).median()]
            save_table(log_df[["group", "mean_log", "median_log", "n"]], f"A4_{short_tag}_renda_log", tables_dir, merged, used, index=False)

        inc_col_w = income_column(grp, winsorized=True)
        if inc_col_w and inc_col_w != inc_col:
            r_grp_w = grp[inc_col_w].dropna()
            r_all_w = merged[inc_col_w].dropna()
            used_w = grp[grp[inc_col_w].notna()].copy()
            inc_df_w = pd.DataFrame(
                {
                    "group": ["Exam-focused", "All observed"],
                    "mean": [r_grp_w.mean(), r_all_w.mean()],
                    "median": [r_grp_w.median(), r_all_w.median()],
                    "p25": [r_grp_w.quantile(0.25), r_all_w.quantile(0.25)],
                    "p75": [r_grp_w.quantile(0.75), r_all_w.quantile(0.75)],
                    "n": [len(r_grp_w), len(r_all_w)],
                }
            )
            save_table(
                inc_df_w,
                f"A4_{short_tag}_renda_winsor",
                tables_dir,
                merged,
                used_w,
                index=False,
                extra_note=WINSOR_NOTE,
                extra_note_pt=WINSOR_NOTE_PT,
            )

            log_df_w = inc_df_w.copy()
            log_df_w["mean_log"] = [positive_log(r_grp_w).mean(), positive_log(r_all_w).mean()]
            log_df_w["median_log"] = [positive_log(r_grp_w).median(), positive_log(r_all_w).median()]
            save_table(
                log_df_w[["group", "mean_log", "median_log", "n"]],
                f"A4_{short_tag}_renda_log_winsor",
                tables_dir,
                merged,
                used_w,
                index=False,
                extra_note=WINSOR_NOTE,
                extra_note_pt=WINSOR_NOTE_PT,
            )

        use_ni_grp = best_column(grp, "nivel_instrucao")
        use_ni_all = best_column(merged, "nivel_instrucao")
        if use_ni_grp in grp.columns and use_ni_all in merged.columns:
            ed_grp = grp[use_ni_grp].value_counts(normalize=True).mul(100).rename("Exam-focused")
            ed_all = merged[use_ni_all].value_counts(normalize=True).mul(100).rename("All observed")
            ed = pd.concat([ed_grp, ed_all], axis=1).fillna(0)
            translated, pairs = translate_labels(ed.index.tolist(), max_len=28)
            ed.index = translated
            fig, ax = plt.subplots(figsize=(12, 5.5))
            ed.plot(kind="barh", ax=ax, color=[BLUE, GREY])
            ax.set_xlabel("Percent of group")
            ax.set_title(f"Education profile: {label}")
            ax.legend(loc="lower right")
            plt.tight_layout()
            save_figure(f"A5_{short_tag}_nivel_instrucao", figures_dir, merged, grp, category_pairs=pairs)

        if "idade" in grp.columns:
            used = merged[merged["idade"].notna()].copy()
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.hist(merged["idade"].dropna(), bins=range(14, 75), color=GREY, alpha=0.5, density=True, label="All observed")
            ax.hist(grp["idade"].dropna(), bins=range(14, 75), color=BLUE, alpha=0.7, density=True, label=label)
            ax.set_xlabel("Age")
            ax.set_ylabel("Density")
            ax.set_title(f"Age distribution: {label}")
            ax.legend()
            plt.tight_layout()
            save_figure(f"A6_{short_tag}_distribuicao_idade", figures_dir, merged, used)

    if "buscando_via_concurso" in merged.columns:
        seekers = merged[merged["buscando_via_concurso"] == 1].copy()
        if not seekers.empty:
            for column, name in [
                ("posicao_emprego_label_t1", "B1_prior_employment_position_t1"),
                ("condicao_ocupacao_label_t1", "B2_prior_employment_status_t1"),
                ("setor_atividade_label_t1", "B3_prior_industry_t1"),
            ]:
                values = maybe_decode(seekers, column, label_maps)
                if values.empty:
                    continue
                table = values.value_counts(dropna=False).rename_axis("label_pt").reset_index(name="n")
                table["label_en"], _ = translate_labels(table["label_pt"].tolist(), max_len=34)
                table["pct"] = (table["n"] / len(seekers) * 100).round(1)
                save_table(table[["label_en", "n", "pct"]], name, tables_dir, merged, seekers, index=False)

            lag_inc_col = income_column(seekers, suffix="_t1")
            if lag_inc_col:
                used = seekers[seekers[lag_inc_col].notna()].copy()
                r = used[lag_inc_col]
                inc = pd.DataFrame({"stat": ["mean", "median", "p25", "p75", "n"], "value": [r.mean(), r.median(), r.quantile(0.25), r.quantile(0.75), len(r)]})
                save_table(inc, "B4_renda_antes_concurso", tables_dir, merged, used, index=False)
                log_inc = pd.DataFrame({"stat": ["mean_log", "median_log", "n"], "value": [positive_log(r).mean(), positive_log(r).median(), len(r)]})
                save_table(log_inc, "B4_renda_antes_concurso_log", tables_dir, merged, used, index=False)

            lag_inc_col_w = income_column(seekers, suffix="_t1", winsorized=True)
            if lag_inc_col_w and lag_inc_col_w != lag_inc_col:
                used_w = seekers[seekers[lag_inc_col_w].notna()].copy()
                r_w = used_w[lag_inc_col_w]
                inc_w = pd.DataFrame({"stat": ["mean", "median", "p25", "p75", "n"], "value": [r_w.mean(), r_w.median(), r_w.quantile(0.25), r_w.quantile(0.75), len(r_w)]})
                save_table(
                    inc_w,
                    "B4_renda_antes_concurso_winsor",
                    tables_dir,
                    merged,
                    used_w,
                    index=False,
                    extra_note=WINSOR_NOTE,
                    extra_note_pt=WINSOR_NOTE_PT,
                )
                log_inc_w = pd.DataFrame({"stat": ["mean_log", "median_log", "n"], "value": [positive_log(r_w).mean(), positive_log(r_w).median(), len(r_w)]})
                save_table(
                    log_inc_w,
                    "B4_renda_antes_concurso_log_winsor",
                    tables_dir,
                    merged,
                    used_w,
                    index=False,
                    extra_note=WINSOR_NOTE,
                    extra_note_pt=WINSOR_NOTE_PT,
                )

            def classify_next(row):
                if pd.isna(row.get("ocupado_f1")):
                    return "Not observed"
                if row.get("servidor_publico_f1") == 1:
                    return "Became public servant"
                if row.get("empregado_setor_pub_f1") == 1:
                    return "Public-sector employee"
                if row.get("buscando_via_concurso_f1") == 1:
                    return "Still searching via exam"
                if row.get("ocupado_f1") == 1 and row.get("formal_f1") == 1:
                    return "Formal private job"
                if row.get("ocupado_f1") == 1 and row.get("trab_familiar_aux_f1") == 1:
                    return "Family auxiliary work"
                if row.get("ocupado_f1") == 1 and row.get("informal_f1") == 1:
                    return "Informal job"
                if row.get("ocupado_f1") == 1:
                    return "Employed, other"
                if row.get("desocupado_f1") == 1:
                    return "Unemployed"
                return "Out of labor force"

            seekers["next_status"] = seekers.apply(classify_next, axis=1)
            trans = seekers["next_status"].value_counts().rename_axis("next_status").reset_index(name="n")
            trans["pct"] = (trans["n"] / len(seekers) * 100).round(1)
            save_table(trans, "B5_transicoes_pos_concurso", tables_dir, merged, seekers, index=False)

            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            colors = [GREEN if "public servant" in s.lower() else BLUE if "search" in s.lower() else GREY for s in trans["next_status"]]
            ax.barh(trans["next_status"][::-1], trans["pct"][::-1], color=colors[::-1])
            ax.set_xlabel("Percent of exam-searchers")
            ax.set_title("What happened in the next quarter?")
            plt.tight_layout()
            save_figure("B1_transicoes_concurseiros", figures_dir, merged, seekers)

            seekers["estado_simplificado_t"] = simplified_state(seekers)
            seekers["estado_simplificado_f1"] = simplified_state(seekers, suffix="_f1")
            seekers["par_consecutivo_f1"] = next_consecutive_observation(seekers)
            seekers_simple = seekers[
                seekers["par_consecutivo_f1"]
                & seekers["estado_simplificado_t"].notna()
                & seekers["estado_simplificado_f1"].notna()
            ].copy()
            if not seekers_simple.empty:
                count_matrix, prob_matrix = simplified_transition_matrices(
                    seekers_simple,
                    "estado_simplificado_t",
                    "estado_simplificado_f1",
                )
                count_matrix.index.name = "Origin (t)"
                count_matrix.columns.name = "Destination (t+1)"
                prob_matrix = prob_matrix.mul(100).round(1)
                prob_matrix.index.name = "Origin (t)"
                prob_matrix.columns.name = "Destination (t+1)"

                pt_count = count_matrix.rename(
                    index=SIMPLIFIED_STATE_LABELS_PT,
                    columns=SIMPLIFIED_STATE_LABELS_PT,
                )
                pt_count.index.name = "Origem (t)"
                pt_count.columns.name = "Destino (t+1)"
                pt_prob = prob_matrix.rename(
                    index=SIMPLIFIED_STATE_LABELS_PT,
                    columns=SIMPLIFIED_STATE_LABELS_PT,
                )
                pt_prob.index.name = "Origem (t)"
                pt_prob.columns.name = "Destino (t+1)"

                save_table(
                    count_matrix.astype(int),
                    "B8_contagem_transicoes_simplificadas",
                    tables_dir,
                    merged,
                    seekers_simple,
                    pt_df=pt_count.astype(int),
                )
                save_table(
                    prob_matrix,
                    "B9_probabilidade_transicoes_simplificadas",
                    tables_dir,
                    merged,
                    seekers_simple,
                    pt_df=pt_prob,
                )

                state_pairs = list(SIMPLIFIED_STATE_LABELS_PT.items())

                fig, ax = plt.subplots(figsize=(7.4, 5.8))
                sns.heatmap(
                    prob_matrix,
                    annot=True,
                    fmt=".1f",
                    cmap="Blues",
                    linewidths=0.5,
                    linecolor="grey",
                    annot_kws={"size": 9},
                    ax=ax,
                    cbar_kws={"label": "Transition probability (%)"},
                )
                ax.set_xlabel("State at t+1")
                ax.set_ylabel("State at t")
                ax.set_title(f"Simplified 3-state transitions for exam-searchers\nPNADC {year_label}")
                ax.tick_params(axis="x", rotation=20)
                ax.tick_params(axis="y", rotation=0)
                plt.tight_layout()
                save_figure(
                    "B3_heatmap_transicoes_simplificadas",
                    figures_dir,
                    merged,
                    seekers_simple,
                    category_pairs=state_pairs,
                )

                fig, ax = plt.subplots(figsize=(10.2, 6.7))
                draw_simplified_sankey(
                    count_matrix,
                    title=f"Simplified 3-state flows for exam-searchers\nPNADC {year_label}",
                    ax=ax,
                )
                plt.tight_layout()
                save_figure(
                    "B4_fluxos_transicoes_simplificadas",
                    figures_dir,
                    merged,
                    seekers_simple,
                    category_pairs=state_pairs,
                )

            if "servidor_publico_f1" in seekers.columns:
                winners = seekers[seekers["servidor_publico_f1"] == 1].copy()
                losers = seekers[seekers["next_status"] == "Still searching via exam"].copy()
                rows = []
                rows_w = []
                future_inc_col = income_column(seekers, suffix="_f1")
                future_inc_col_w = income_column(seekers, suffix="_f1", winsorized=True)
                for group_name, frame in [("Became public servant", winners), ("Still searching", losers)]:
                    row = {"group": group_name, "n": len(frame)}
                    row_w = {"group": group_name, "n": len(frame)}
                    for column in ["idade", "anos_estudo"]:
                        if column in frame.columns:
                            row[f"{column}_mean"] = frame[column].mean()
                            row_w[f"{column}_mean"] = frame[column].mean()
                    if future_inc_col and future_inc_col in frame.columns:
                        row["new_earnings_mean"] = frame[future_inc_col].mean()
                    if lag_inc_col and lag_inc_col in frame.columns:
                        row["prior_earnings_mean"] = frame[lag_inc_col].mean()
                    if future_inc_col_w and future_inc_col_w in frame.columns:
                        row_w["new_earnings_mean"] = frame[future_inc_col_w].mean()
                    if lag_inc_col_w and lag_inc_col_w in frame.columns:
                        row_w["prior_earnings_mean"] = frame[lag_inc_col_w].mean()
                    rows.append(row)
                    rows_w.append(row_w)
                if rows:
                    save_table(pd.DataFrame(rows), "B6_aprovados_vs_continuando", tables_dir, merged, pd.concat([winners, losers]), index=False)
                if rows_w and lag_inc_col_w and future_inc_col_w:
                    save_table(
                        pd.DataFrame(rows_w),
                        "B6_aprovados_vs_continuando_winsor",
                        tables_dir,
                        merged,
                        pd.concat([winners, losers]),
                        index=False,
                        extra_note=WINSOR_NOTE,
                        extra_note_pt=WINSOR_NOTE_PT,
                    )

                if not winners.empty and lag_inc_col and future_inc_col and all(col in winners.columns for col in [lag_inc_col, future_inc_col]):
                    before = winners[lag_inc_col].dropna()
                    after = winners[future_inc_col].dropna()
                    if len(before) > 2 and len(after) > 2:
                        fig, ax = plt.subplots(figsize=(8, 4.8))
                        ax.hist(before, bins=20, alpha=0.6, color=GREY, label="Before (t-1)", density=True)
                        ax.hist(after, bins=20, alpha=0.6, color=GREEN, label="After (t+1)", density=True)
                        ax.axvline(before.median(), color=GREY, linestyle="--", label=f"Median before: R${before.median():,.0f}")
                        ax.axvline(after.median(), color=GREEN, linestyle="--", label=f"Median after: R${after.median():,.0f}")
                        ax.set_xlabel("Monthly earnings (R$)")
                        ax.set_title(f"Earnings before and after entering public service\n(successful cases, {year_label})")
                        ax.legend(fontsize=8)
                        plt.tight_layout()
                        save_figure("B2_renda_antes_depois_aprovados", figures_dir, merged, winners)

                        fig, ax = plt.subplots(figsize=(8, 4.8))
                        log_before = positive_log(before)
                        log_after = positive_log(after)
                        ax.hist(log_before.dropna(), bins=20, alpha=0.6, color=GREY, label="Log before (t-1)", density=True)
                        ax.hist(log_after.dropna(), bins=20, alpha=0.6, color=GREEN, label="Log after (t+1)", density=True)
                        ax.axvline(log_before.median(), color=GREY, linestyle="--", label=f"Median log before: {log_before.median():.2f}")
                        ax.axvline(log_after.median(), color=GREEN, linestyle="--", label=f"Median log after: {log_after.median():.2f}")
                        ax.set_xlabel("Log monthly earnings")
                        ax.set_title(f"Log earnings before and after entering public service\n(successful cases, {year_label})")
                        ax.legend(fontsize=8)
                        plt.tight_layout()
                        save_figure("B2_renda_antes_depois_aprovados_log", figures_dir, merged, winners)

                if not winners.empty and lag_inc_col_w and future_inc_col_w and all(col in winners.columns for col in [lag_inc_col_w, future_inc_col_w]):
                    before_w = winners[lag_inc_col_w].dropna()
                    after_w = winners[future_inc_col_w].dropna()
                    if len(before_w) > 2 and len(after_w) > 2:
                        fig, ax = plt.subplots(figsize=(8, 4.8))
                        ax.hist(before_w, bins=20, alpha=0.6, color=GREY, label="Before (t-1)", density=True)
                        ax.hist(after_w, bins=20, alpha=0.6, color=GREEN, label="After (t+1)", density=True)
                        ax.axvline(before_w.median(), color=GREY, linestyle="--", label=f"Median before: R${before_w.median():,.0f}")
                        ax.axvline(after_w.median(), color=GREEN, linestyle="--", label=f"Median after: R${after_w.median():,.0f}")
                        ax.set_xlabel("Monthly earnings (R$)")
                        ax.set_title(f"Earnings before and after entering public service\n(successful cases, {year_label})")
                        ax.legend(fontsize=8)
                        plt.tight_layout()
                        save_figure(
                            "B2_renda_antes_depois_aprovados_winsor",
                            figures_dir,
                            merged,
                            winners,
                            extra_note=WINSOR_NOTE,
                            extra_note_pt=WINSOR_NOTE_PT,
                        )

                        fig, ax = plt.subplots(figsize=(8, 4.8))
                        log_before_w = positive_log(before_w)
                        log_after_w = positive_log(after_w)
                        ax.hist(log_before_w.dropna(), bins=20, alpha=0.6, color=GREY, label="Log before (t-1)", density=True)
                        ax.hist(log_after_w.dropna(), bins=20, alpha=0.6, color=GREEN, label="Log after (t+1)", density=True)
                        ax.axvline(log_before_w.median(), color=GREY, linestyle="--", label=f"Median log before: {log_before_w.median():.2f}")
                        ax.axvline(log_after_w.median(), color=GREEN, linestyle="--", label=f"Median log after: {log_after_w.median():.2f}")
                        ax.set_xlabel("Log monthly earnings")
                        ax.set_title(f"Log earnings before and after entering public service\n(successful cases, {year_label})")
                        ax.legend(fontsize=8)
                        plt.tight_layout()
                        save_figure(
                            "B2_renda_antes_depois_aprovados_log_winsor",
                            figures_dir,
                            merged,
                            winners,
                            extra_note=WINSOR_NOTE,
                            extra_note_pt=WINSOR_NOTE_PT,
                        )

            gave_up = seekers[seekers["next_status"].isin(["Formal private job", "Family auxiliary work", "Informal job", "Unemployed", "Out of labor force"])].copy()
            if not gave_up.empty:
                values = maybe_decode(gave_up, "posicao_emprego_label_f1", label_maps)
                if not values.empty:
                    table = values.value_counts(dropna=False).rename_axis("destination_pt").reset_index(name="n")
                    table["destination_en"], _ = translate_labels(table["destination_pt"].tolist(), max_len=34)
                    table["pct"] = (table["n"] / len(gave_up) * 100).round(1)
                    save_table(table[["destination_en", "n", "pct"]], "B7_destino_quem_desistiu", tables_dir, merged, gave_up, index=False)

    if "servidor_publico" in merged.columns and "ocupado" in merged.columns:
        public = merged[merged["servidor_publico"] == 1].copy()
        private = merged[(merged["servidor_publico"] != 1) & (merged["ocupado"] == 1)].copy()

        rows = []
        rows_w = []
        for group_name, frame in [("Public servant", public), ("Private worker", private)]:
            row = {"group": group_name, "n": len(frame)}
            row_w = {"group": group_name, "n": len(frame)}
            for column in ["idade", "anos_estudo"]:
                if column in frame.columns:
                    row[f"{column}_mean"] = round(frame[column].mean(), 1)
                    row_w[f"{column}_mean"] = round(frame[column].mean(), 1)
            for column in ["sexo", "raca_cor", "nivel_instrucao"]:
                use = best_column(frame, column)
                if use in frame.columns:
                    top = frame[use].value_counts(normalize=True).mul(100).head(3)
                    for key, value in top.items():
                        row[f"{column}_{key}_pct"] = round(value, 1)
                        row_w[f"{column}_{key}_pct"] = round(value, 1)
            inc_col = income_column(frame)
            if inc_col:
                row["earnings_mean"] = round(frame[inc_col].mean(), 0)
                row["earnings_median"] = round(frame[inc_col].median(), 0)
            inc_col_w_frame = income_column(frame, winsorized=True)
            if inc_col_w_frame:
                row_w["earnings_mean"] = round(frame[inc_col_w_frame].mean(), 0)
                row_w["earnings_median"] = round(frame[inc_col_w_frame].median(), 0)
            rows.append(row)
            rows_w.append(row_w)
        save_table(pd.DataFrame(rows), "C1_perfil_publico_vs_privado", tables_dir, merged, pd.concat([public, private]), index=False)
        save_table(
            pd.DataFrame(rows_w),
            "C1_perfil_publico_vs_privado_winsor",
            tables_dir,
            merged,
            pd.concat([public, private]),
            index=False,
            extra_note=WINSOR_NOTE,
            extra_note_pt=WINSOR_NOTE_PT,
        )

        inc_col = income_column(merged)
        inc_col_w = income_column(merged, winsorized=True)
        ni_col = best_column(merged, "nivel_instrucao")
        if inc_col and ni_col in merged.columns:
            both = merged[merged["ocupado"] == 1].copy()
            both = both[both[inc_col].notna() & both[ni_col].notna()]
            both["sector"] = both["servidor_publico"].map({1: "Public", 0: "Private"}).fillna("Private")
            med = both.groupby([ni_col, "sector"])[inc_col].median().reset_index()
            med = med.rename(columns={ni_col: "education_pt", inc_col: "median_earnings"})
            med["education_en"], pairs = translate_labels(med["education_pt"].tolist(), max_len=28)
            save_table(med[["education_en", "sector", "median_earnings"]], "C2_renda_por_instrucao_setor", tables_dir, merged, both, index=False)

            piv = med.pivot(index="education_en", columns="sector", values="median_earnings").fillna(0)
            fig, ax = plt.subplots(figsize=(11.5, 5.8))
            piv.plot(kind="bar", ax=ax, color=[BLUE, RED], width=0.7)
            ax.set_xlabel("")
            ax.set_ylabel("Median earnings (R$)")
            ax.set_title(f"Median earnings by education level\nPublic vs. private sector (PNADC {year_label})")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
            ax.tick_params(axis="x", rotation=35, labelsize=9)
            ax.legend(title="Sector")
            plt.tight_layout()
            save_figure("C1_renda_instrucao_publico_privado", figures_dir, merged, both, category_pairs=pairs)

            med_log = both.copy()
            med_log["log_earnings"] = positive_log(med_log[inc_col])
            med_log = med_log.groupby([ni_col, "sector"])["log_earnings"].median().reset_index()
            med_log = med_log.rename(columns={ni_col: "education_pt", "log_earnings": "median_log_earnings"})
            med_log["education_en"], pairs_log = translate_labels(med_log["education_pt"].tolist(), max_len=28)
            save_table(med_log[["education_en", "sector", "median_log_earnings"]], "C2_renda_por_instrucao_setor_log", tables_dir, merged, both, index=False)

            piv_log = med_log.pivot(index="education_en", columns="sector", values="median_log_earnings").fillna(0)
            fig, ax = plt.subplots(figsize=(11.5, 5.8))
            piv_log.plot(kind="bar", ax=ax, color=[BLUE, RED], width=0.7)
            ax.set_xlabel("")
            ax.set_ylabel("Median log earnings")
            ax.set_title(f"Median log earnings by education level\nPublic vs. private sector (PNADC {year_label})")
            ax.tick_params(axis="x", rotation=35, labelsize=9)
            ax.legend(title="Sector")
            plt.tight_layout()
            save_figure("C1_renda_instrucao_publico_privado_log", figures_dir, merged, both, category_pairs=pairs_log)

            prem = both.groupby([ni_col, "servidor_publico"])[inc_col].median().unstack("servidor_publico")
            if 0 in prem.columns and 1 in prem.columns:
                prem.columns = ["Private", "Public"]
                prem["Public_premium_pct"] = ((prem["Public"] / prem["Private"]) - 1).mul(100).round(1)
                prem_out = prem.reset_index().rename(columns={ni_col: "education_pt"})
                prem_out["education_en"], pairs = translate_labels(prem_out["education_pt"].tolist(), max_len=28)
                save_table(prem_out[["education_en", "Private", "Public", "Public_premium_pct"]], "C8_premio_salarial_publico_por_instrucao", tables_dir, merged, both, index=False)

                fig, ax = plt.subplots(figsize=(10.5, 5))
                plot_df = prem_out.dropna(subset=["Public_premium_pct"]).copy()
                ax.bar(plot_df["education_en"], plot_df["Public_premium_pct"], color=BLUE)
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax.set_ylabel("Public wage premium (%)")
                ax.set_title(f"Public-sector wage premium by education\n(Public median / private median - 1, {year_label})")
                ax.tick_params(axis="x", rotation=35, labelsize=9)
                plt.tight_layout()
                save_figure("C5_premio_salarial_por_instrucao", figures_dir, merged, both, category_pairs=pairs)

        if inc_col_w and inc_col_w != inc_col and ni_col in merged.columns:
            both_w = merged[merged["ocupado"] == 1].copy()
            both_w = both_w[both_w[inc_col_w].notna() & both_w[ni_col].notna()]
            both_w["sector"] = both_w["servidor_publico"].map({1: "Public", 0: "Private"}).fillna("Private")
            med_w = both_w.groupby([ni_col, "sector"])[inc_col_w].median().reset_index()
            med_w = med_w.rename(columns={ni_col: "education_pt", inc_col_w: "median_earnings"})
            med_w["education_en"], pairs_w = translate_labels(med_w["education_pt"].tolist(), max_len=28)
            save_table(
                med_w[["education_en", "sector", "median_earnings"]],
                "C2_renda_por_instrucao_setor_winsor",
                tables_dir,
                merged,
                both_w,
                index=False,
                extra_note=WINSOR_NOTE,
                extra_note_pt=WINSOR_NOTE_PT,
            )

            piv_w = med_w.pivot(index="education_en", columns="sector", values="median_earnings").fillna(0)
            fig, ax = plt.subplots(figsize=(11.5, 5.8))
            piv_w.plot(kind="bar", ax=ax, color=[BLUE, RED], width=0.7)
            ax.set_xlabel("")
            ax.set_ylabel("Median earnings (R$)")
            ax.set_title(f"Median earnings by education level\nPublic vs. private sector (PNADC {year_label})")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:,.0f}"))
            ax.tick_params(axis="x", rotation=35, labelsize=9)
            ax.legend(title="Sector")
            plt.tight_layout()
            save_figure(
                "C1_renda_instrucao_publico_privado_winsor",
                figures_dir,
                merged,
                both_w,
                category_pairs=pairs_w,
                extra_note=WINSOR_NOTE,
                extra_note_pt=WINSOR_NOTE_PT,
            )

            med_log_w = both_w.copy()
            med_log_w["log_earnings"] = positive_log(med_log_w[inc_col_w])
            med_log_w = med_log_w.groupby([ni_col, "sector"])["log_earnings"].median().reset_index()
            med_log_w = med_log_w.rename(columns={ni_col: "education_pt", "log_earnings": "median_log_earnings"})
            med_log_w["education_en"], pairs_log_w = translate_labels(med_log_w["education_pt"].tolist(), max_len=28)
            save_table(
                med_log_w[["education_en", "sector", "median_log_earnings"]],
                "C2_renda_por_instrucao_setor_log_winsor",
                tables_dir,
                merged,
                both_w,
                index=False,
                extra_note=WINSOR_NOTE,
                extra_note_pt=WINSOR_NOTE_PT,
            )

            piv_log_w = med_log_w.pivot(index="education_en", columns="sector", values="median_log_earnings").fillna(0)
            fig, ax = plt.subplots(figsize=(11.5, 5.8))
            piv_log_w.plot(kind="bar", ax=ax, color=[BLUE, RED], width=0.7)
            ax.set_xlabel("")
            ax.set_ylabel("Median log earnings")
            ax.set_title(f"Median log earnings by education level\nPublic vs. private sector (PNADC {year_label})")
            ax.tick_params(axis="x", rotation=35, labelsize=9)
            ax.legend(title="Sector")
            plt.tight_layout()
            save_figure(
                "C1_renda_instrucao_publico_privado_log_winsor",
                figures_dir,
                merged,
                both_w,
                category_pairs=pairs_log_w,
                extra_note=WINSOR_NOTE,
                extra_note_pt=WINSOR_NOTE_PT,
            )

            prem_w = both_w.groupby([ni_col, "servidor_publico"])[inc_col_w].median().unstack("servidor_publico")
            if 0 in prem_w.columns and 1 in prem_w.columns:
                prem_w.columns = ["Private", "Public"]
                prem_w["Public_premium_pct"] = ((prem_w["Public"] / prem_w["Private"]) - 1).mul(100).round(1)
                prem_out_w = prem_w.reset_index().rename(columns={ni_col: "education_pt"})
                prem_out_w["education_en"], pairs_prem_w = translate_labels(prem_out_w["education_pt"].tolist(), max_len=28)
                save_table(
                    prem_out_w[["education_en", "Private", "Public", "Public_premium_pct"]],
                    "C8_premio_salarial_publico_por_instrucao_winsor",
                    tables_dir,
                    merged,
                    both_w,
                    index=False,
                    extra_note=WINSOR_NOTE,
                    extra_note_pt=WINSOR_NOTE_PT,
                )

                fig, ax = plt.subplots(figsize=(10.5, 5))
                plot_df_w = prem_out_w.dropna(subset=["Public_premium_pct"]).copy()
                ax.bar(plot_df_w["education_en"], plot_df_w["Public_premium_pct"], color=BLUE)
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax.set_ylabel("Public wage premium (%)")
                ax.set_title(f"Public-sector wage premium by education\n(Public median / private median - 1, {year_label})")
                ax.tick_params(axis="x", rotation=35, labelsize=9)
                plt.tight_layout()
                save_figure(
                    "C5_premio_salarial_por_instrucao_winsor",
                    figures_dir,
                    merged,
                    both_w,
                    category_pairs=pairs_prem_w,
                    extra_note=WINSOR_NOTE,
                    extra_note_pt=WINSOR_NOTE_PT,
                )

        if "idade" in merged.columns:
            used = merged[merged["idade"].notna() & (merged["ocupado"] == 1)].copy()
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.hist(private["idade"].dropna(), bins=range(14, 75), density=True, alpha=0.5, color=RED, label="Private")
            ax.hist(public["idade"].dropna(), bins=range(14, 75), density=True, alpha=0.6, color=BLUE, label="Public")
            ax.set_xlabel("Age")
            ax.set_ylabel("Density")
            ax.set_title("Age distribution: public vs. private sector")
            ax.legend()
            plt.tight_layout()
            save_figure("C2_idade_publico_privado", figures_dir, merged, used)

        if inc_col and "horas_habituais_principal" in merged.columns:
            hours = merged[merged["ocupado"] == 1].copy()
            hours = hours[hours[inc_col].notna() & hours["horas_habituais_principal"].notna()]
            hours["sector"] = hours["servidor_publico"].map({1: "Public", 0: "Private"}).fillna("Private")
            stats = hours.groupby("sector").agg(
                earnings_mean=(inc_col, "mean"),
                earnings_median=(inc_col, "median"),
                hours_mean=("horas_habituais_principal", "mean"),
                n=("id_domicilio", "count"),
            ).reset_index()
            save_table(stats, "C3_renda_horas_setor", tables_dir, merged, hours, index=False)

            stats_log = hours.copy()
            stats_log["log_earnings"] = positive_log(stats_log[inc_col])
            stats_log = stats_log.groupby("sector").agg(
                log_earnings_mean=("log_earnings", "mean"),
                log_earnings_median=("log_earnings", "median"),
                hours_mean=("horas_habituais_principal", "mean"),
                n=("id_domicilio", "count"),
            ).reset_index()
            save_table(stats_log, "C3_renda_horas_setor_log", tables_dir, merged, hours, index=False)

            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
            for ax, var, ylabel in [
                (axes[0], inc_col, "Habitual earnings (R$)"),
                (axes[1], "horas_habituais_principal", "Hours per week"),
            ]:
                plot_data = [hours[hours["sector"] == sector][var].dropna() for sector in ["Public", "Private"]]
                ax.boxplot(plot_data, tick_labels=["Public", "Private"], medianprops={"color": "black", "linewidth": 2})
                ax.set_ylabel(ylabel)
            axes[0].set_title("Earnings")
            axes[1].set_title("Hours")
            fig.suptitle("Public servants vs. private workers", fontsize=12)
            plt.tight_layout()
            save_figure("C3_renda_horas_publico_privado", figures_dir, merged, hours)

            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
            for ax, var, ylabel in [
                (axes[0], "log_earnings", "Log habitual earnings"),
                (axes[1], "horas_habituais_principal", "Hours per week"),
            ]:
                plot_frame = hours.copy()
                plot_frame["log_earnings"] = positive_log(plot_frame[inc_col])
                plot_data = [plot_frame[plot_frame["sector"] == sector][var].dropna() for sector in ["Public", "Private"]]
                ax.boxplot(plot_data, tick_labels=["Public", "Private"], medianprops={"color": "black", "linewidth": 2})
                ax.set_ylabel(ylabel)
            axes[0].set_title("Log earnings")
            axes[1].set_title("Hours")
            fig.suptitle("Public servants vs. private workers", fontsize=12)
            plt.tight_layout()
            save_figure("C3_renda_horas_publico_privado_log", figures_dir, merged, hours)

            wages = merged[merged["ocupado"] == 1].copy()
            wages = wages[wages[inc_col].notna() & (wages[inc_col] > 0)].copy()
            wages["sector"] = wages["servidor_publico"].map({1: "Public", 0: "Private"}).fillna("Private")
            wages = add_period_labels(wages)

            private_pct = percentile_table_by_period(wages[wages["sector"] == "Private"], inc_col)
            if not private_pct.empty:
                save_table(
                    private_pct,
                    "C9_percentis_renda_privado_trimestre",
                    tables_dir,
                    merged,
                    wages[wages["sector"] == "Private"],
                    index=False,
                )
                if plot_log_distribution_by_period(
                    wages[wages["sector"] == "Private"],
                    inc_col,
                    sector_label="Private",
                    year_label=year_label,
                ):
                    save_figure(
                        "C9_distribuicao_log_renda_privado_trimestre",
                        figures_dir,
                        merged,
                        wages[wages["sector"] == "Private"],
                    )

            public_pct = percentile_table_by_period(wages[wages["sector"] == "Public"], inc_col)
            if not public_pct.empty:
                save_table(
                    public_pct,
                    "C10_percentis_renda_publico_trimestre",
                    tables_dir,
                    merged,
                    wages[wages["sector"] == "Public"],
                    index=False,
                )
                if plot_log_distribution_by_period(
                    wages[wages["sector"] == "Public"],
                    inc_col,
                    sector_label="Public",
                    year_label=year_label,
                ):
                    save_figure(
                        "C10_distribuicao_log_renda_publico_trimestre",
                        figures_dir,
                        merged,
                        wages[wages["sector"] == "Public"],
                    )

            new_jobs = merged.copy()
            new_jobs["is_new_job_proxy"] = new_job_proxy(new_jobs)
            new_jobs = new_jobs[new_jobs["is_new_job_proxy"] & new_jobs[inc_col].notna() & (new_jobs[inc_col] > 0)].copy()
            if not new_jobs.empty:
                new_jobs = add_period_labels(new_jobs)
                new_jobs_pct = percentile_table_by_period(new_jobs, inc_col)
                save_table(
                    new_jobs_pct,
                    "C11_percentis_renda_novos_empregos_trimestre",
                    tables_dir,
                    merged,
                    new_jobs,
                    index=False,
                )

            if inc_col_w and inc_col_w != inc_col:
                hours_w = merged[merged["ocupado"] == 1].copy()
                hours_w = hours_w[hours_w[inc_col_w].notna() & hours_w["horas_habituais_principal"].notna()]
                hours_w["sector"] = hours_w["servidor_publico"].map({1: "Public", 0: "Private"}).fillna("Private")
                stats_w = hours_w.groupby("sector").agg(
                    earnings_mean=(inc_col_w, "mean"),
                    earnings_median=(inc_col_w, "median"),
                    hours_mean=("horas_habituais_principal", "mean"),
                    n=("id_domicilio", "count"),
                ).reset_index()
                save_table(
                    stats_w,
                    "C3_renda_horas_setor_winsor",
                    tables_dir,
                    merged,
                    hours_w,
                    index=False,
                    extra_note=WINSOR_NOTE,
                    extra_note_pt=WINSOR_NOTE_PT,
                )

                stats_log_w = hours_w.copy()
                stats_log_w["log_earnings"] = positive_log(stats_log_w[inc_col_w])
                stats_log_w = stats_log_w.groupby("sector").agg(
                    log_earnings_mean=("log_earnings", "mean"),
                    log_earnings_median=("log_earnings", "median"),
                    hours_mean=("horas_habituais_principal", "mean"),
                    n=("id_domicilio", "count"),
                ).reset_index()
                save_table(
                    stats_log_w,
                    "C3_renda_horas_setor_log_winsor",
                    tables_dir,
                    merged,
                    hours_w,
                    index=False,
                    extra_note=WINSOR_NOTE,
                    extra_note_pt=WINSOR_NOTE_PT,
                )

                fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
                for ax, var, ylabel in [
                    (axes[0], inc_col_w, "Habitual earnings (R$)"),
                    (axes[1], "horas_habituais_principal", "Hours per week"),
                ]:
                    plot_data = [hours_w[hours_w["sector"] == sector][var].dropna() for sector in ["Public", "Private"]]
                    ax.boxplot(plot_data, tick_labels=["Public", "Private"], medianprops={"color": "black", "linewidth": 2})
                    ax.set_ylabel(ylabel)
                axes[0].set_title("Earnings")
                axes[1].set_title("Hours")
                fig.suptitle("Public servants vs. private workers", fontsize=12)
                plt.tight_layout()
                save_figure(
                    "C3_renda_horas_publico_privado_winsor",
                    figures_dir,
                    merged,
                    hours_w,
                    extra_note=WINSOR_NOTE,
                    extra_note_pt=WINSOR_NOTE_PT,
                )

                fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
                for ax, var, ylabel in [
                    (axes[0], "log_earnings", "Log habitual earnings"),
                    (axes[1], "horas_habituais_principal", "Hours per week"),
                ]:
                    plot_frame_w = hours_w.copy()
                    plot_frame_w["log_earnings"] = positive_log(plot_frame_w[inc_col_w])
                    plot_data = [plot_frame_w[plot_frame_w["sector"] == sector][var].dropna() for sector in ["Public", "Private"]]
                    ax.boxplot(plot_data, tick_labels=["Public", "Private"], medianprops={"color": "black", "linewidth": 2})
                    ax.set_ylabel(ylabel)
                axes[0].set_title("Log earnings")
                axes[1].set_title("Hours")
                fig.suptitle("Public servants vs. private workers", fontsize=12)
                plt.tight_layout()
                save_figure(
                    "C3_renda_horas_publico_privado_log_winsor",
                    figures_dir,
                    merged,
                    hours_w,
                    extra_note=WINSOR_NOTE,
                    extra_note_pt=WINSOR_NOTE_PT,
                )

                wages_w = merged[merged["ocupado"] == 1].copy()
                wages_w = wages_w[wages_w[inc_col_w].notna() & (wages_w[inc_col_w] > 0)].copy()
                wages_w["sector"] = wages_w["servidor_publico"].map({1: "Public", 0: "Private"}).fillna("Private")
                wages_w = add_period_labels(wages_w)

                private_pct_w = percentile_table_by_period(wages_w[wages_w["sector"] == "Private"], inc_col_w)
                if not private_pct_w.empty:
                    save_table(
                        private_pct_w,
                        "C9_percentis_renda_privado_trimestre_winsor",
                        tables_dir,
                        merged,
                        wages_w[wages_w["sector"] == "Private"],
                        index=False,
                        extra_note=WINSOR_NOTE,
                        extra_note_pt=WINSOR_NOTE_PT,
                    )
                    if plot_log_distribution_by_period(
                        wages_w[wages_w["sector"] == "Private"],
                        inc_col_w,
                        sector_label="Private",
                        year_label=year_label,
                    ):
                        save_figure(
                            "C9_distribuicao_log_renda_privado_trimestre_winsor",
                            figures_dir,
                            merged,
                            wages_w[wages_w["sector"] == "Private"],
                            extra_note=WINSOR_NOTE,
                            extra_note_pt=WINSOR_NOTE_PT,
                        )

                public_pct_w = percentile_table_by_period(wages_w[wages_w["sector"] == "Public"], inc_col_w)
                if not public_pct_w.empty:
                    save_table(
                        public_pct_w,
                        "C10_percentis_renda_publico_trimestre_winsor",
                        tables_dir,
                        merged,
                        wages_w[wages_w["sector"] == "Public"],
                        index=False,
                        extra_note=WINSOR_NOTE,
                        extra_note_pt=WINSOR_NOTE_PT,
                    )
                    if plot_log_distribution_by_period(
                        wages_w[wages_w["sector"] == "Public"],
                        inc_col_w,
                        sector_label="Public",
                        year_label=year_label,
                    ):
                        save_figure(
                            "C10_distribuicao_log_renda_publico_trimestre_winsor",
                            figures_dir,
                            merged,
                            wages_w[wages_w["sector"] == "Public"],
                            extra_note=WINSOR_NOTE,
                            extra_note_pt=WINSOR_NOTE_PT,
                        )

                new_jobs_w = merged.copy()
                new_jobs_w["is_new_job_proxy"] = new_job_proxy(new_jobs_w)
                new_jobs_w = new_jobs_w[new_jobs_w["is_new_job_proxy"] & new_jobs_w[inc_col_w].notna() & (new_jobs_w[inc_col_w] > 0)].copy()
                if not new_jobs_w.empty:
                    new_jobs_w = add_period_labels(new_jobs_w)
                    new_jobs_pct_w = percentile_table_by_period(new_jobs_w, inc_col_w)
                    save_table(
                        new_jobs_pct_w,
                        "C11_percentis_renda_novos_empregos_trimestre_winsor",
                        tables_dir,
                        merged,
                        new_jobs_w,
                        index=False,
                        extra_note=WINSOR_NOTE,
                        extra_note_pt=WINSOR_NOTE_PT,
                    )

        for lag_col, name in [
            ("posicao_emprego_label_t1", "C5_origin_of_public_servants_position_t1"),
            ("condicao_ocupacao_label_t1", "C6_origin_of_public_servants_status_t1"),
        ]:
            values = maybe_decode(public, lag_col, label_maps)
            if values.empty:
                continue
            table = values.value_counts(dropna=False).rename_axis("status_pt").reset_index(name="n")
            table["status_en"], _ = translate_labels(table["status_pt"].tolist(), max_len=34)
            table["pct"] = (table["n"] / len(public) * 100).round(1)
            save_table(table[["status_en", "n", "pct"]], name, tables_dir, merged, public, index=False)

        if "servidor_publico_t1" in public.columns:
            new_public = public[(public["servidor_publico_t1"] != 1) & public["servidor_publico_t1"].notna()].copy()
            if not new_public.empty:
                values = maybe_decode(new_public, "posicao_emprego_label_t1", label_maps)
                if not values.empty:
                    table = values.value_counts(dropna=False).rename_axis("origin_pt").reset_index(name="n")
                    table["origin_en"], pairs = translate_labels(table["origin_pt"].tolist(), max_len=34)
                    table["pct"] = (table["n"] / len(new_public) * 100).round(1)
                    save_table(table[["origin_en", "n", "pct"]], "C6_novos_servidores_origem", tables_dir, merged, new_public, index=False)

                    fig, ax = plt.subplots(figsize=(10.2, 4.8))
                    plot_table = table.head(8).copy()
                    ax.barh(plot_table["origin_en"][::-1], plot_table["n"][::-1], color=BLUE)
                    ax.set_xlabel("Number of public servants")
                    ax.set_title("Where did newly public workers come from?\nEmployment position in the prior quarter (t-1)")
                    plt.tight_layout()
                    save_figure("C4_origem_servidores", figures_dir, merged, new_public, category_pairs=pairs)

                if "buscando_via_concurso_t1" in new_public.columns:
                    via_exam = int((new_public["buscando_via_concurso_t1"] == 1).sum())
                    table = pd.DataFrame(
                        [
                            {
                                "group": "New public servants",
                                "n_total": len(new_public),
                                "came_from_exam_search": via_exam,
                                "pct_from_exam_search": round(100 * via_exam / len(new_public), 1),
                            }
                        ]
                    )
                    save_table(table, "C7_novos_servidores_via_concurso", tables_dir, merged, new_public, index=False)

    merged["estado_simplificado_t"] = simplified_state(merged)
    merged["estado_simplificado_f1"] = simplified_state(merged, suffix="_f1")
    merged["par_consecutivo_f1"] = next_consecutive_observation(merged)

    merged["figure3_estado_detalhado_t"] = figure3_detailed_state(merged)
    merged["figure3_estado_detalhado_f1"] = figure3_detailed_state(merged, suffix="_f1")
    merged["figure3_estado_colapsado_t"] = figure3_collapsed_state(merged)
    merged["figure3_estado_colapsado_f1"] = figure3_collapsed_state(merged, suffix="_f1")

    figure3_detailed_panel = merged[
        merged["par_consecutivo_f1"]
        & merged["figure3_estado_detalhado_t"].notna()
        & merged["figure3_estado_detalhado_f1"].notna()
    ].copy()
    if not figure3_detailed_panel.empty:
        count_matrix_d, prob_matrix_d = transition_matrices(
            figure3_detailed_panel,
            "figure3_estado_detalhado_t",
            "figure3_estado_detalhado_f1",
            FIGURE3_DETAILED_STATES,
        )
        prob_matrix_d = prob_matrix_d.mul(100).round(1)
        prob_matrix_d.index.name = "Employment status at t"
        prob_matrix_d.columns.name = "Employment status at t+1"
        pt_prob_d = prob_matrix_d.rename(index=FIGURE3_DETAILED_LABELS_PT, columns=FIGURE3_DETAILED_LABELS_PT)
        pt_prob_d.index.name = "Status ocupacional em t"
        pt_prob_d.columns.name = "Status ocupacional em t+1"
        save_table(
            prob_matrix_d,
            "figure_1_transition_probabilities_table",
            tables_dir,
            merged,
            figure3_detailed_panel,
            pt_df=pt_prob_d,
        )

        detailed_pairs = list(FIGURE3_DETAILED_LABELS_PT.items())
        fig, ax = plt.subplots(figsize=(9.6, 7.2))
        sns.heatmap(
            prob_matrix_d,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            linewidths=0.5,
            linecolor="grey",
            annot_kws={"size": 8.5},
            ax=ax,
            cbar_kws={"label": "Transition probability (%)"},
        )
        ax.set_xlabel("Employment status at t+1")
        ax.set_ylabel("Employment status at t")
        ax.set_title("")
        ax.tick_params(axis="x", rotation=30, labelsize=8.5)
        ax.tick_params(axis="y", rotation=0, labelsize=8.5)
        plt.tight_layout()
        save_figure(
            "figure_1_transition_probabilities",
            figures_dir,
            merged,
            figure3_detailed_panel,
            category_pairs=detailed_pairs,
            export_pdf=True,
            include_notes=False,
        )

    figure3_collapsed_panel = merged[
        merged["par_consecutivo_f1"]
        & merged["figure3_estado_colapsado_t"].notna()
        & merged["figure3_estado_colapsado_f1"].notna()
    ].copy()
    if not figure3_collapsed_panel.empty:
        count_matrix_c, prob_matrix_c = transition_matrices(
            figure3_collapsed_panel,
            "figure3_estado_colapsado_t",
            "figure3_estado_colapsado_f1",
            FIGURE3_COLLAPSED_STATES,
        )
        prob_matrix_c = prob_matrix_c.mul(100).round(1)
        prob_matrix_c.index.name = "Employment status at t"
        prob_matrix_c.columns.name = "Employment status at t+1"
        pt_prob_c = prob_matrix_c.rename(index=FIGURE3_COLLAPSED_LABELS_PT, columns=FIGURE3_COLLAPSED_LABELS_PT)
        pt_prob_c.index.name = "Status ocupacional em t"
        pt_prob_c.columns.name = "Status ocupacional em t+1"
        save_table(
            prob_matrix_c,
            "figure_1_transition_probabilities_2_table",
            tables_dir,
            merged,
            figure3_collapsed_panel,
            pt_df=pt_prob_c,
        )

        collapsed_pairs = list(FIGURE3_COLLAPSED_LABELS_PT.items())
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        sns.heatmap(
            prob_matrix_c,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            linewidths=0.5,
            linecolor="grey",
            annot_kws={"size": 9},
            ax=ax,
            cbar_kws={"label": "Transition probability (%)"},
        )
        ax.set_xlabel("Employment status at t+1")
        ax.set_ylabel("Employment status at t")
        ax.set_title("")
        ax.tick_params(axis="x", rotation=18, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        plt.tight_layout()
        save_figure(
            "figure_1_transition_probabilities_2",
            figures_dir,
            merged,
            figure3_collapsed_panel,
            category_pairs=collapsed_pairs,
            export_pdf=True,
            include_notes=False,
        )

    simplified_panel = merged[
        merged["par_consecutivo_f1"]
        & merged["estado_simplificado_t"].notna()
        & merged["estado_simplificado_f1"].notna()
    ].copy()
    if not simplified_panel.empty:
        count_matrix, prob_matrix = simplified_transition_matrices(
            simplified_panel,
            "estado_simplificado_t",
            "estado_simplificado_f1",
        )
        count_matrix.index.name = "Origin (t)"
        count_matrix.columns.name = "Destination (t+1)"
        prob_matrix = prob_matrix.mul(100).round(1)
        prob_matrix.index.name = "Origin (t)"
        prob_matrix.columns.name = "Destination (t+1)"

        pt_count = count_matrix.rename(
            index=SIMPLIFIED_STATE_LABELS_PT,
            columns=SIMPLIFIED_STATE_LABELS_PT,
        )
        pt_count.index.name = "Origem (t)"
        pt_count.columns.name = "Destino (t+1)"
        pt_prob = prob_matrix.rename(
            index=SIMPLIFIED_STATE_LABELS_PT,
            columns=SIMPLIFIED_STATE_LABELS_PT,
        )
        pt_prob.index.name = "Origem (t)"
        pt_prob.columns.name = "Destino (t+1)"

        save_table(
            count_matrix.astype(int),
            "D0_contagem_transicoes_simplificadas",
            tables_dir,
            merged,
            simplified_panel,
            pt_df=pt_count.astype(int),
        )
        save_table(
            prob_matrix,
            "D0_probabilidade_transicoes_simplificadas",
            tables_dir,
            merged,
            simplified_panel,
            pt_df=pt_prob,
        )

        state_pairs = list(SIMPLIFIED_STATE_LABELS_PT.items())

        fig, ax = plt.subplots(figsize=(7.8, 6.0))
        sns.heatmap(
            prob_matrix,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            linewidths=0.5,
            linecolor="grey",
            annot_kws={"size": 9},
            ax=ax,
            cbar_kws={"label": "Transition probability (%)"},
        )
        ax.set_xlabel("State at t+1")
        ax.set_ylabel("State at t")
        ax.set_title(f"Simplified 3-state transitions in the pooled panel\nPNADC {year_label}")
        ax.tick_params(axis="x", rotation=20)
        ax.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        save_figure(
            "D0_heatmap_transicoes_simplificadas",
            figures_dir,
            merged,
            simplified_panel,
            category_pairs=state_pairs,
        )

        fig, ax = plt.subplots(figsize=(10.4, 6.8))
        draw_simplified_sankey(
            count_matrix,
            title=f"Simplified 3-state flows in the pooled panel\nPNADC {year_label}",
            ax=ax,
        )
        plt.tight_layout()
        save_figure(
            "D0_fluxos_transicoes_simplificadas",
            figures_dir,
            merged,
            simplified_panel,
            category_pairs=state_pairs,
        )

    return {"rows": len(merged), "columns": len(merged.columns), "tag": tag, "years": years_sorted}
