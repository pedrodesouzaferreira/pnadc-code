"""
3_paper_analysis.py  --  Consolidated analysis for "The Value of Government Jobs"
==============================================================================
Single script that regenerates EVERY figure and table used in the paper
(20260428_LaborPaper.tex) and the presentation (20260428_Labor_Presentation.tex).

It replaces the scattered pipeline (old scripts 4-10 + pnadc_superpc/analysis.py)
with one file. The paper-relevant outputs, and where they used to come from, are:

    figure_1_transition_probabilities.pdf     (Panel A, detailed 7-state)   <- pnadc_superpc/analysis.py
    figure_1_transition_probabilities_2.pdf   (Panel B, collapsed U/Priv/Pub)<- pnadc_superpc/analysis.py
    figure_2_delta_renda_relative_w.pdf       (wage-change % at pub<->priv)  <- 6_analysis.do
    figure_3_renda_USD.pdf                     (wage distributions, USD)      <- 6_analysis.do
    table_1_renda_USD.tex                      (wage summary stats, USD)      <- was a MANUAL .tex; now generated,
                                                                                and FIXED to add U->Public / U->Private
    table_V10E.tex                             (reservation wages & a/beta)   <- 9_GMM.py

Data source
-----------
Reads the CLEANED Stata files directly (they are the canonical cleaned data):
    Cleaned Data/PNADC_limpo_2023.dta
    Cleaned Data/PNADC_limpo_2024.dta
Only the columns needed below are loaded, to keep memory manageable.

Labor-market states (paper mapping, Section "Data: PNADC")
    U = unemployed          : desocupado OR unpaid family worker (trab_familiar_aux)
    R = private sector       : private employee, self-employed, employer, domestic worker
    P = public sector        : public employee
Out-of-labor-force transitions are disregarded (na_pea == 0 dropped for the
transition matrices).

Usage
-----
    python 3_paper_analysis.py                       # all figures + table_1, GMM with default grid
    python 3_paper_analysis.py --no-gmm              # skip the (slow) bootstrap GMM table
    python 3_paper_analysis.py --bootstrap 500 --alphas 0.03 0.06 --lambda0s 0.50 0.80

NOTE (author): this consolidation was written without being run against the full
data (the .dta files are too large for the machine it was drafted on). Please run
it once and eyeball the figures/tables against the previous versions before
deleting the superseded reference scripts (6_analysis.do, 9_GMM.py).
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, gaussian_kde

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # .../PNADC
CLEANED_DIR = ROOT / "Cleaned Data"
F_DIR = ROOT / "Output" / "Figures"
T_DIR = ROOT / "Output" / "Tables"
F_DIR.mkdir(parents=True, exist_ok=True)
T_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2023, 2024]
USD_RATE = 5.0            # BRL per USD (matches 6_analysis.do: renda_USD = renda / 5)
MIN_WAGE = 1302.0         # 2023 statutory monthly minimum wage (BRL), GMM trimming anchor

# Columns pulled from each .dta (keep small for memory)
COLS = [
    "id_pessoa", "ano", "trimestre", "peso",
    "renda_habitual_principal", "horas_habituais_principal",
    "empregado_setor_pub", "empregado_setor_priv", "desocupado",
    "conta_propria", "empregador", "trab_domestico", "trab_familiar_aux",
    "na_pea",
]

# Palette (matches the Stata orange/blue convention)
ORANGE, BLUE, MIDBLUE, GREY = "#E8952B", "#2B4B9B", "#3B6EA5", "#888888"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)

# Transition-matrix state definitions (identical to pnadc_superpc/analysis.py)
DETAILED_STATES = [
    "Unemployed", "Private employee", "Domestic worker", "Self-employed",
    "Employer", "Family auxiliary", "Public employee",
]
COLLAPSED_STATES = ["Unemployed", "Private sector", "Public sector"]


# ---------------------------------------------------------------------------
# Data loading and panel construction
# ---------------------------------------------------------------------------
def _ind(df, col):
    """1 where numeric column equals 1, else 0 (NaN-safe)."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).eq(1)


def load_panel():
    """Load both years from .dta, keep needed columns, build the person-quarter panel."""
    frames = []
    for year in YEARS:
        path = CLEANED_DIR / f"PNADC_limpo_{year}.dta"
        print(f"Loading {path.name} ...")
        avail = pd.read_stata(path, convert_categoricals=False, iterator=True).varlist
        use = [c for c in COLS if c in avail]
        df = pd.read_stata(path, columns=use, convert_categoricals=False)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    for c in ["ano", "trimestre", "renda_habitual_principal",
              "horas_habituais_principal", "peso"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "peso" not in df.columns:
        df["peso"] = 1.0

    base = int(df["ano"].min())
    df["time"] = (df["ano"] - base) * 4 + df["trimestre"]
    df = df.sort_values(["id_pessoa", "time"]).reset_index(drop=True)
    print(f"Panel: {len(df):,} person-quarter rows, "
          f"{df['id_pessoa'].nunique():,} individuals")
    return df


def detailed_state(df):
    conds = [
        _ind(df, "desocupado"),
        _ind(df, "empregado_setor_priv"),
        _ind(df, "trab_domestico"),
        _ind(df, "conta_propria"),
        _ind(df, "empregador"),
        _ind(df, "trab_familiar_aux"),
        _ind(df, "empregado_setor_pub"),
    ]
    out = pd.Series(np.select(conds, DETAILED_STATES, default=None),
                    index=df.index, dtype="object")
    if "na_pea" in df.columns:                       # drop out-of-labor-force
        out = out.where(_ind(df, "na_pea"))
    return out


def collapsed_state(df):
    """U / Private / Public per the paper mapping. Also the model's U/R/P."""
    public = _ind(df, "empregado_setor_pub")
    private = (_ind(df, "empregado_setor_priv") | _ind(df, "conta_propria")
               | _ind(df, "empregador") | _ind(df, "trab_domestico"))
    unemployed = _ind(df, "desocupado") | _ind(df, "trab_familiar_aux")
    out = pd.Series(np.select([unemployed, private, public], COLLAPSED_STATES,
                              default=None), index=df.index, dtype="object")
    if "na_pea" in df.columns:
        out = out.where(_ind(df, "na_pea"))
    return out


def add_next_period(df, cols):
    """For each column in `cols`, add a `_t1` (next consecutive quarter) version.
    Only defined when the next observation is exactly one quarter later."""
    g = df.groupby("id_pessoa")
    next_time = g["time"].shift(-1)
    consec = (next_time - df["time"]).eq(1)
    df["_consec_t1"] = consec
    for c in cols:
        df[f"{c}_t1"] = g[c].shift(-1).where(consec)
    return df


def transition_matrix(df, origin, dest, states):
    counts = (df.dropna(subset=[origin, dest])
                .groupby([origin, dest]).size().unstack(fill_value=0)
                .reindex(index=states, columns=states, fill_value=0))
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    return counts, probs


# ---------------------------------------------------------------------------
# figure_1 -- transition-probability heatmaps (Panels A and B)
# ---------------------------------------------------------------------------
def make_figure_1(df):
    df = df.copy()
    df["st_det"] = detailed_state(df)
    df["st_col"] = collapsed_state(df)
    df = add_next_period(df, ["st_det", "st_col"])

    # Panel A: detailed 7-state
    _, prob_d = transition_matrix(df, "st_det", "st_det_t1", DETAILED_STATES)
    _plot_heatmap(prob_d.mul(100).round(1), (9.6, 7.2),
                  F_DIR / "figure_1_transition_probabilities.pdf")

    # Panel B: collapsed U / Private / Public
    _, prob_c = transition_matrix(df, "st_col", "st_col_t1", COLLAPSED_STATES)
    _plot_heatmap(prob_c.mul(100).round(1), (7.2, 5.8),
                  F_DIR / "figure_1_transition_probabilities_2.pdf")


def _plot_heatmap(prob, figsize, out_path):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(prob, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5,
                linecolor="grey", annot_kws={"size": 8.5}, ax=ax,
                cbar_kws={"label": "Transition probability (%)"})
    ax.set_xlabel("Employment status at t+1")
    ax.set_ylabel("Employment status at t")
    ax.tick_params(axis="x", rotation=30, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0, labelsize=8.5)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] {out_path.name}")


# ---------------------------------------------------------------------------
# Wage variables (winsorized, USD, deltas) + transition dummies
# ---------------------------------------------------------------------------
def build_wage_frame(df):
    """Replicates the wage construction in 6_analysis.do (sections 1.1-1.3)."""
    df = df.copy()
    df["st_col"] = collapsed_state(df)
    df = add_next_period(df, ["st_col", "renda_habitual_principal"])

    # setor codes to match the .do: 0=Unemployed, 1=Private, 2=Public
    setor_map = {"Unemployed": 0, "Private sector": 1, "Public sector": 2}
    df["setor"] = df["st_col"].map(setor_map)
    df["setor_t1_from"] = df["st_col_t1"].map(setor_map)   # state one quarter ahead

    # Transition dummies: current row is the DESTINATION state (prev quarter differs).
    prev = df.groupby("id_pessoa")["st_col"].shift(1)
    prev_consec = (df["time"] - df.groupby("id_pessoa")["time"].shift(1)).eq(1)
    prev = prev.where(prev_consec)
    df["tr_priv_to_pub"] = (prev == "Private sector") & (df["st_col"] == "Public sector")
    df["tr_pub_to_priv"] = (prev == "Public sector") & (df["st_col"] == "Private sector")
    df["tr_unemp_to_pub"] = (prev == "Unemployed") & (df["st_col"] == "Public sector")
    df["tr_unemp_to_priv"] = (prev == "Unemployed") & (df["st_col"] == "Private sector")

    # Winsorize renda at 1st/99th pct within year, then overall (as in the .do).
    r = df["renda_habitual_principal"].astype(float).copy()
    for yr in df["ano"].dropna().unique():
        m = df["ano"].eq(yr) & r.notna()
        if m.sum() == 0:
            continue
        lo, hi = r[m].quantile([0.01, 0.99])
        r.loc[m] = r[m].clip(lo, hi)
    lo, hi = r.dropna().quantile([0.01, 0.99])
    r = r.clip(lo, hi)
    df["renda"] = r
    df["renda_USD"] = r / USD_RATE

    # delta wage (%) relative to previous-quarter own wage, winsorized by year
    r_prev = df.groupby("id_pessoa")["renda"].shift(1).where(prev_consec)
    drel = ((df["renda"] - r_prev) / r_prev) * 100.0
    df["delta_renda_relative"] = drel
    drel_w = drel.copy()
    for yr in df["ano"].dropna().unique():
        m = df["ano"].eq(yr) & drel.notna()
        if m.sum() == 0:
            continue
        lo, hi = _wq(drel[m], df.loc[m, "peso"], [0.01, 0.99])
        drel_w.loc[m] = drel[m].clip(lo, hi)
    df["delta_renda_relative_w"] = drel_w
    return df


def _wq(values, weights, q):
    """Weighted quantile(s)."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return [np.nan for _ in q]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return list(np.interp(q, cw, v))


# ---------------------------------------------------------------------------
# figure_2 -- distribution of wage changes (%) at pub<->priv transitions
# ---------------------------------------------------------------------------
def make_figure_2(df):
    var = "delta_renda_relative_w"
    fig, ax = plt.subplots(figsize=(8, 5))
    stats = {}
    for mask_col, color, lab in [("tr_priv_to_pub", ORANGE, "Private to Public"),
                                 ("tr_pub_to_priv", MIDBLUE, "Public to Private")]:
        sub = df.loc[df[mask_col], [var, "peso"]].dropna()
        if len(sub) < 5:
            continue
        _weighted_kde(ax, sub[var].to_numpy(), sub["peso"].to_numpy(), color, lab)
        mean = np.average(sub[var], weights=sub["peso"])
        stats[lab] = (mean, len(sub))
        ax.axvline(mean, color=color, ls="--", lw=1.2)
    ax.set_xlabel("Change in Wage (%)")
    ax.set_ylabel("Density")
    note = "Dashed lines = mean change per transition type. " + \
        ", ".join(f"{k}: {v[0]:.2f} (N={v[1]:,})" for k, v in stats.items())
    ax.legend(loc="upper right")
    fig.text(0.5, -0.02, note, ha="center", fontsize=7)
    plt.tight_layout()
    out = F_DIR / "figure_2_delta_renda_relative_w.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] {out.name}")


# ---------------------------------------------------------------------------
# figure_3 -- wage distributions (USD): switchers vs incumbents
# ---------------------------------------------------------------------------
def make_figure_3(df):
    var = "renda_USD"
    fig, ax = plt.subplots(figsize=(8, 5))
    series = [
        (df["tr_priv_to_pub"], ORANGE, "--", "Private to Public"),
        (df["tr_pub_to_priv"], MIDBLUE, "--", "Public to Private"),
        (df["setor"].eq(2), ORANGE, "-", "Public Sector Incumbents"),
        (df["setor"].eq(1), BLUE, "-", "Private Sector Incumbents"),
    ]
    for mask, color, ls, lab in series:
        vals = df.loc[mask, var].dropna().to_numpy()
        if len(vals) < 5:
            continue
        _weighted_kde(ax, vals, None, color, lab, ls=ls)   # figure_3 is unweighted (as in .do)
    ax.set_xlabel("Wage (USD)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, np.nanpercentile(df[var].dropna(), 99))
    ax.legend(loc="upper right", fontsize=8)
    fig.text(0.5, -0.02,
             "Dashed = switchers, solid = incumbents.", ha="center", fontsize=7)
    plt.tight_layout()
    out = F_DIR / "figure_3_renda_USD.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] {out.name}")


def _weighted_kde(ax, values, weights, color, label, ls="-", n=512):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 5 or np.ptp(values) == 0:
        return
    kde = gaussian_kde(values, weights=weights) if weights is not None \
        else gaussian_kde(values)
    xs = np.linspace(values.min(), values.max(), n)
    ax.plot(xs, kde(xs), color=color, ls=ls, lw=1.8, label=label)


# ---------------------------------------------------------------------------
# table_1 -- wage summary statistics (USD), by group  [FIXED: adds U->P, U->R]
# ---------------------------------------------------------------------------
def make_table_1(df, weighted=True):
    """Percentiles and summary stats of renda_USD by group.
    Adds the two groups the paper note asked for: Unemployment->Public and
    Unemployment->Private (previously only pub<->priv switchers + incumbents)."""
    groups = [
        ("Private incumbents", df["setor"].eq(1)),
        ("Public incumbents", df["setor"].eq(2)),
        ("Unemp. to Private", df["tr_unemp_to_priv"]),
        ("Unemp. to Public", df["tr_unemp_to_pub"]),
        ("Private to Public", df["tr_priv_to_pub"]),
        ("Public to Private", df["tr_pub_to_priv"]),
    ]
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    rows = {}
    for name, mask in groups:
        sub = df.loc[mask, ["renda_USD", "peso"]].dropna()
        if len(sub) == 0:
            continue
        v = sub["renda_USD"].to_numpy()
        w = sub["peso"].to_numpy()
        if weighted:
            mean = np.average(v, weights=w)
            sd = np.sqrt(np.average((v - mean) ** 2, weights=w))
            p10, p25, p50, p75, p90 = _wq(v, w, qs)
        else:
            mean, sd = v.mean(), v.std(ddof=1)
            p10, p25, p50, p75, p90 = np.quantile(v, qs)
        rows[name] = [len(sub), mean, sd, p10, p25, p50, p75, p90]

    tbl = pd.DataFrame(rows, index=["N", "Mean", "SD", "P10", "P25",
                                    "P50", "P75", "P90"]).T

    # Emit as a booktabs tabular (no table wrapper: the .tex is \input into the
    # paper's own \begin{table}...\resizebox{}{}).
    cols = ["N", "Mean", "SD", "P10", "P25", "P50", "P75", "P90"]
    lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(cols) + r" \\", r"\midrule"]
    for name, r in tbl.iterrows():
        cells = [f"{int(r['N']):,}"] + [f"{r[c]:,.1f}" for c in cols[1:]]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = T_DIR / "table_1_renda_USD.tex"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [TABLE] {out.name}")
    print(tbl.round(1).to_string())


# ---------------------------------------------------------------------------
# table_V10E -- reservation wages & a/beta (GMM). Ported from 9_GMM.py.
# ---------------------------------------------------------------------------
def _upper_tail_imr(z):
    return float(np.exp(norm.logpdf(z) - norm.logsf(z)))


def _recover_sector(mean_log_acc, var_log_acc, p_accept, name):
    if not (0 < p_accept < 1):
        raise ValueError(f"{name}: p_accept={p_accept:.4f} outside (0,1).")
    z = float(norm.isf(p_accept))
    imr = _upper_tail_imr(z)
    denom = 1.0 + z * imr - imr ** 2
    if denom <= 1e-10:
        raise ValueError(f"{name}: implied variance factor non-positive.")
    sigma = float(np.sqrt(var_log_acc / denom))
    mu = float(mean_log_acc - sigma * imr)
    return float(np.exp(mu + sigma * z))               # reservation wage R


def _extract_moments(df, min_wage):
    """Deltas from full sample; accepted U->s wages trimmed at the minimum wage."""
    st = df["st_col"]
    # separation rates: origin employed -> next-quarter unemployed (full sample)
    nxt = df["st_col_next"]
    emp_pub = st.eq("Public sector") & nxt.notna()
    emp_priv = st.eq("Private sector") & nxt.notna()
    delta_P = float((nxt[emp_pub] == "Unemployed").mean())
    delta_R = float((nxt[emp_priv] == "Unemployed").mean())

    unemp = st.eq("Unemployed") & nxt.notna()
    n_unemp = int(unemp.sum())
    w1 = df["renda_next"]
    up = w1[unemp & nxt.eq("Public sector")].dropna()
    ur = w1[unemp & nxt.eq("Private sector")].dropna()
    up = up[up >= min_wage].to_numpy()
    ur = ur[ur >= min_wage].to_numpy()
    if len(up) < 10 or len(ur) < 10:
        raise ValueError(f"Too few accepted wages: n_UP={len(up)}, n_UR={len(ur)}")
    return {
        "delta_P": delta_P, "delta_R": delta_R,
        "m_UP": len(up) / n_unemp, "m_UR": len(ur) / n_unemp,
        "n_UP": len(up), "n_UR": len(ur),
        "mean_log_UP": float(np.log(up).mean()), "var_log_UP": float(np.log(up).var()),
        "mean_log_UR": float(np.log(ur).mean()), "var_log_UR": float(np.log(ur).var()),
    }


def _estimate_one(m, alpha, lambda0):
    p_P = m["m_UP"] / (lambda0 * alpha)
    p_R = m["m_UR"] / (lambda0 * (1 - alpha))
    R_P = _recover_sector(m["mean_log_UP"], m["var_log_UP"], p_P, "Public")
    R_R = _recover_sector(m["mean_log_UR"], m["var_log_UR"], p_R, "Private")
    return {"R_P": R_P, "R_R": R_R, "a_over_beta": R_R - R_P,
            "p_P": p_P, "p_R": p_R}


def make_table_V10E(df, alphas, lambda0s, bootstrap, min_wage, seed=42):
    # helper next-quarter columns for the moment extractor
    df = df.copy()
    g = df.groupby("id_pessoa")
    consec = (g["time"].shift(-1) - df["time"]).eq(1)
    df["st_col_next"] = g["st_col"].shift(-1).where(consec)
    df["renda_next"] = g["renda"].shift(-1).where(consec)

    combos = list(product(alphas, lambda0s))
    m = _extract_moments(df, min_wage)
    point = {c: _safe(lambda: _estimate_one(m, *c)) for c in combos}

    # cluster bootstrap by individual
    ci = {c: {} for c in combos}
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        ids = df["id_pessoa"].unique()
        idx = df.groupby("id_pessoa").indices
        draws = {c: {k: [] for k in ("R_P", "R_R", "a_over_beta")} for c in combos}
        for b in range(bootstrap):
            take = rng.choice(ids, size=len(ids), replace=True)
            rows = np.concatenate([idx[i] for i in take])
            bdf = df.iloc[rows]
            try:
                mb = _extract_moments(bdf, min_wage)
            except ValueError:
                continue
            for c in combos:
                est = _safe(lambda: _estimate_one(mb, *c))
                if est:
                    for k in draws[c]:
                        draws[c][k].append(est[k])
            if (b + 1) % 50 == 0:
                print(f"    bootstrap {b + 1}/{bootstrap}")
        for c in combos:
            for k in ("R_P", "R_R", "a_over_beta"):
                arr = np.array(draws[c][k], float)
                ci[c][k] = np.quantile(arr, [0.025, 0.975]) if arr.size else (np.nan, np.nan)

    out = _build_v10e_latex(combos, point, ci, m, bootstrap)
    p = T_DIR / "table_V10E.tex"
    p.write_text(out, encoding="utf-8")
    print(f"  [TABLE] {p.name}")


def _safe(fn):
    try:
        return fn()
    except Exception as exc:                            # noqa: BLE001
        print(f"    (estimate failed: {exc})")
        return None


def _build_v10e_latex(combos, point, ci, m, bootstrap):
    def cell(c, k):
        est = point[c]
        if not est:
            return "--"
        s = f"{est[k]:,.1f}"
        if bootstrap > 0 and k in ci[c]:
            lo, hi = ci[c][k]
            if np.isfinite(lo):
                s = r"\makecell[c]{" + s + r" \\ [" + f"{lo:,.1f}, {hi:,.1f}" + "]}"
        return s

    hdr = [rf"$\alpha={a:.2f},\ \lambda_0={l:.2f}$" for a, l in combos]
    L = [r"\begin{tabular}{l" + "c" * len(combos) + "}", r"\toprule",
         " & " + " & ".join(hdr) + r" \\", r"\midrule",
         r"$R_R$ & " + " & ".join(cell(c, "R_R") for c in combos) + r" \\",
         r"$R_P$ & " + " & ".join(cell(c, "R_P") for c in combos) + r" \\",
         r"$a/\beta$ & " + " & ".join(cell(c, "a_over_beta") for c in combos) + r" \\",
         r"\midrule",
         r"$\alpha$ & " + " & ".join(f"{a:.2f}" for a, _ in combos) + r" \\",
         r"$\lambda_0$ & " + " & ".join(f"{l:.2f}" for _, l in combos) + r" \\",
         r"$\delta_P$ & " + " & ".join(f"{m['delta_P']:.4f}" for _ in combos) + r" \\",
         r"$\delta_R$ & " + " & ".join(f"{m['delta_R']:.4f}" for _ in combos) + r" \\",
         r"$m_{UP}$ & " + " & ".join(f"{m['m_UP']:.4f}" for _ in combos) + r" \\",
         r"$m_{UR}$ & " + " & ".join(f"{m['m_UR']:.4f}" for _ in combos) + r" \\",
         r"$n_{UP}$ & " + " & ".join(f"{m['n_UP']:,}" for _ in combos) + r" \\",
         r"$n_{UR}$ & " + " & ".join(f"{m['n_UR']:,}" for _ in combos) + r" \\",
         r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.03, 0.06])
    ap.add_argument("--lambda0s", type=float, nargs="+", default=[0.50, 0.80])
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--min-wage", type=float, default=MIN_WAGE)
    ap.add_argument("--no-gmm", action="store_true", help="Skip the GMM table.")
    ap.add_argument("--unweighted-table1", action="store_true")
    args = ap.parse_args()

    df = load_panel()

    print("\n== figure_1 (transition probabilities) ==")
    make_figure_1(df)

    print("\n== wage variables ==")
    wdf = build_wage_frame(df)

    print("\n== figure_2 (wage-change distribution) ==")
    make_figure_2(wdf)

    print("\n== figure_3 (wage distributions, USD) ==")
    make_figure_3(wdf)

    print("\n== table_1 (wage summary stats, USD) ==")
    make_table_1(wdf, weighted=not args.unweighted_table1)

    if not args.no_gmm:
        print("\n== table_V10E (reservation wages & a/beta) ==")
        make_table_V10E(wdf, args.alphas, args.lambda0s,
                        args.bootstrap, args.min_wage)

    print("\nDone.")
    print(f"  Figures -> {F_DIR}")
    print(f"  Tables  -> {T_DIR}")


if __name__ == "__main__":
    main()
