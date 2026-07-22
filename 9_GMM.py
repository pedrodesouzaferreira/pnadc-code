"""
GMM v10E - 100% minimum-wage cutoff, alpha/lambda grid, LaTeX table output.

What this script does
---------------------
Starting from the v10D logic, this script fixes the trimming cutoff at 100% of the
minimum wage and evaluates a grid of calibrated (alpha, lambda_0) combinations.

For each combination, it computes:
    - R_R
    - R_P
    - a_over_beta = R_R - R_P
and percentile bootstrap 95% confidence intervals.

It then prints a LaTeX-formatted table with:
    - one column per (alpha, lambda_0) pair
    - rows for R_R, R_P, a/beta (with CIs in brackets)
    - bottom rows for alpha, lambda_0, delta_P, delta_R, m_UP, m_UR, n_UP, n_UR

Examples
--------
python 9_GMM.py --alphas 0.03 0.06 --lambda0s 0.50 0.80 --bootstrap 300
python 9_GMM.py --sample higher-ed --alphas 0.03 --lambda0s 0.50 --bootstrap 500
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from harmonized_data import add_sample_argument, load_harmonized

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "Cleaned Data"
DEFAULT_MIN_WAGE = 1302.0   # statutory monthly minimum wage (BRL); floor on accepted-wage
#                             moments. formal+full-time defines the SAMPLE; this floor cleans
#                             the WAGE MOMENTS (drops the sub-minimum private tail that would
#                             otherwise push R_R below R_P and make a/beta negative).
STATUTORY_MIN_WAGE = 1302.0  # used by diagnostics regardless of --min-wage
YEARS = (2023, 2024)
HOURS_MIN = 30
REAL_WAGE_COLUMN = "renda_habitual_principal_real_2025q3"  # deflated to 2025q3 by 2_harmonize.py
MIN_WAGE_BY_YEAR = {                                       # nominal statutory monthly minimum (BRL)
    2016: 880.0, 2017: 937.0, 2018: 954.0, 2019: 998.0, 2020: 1045.0,
    2021: 1100.0, 2022: 1212.0, 2023: 1302.0, 2024: 1412.0, 2025: 1518.0,
}
RAW_COLUMNS = [
    "id_pessoa", "ano", "trimestre",
    "renda_habitual_principal", REAL_WAGE_COLUMN,
    "horas_habituais_principal", "formal", "empregado_setor_pub",
    "empregado_setor_priv", "desocupado", "conta_propria", "empregador",
    "trab_domestico", "trab_familiar_aux",
]


# -----------------------------------------------------------------------------
# Stable helpers
# -----------------------------------------------------------------------------

def upper_tail_imr(z):
    return float(np.exp(norm.logpdf(z) - norm.logsf(z)))


def lognormal_mean(mu, sigma):
    return float(np.exp(mu + 0.5 * sigma ** 2))


def lognormal_trunc_mean_above(mu, sigma, cutoff):
    cutoff = float(max(cutoff, 1e-12))
    z = (np.log(cutoff) - mu) / sigma
    log_num = mu + 0.5 * sigma ** 2 + norm.logsf(z - sigma)
    log_den = norm.logsf(z)
    return float(np.exp(log_num - log_den))


def fmt_num(x, digits=1):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:,.{digits}f}"


def fmt_int(x):
    if x is None or not np.isfinite(x):
        return "--"
    return f"{int(round(x)):,}"


def latex_escape(text):
    return str(text).replace("_", r"\_")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_prepared_panel(path):
    df = pd.read_parquet(path)
    if "valid_transition_pair" in df.columns:
        df = df[df["valid_transition_pair"]].copy()
    else:
        df = df[df["sector_t1"].notna()].copy()
    df["wage_t"] = pd.to_numeric(df["wage_t"], errors="coerce")
    df["wage_t1"] = pd.to_numeric(df["wage_t1"], errors="coerce")
    return df


def _indicator(df, column):
    if column not in df:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0).eq(1)


def _print_sample_diagnostics(df, statutory_min=STATUTORY_MIN_WAGE):
    """Funnel of who is kept, employment tabulations, transition counts, and the
    sub-minimum-wage share of accepted wages. Uses the frame BEFORE it is reduced
    to valid transition pairs (so `sector_t` still contains dropped rows as None)."""
    print("\n----- SAMPLE DIAGNOSTICS -----")
    _yrs = sorted(pd.to_numeric(df["ano"], errors="coerce").dropna().astype(int).unique().tolist())
    _yr_lbl = f"{_yrs[0]}-{_yrs[-1]}" if _yrs else "n/a"
    print(f"person-quarter rows loaded: {len(df):,}  "
          f"(years actually present: {_yr_lbl}; {len(_yrs)} year(s))")
    _by_year = pd.to_numeric(df["ano"], errors="coerce").value_counts().sort_index()
    print("  rows by year: " + ", ".join(f"{int(y)}:{int(c):,}" for y, c in _by_year.items()))

    pos_emp = (
        _indicator(df, "empregado_setor_pub")
        | _indicator(df, "empregado_setor_priv")
        | _indicator(df, "conta_propria")
        | _indicator(df, "empregador")
        | _indicator(df, "trab_domestico")
    )
    formal = _indicator(df, "formal")
    ft = df["horas_habituais_principal"] >= HOURS_MIN
    print(f"  employed (any position):          {int(pos_emp.sum()):,}")
    print(f"    formal (formal==1):             {int((pos_emp & formal).sum()):,}")
    print(f"    full-time (hrs>={HOURS_MIN}):           {int((pos_emp & ft).sum()):,}")
    print(f"    formal AND full-time (kept):    {int((pos_emp & formal & ft).sum()):,}")
    print(f"  unemployed (desocupado==1):       {int(_indicator(df, 'desocupado').sum()):,}")

    print("\n  state assignment (sector_t), including dropped rows (None):")
    for k, v in df["sector_t"].value_counts(dropna=False).items():
        print(f"    {str(k):<12}: {v:,}")

    if "valid_transition_pair" in df.columns:
        vp = df[df["valid_transition_pair"]]
        print(f"\n  valid consecutive-quarter pairs:  {len(vp):,}")
        order = ["unemployed", "public", "private"]
        ct = (vp.groupby(["sector_t", "sector_t1"]).size().unstack(fill_value=0)
                .reindex(index=order, columns=order, fill_value=0))
        print("  transition COUNTS (origin rows -> dest cols):")
        print("    " + ct.to_string().replace("\n", "\n    "))

        unemp = vp[vp["sector_t"] == "unemployed"]
        has_nom = "wage_nom_t1" in vp.columns
        for dest, lab in [("public", "U->P (public)"), ("private", "U->R (private)")]:
            sub_d = unemp[unemp["sector_t1"] == dest]
            nom = pd.to_numeric(sub_d["wage_nom_t1" if has_nom else "wage_t1"], errors="coerce")
            nom = nom[nom > 0]
            real_kept = pd.to_numeric(sub_d["wage_t1"], errors="coerce").dropna()
            dropped = len(nom) - len(real_kept)
            pct = 100 * dropped / max(len(nom), 1)
            print(f"\n  accepted {lab}: transitions with a wage={len(nom):,}; "
                  f"dropped by min-wage floor={dropped:,} ({pct:.1f}%); "
                  f"kept for moments={len(real_kept):,}")
            if len(real_kept):
                q = real_kept.quantile([0, .01, .05, .10, .25, .50]).round(0).astype(int).tolist()
                print(f"    real (2025q3) min/p1/p5/p10/p25/p50 = {q}")
    print("----- END DIAGNOSTICS -----\n")


def load_harmonized_panel(sample, diagnose=True, min_wage_override=None, years=YEARS):
    """Construct the consecutive-quarter GMM panel from harmonized data."""
    df = load_harmonized(CLEANED_DIR, sample, columns=RAW_COLUMNS, years=years)
    for column in ["ano", "trimestre", "renda_habitual_principal",
                   "horas_habituais_principal", REAL_WAGE_COLUMN]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Sample filter (replaces the minimum-wage cutoff). A person-quarter counts as
    #   EMPLOYED only if formal (formal == 1) AND full-time (usual hours >= HOURS_MIN);
    #   UNEMPLOYED iff desocupado == 1 (i.e. condicao_ocupacao == 2).
    # Informal, part-time, self-employed and out-of-labor-force rows get sector_t = None
    # and are dropped from the transition pairs below.  (Note: 'formal' = VD4009 in
    # {1,3,5,8}, so statutory servants/military (code 5) and public-with-carteira (3)
    # are retained; only public-without-carteira (code 4) is treated as informal.)
    if "formal" not in df.columns:
        raise ValueError(
            "Column 'formal' is missing from the harmonized data; re-run "
            "2_harmonize.py so 1_clean's 'formal' flag is carried through."
        )
    formal_ft = _indicator(df, "formal") & (df["horas_habituais_principal"] >= HOURS_MIN)
    public = _indicator(df, "empregado_setor_pub") & formal_ft
    private = (
        _indicator(df, "empregado_setor_priv")
        | _indicator(df, "conta_propria")
        | _indicator(df, "empregador")
        | _indicator(df, "trab_domestico")
    ) & formal_ft
    unemployed = _indicator(df, "desocupado")            # condicao_ocupacao == 2
    df["sector_t"] = np.select(
        [public, private, unemployed],
        ["public", "private", "unemployed"],
        default=None,
    )
    # Wages: use the deflated (real, 2025q3) column for the moments if available.
    nominal = df["renda_habitual_principal"]
    if REAL_WAGE_COLUMN in df.columns:
        real, using_real = df[REAL_WAGE_COLUMN], True
    else:
        real, using_real = nominal, False
        print(f"  WARNING: {REAL_WAGE_COLUMN!r} not in data; using NOMINAL wages.")
    # Minimum-wage floor on the NOMINAL wage vs each year's statutory minimum
    # (equivalent to a real floor, since both scale by the same per-quarter deflator).
    if min_wage_override is not None and min_wage_override >= 0:
        row_min = pd.Series(float(min_wage_override), index=df.index)
        floor_desc = f"flat {min_wage_override:.0f} (nominal, all years)"
    else:
        row_min = df["ano"].map(MIN_WAGE_BY_YEAR).fillna(max(MIN_WAGE_BY_YEAR.values()))
        floor_desc = "year-specific statutory min (" + \
            ", ".join(f"{y}:{int(v)}" for y, v in sorted(MIN_WAGE_BY_YEAR.items())) + ")"
    keep_wage = (nominal >= row_min) & (real > 0)
    df["wage_t"] = real.where(keep_wage)           # REAL wage, floored -> used for moments
    df["wage_nom_t"] = nominal.where(nominal > 0)  # nominal, kept for diagnostics only
    print(f"  wages: {'REAL (2025q3)' if using_real else 'NOMINAL (fallback)'}; "
          f"min-wage floor = {floor_desc}")

    base_year = int(df["ano"].min())
    df["time"] = (df["ano"] - base_year) * 4 + df["trimestre"]
    df = df.sort_values(["id_pessoa", "time"]).reset_index(drop=True)
    next_time = df.groupby("id_pessoa")["time"].shift(-1)
    consecutive = next_time.sub(df["time"]).eq(1)
    df["sector_t1"] = df.groupby("id_pessoa")["sector_t"].shift(-1).where(consecutive)
    df["wage_t1"] = df.groupby("id_pessoa")["wage_t"].shift(-1).where(consecutive)
    df["wage_nom_t1"] = df.groupby("id_pessoa")["wage_nom_t"].shift(-1).where(consecutive)
    df["valid_transition_pair"] = consecutive & df["sector_t1"].notna()
    if diagnose:
        _print_sample_diagnostics(df)
    df = df[df["valid_transition_pair"]].copy()
    print(f"Prepared {len(df):,} valid consecutive-quarter transition pairs")
    return df


# -----------------------------------------------------------------------------
# Raw data extraction
# -----------------------------------------------------------------------------

def extract_raw_objects(df):
    out = {
        "rows": len(df),
        "n_people": int(df["id_pessoa"].nunique()) if "id_pessoa" in df.columns else np.nan,
    }

    priv = df[df["sector_t"] == "private"]
    pub = df[df["sector_t"] == "public"]
    unemp = df[df["sector_t"] == "unemployed"]

    out["delta_R"] = float((priv["sector_t1"] == "unemployed").mean())
    out["delta_P"] = float((pub["sector_t1"] == "unemployed").mean())

    wages_up = unemp.loc[
        (unemp["sector_t1"] == "public")
        & unemp["wage_t1"].notna() & (unemp["wage_t1"] > 0),
        "wage_t1",
    ].astype(float).to_numpy()

    wages_ur = unemp.loc[
        (unemp["sector_t1"] == "private")
        & unemp["wage_t1"].notna() & (unemp["wage_t1"] > 0),
        "wage_t1",
    ].astype(float).to_numpy()

    if len(wages_up) < 10 or len(wages_ur) < 10:
        raise ValueError("Need at least 10 accepted U->sector wages in each sector.")

    out["n_unemployed"] = len(unemp)
    out["wages_UP_raw"] = wages_up
    out["wages_UR_raw"] = wages_ur
    return out


# -----------------------------------------------------------------------------
# Moments after trimming (fixed at 100% of minimum wage)
# -----------------------------------------------------------------------------

def build_trimmed_moments(raw, cut_public, cut_private):
    wages_up = raw["wages_UP_raw"]
    wages_ur = raw["wages_UR_raw"]
    n_unemployed = raw["n_unemployed"]

    keep_up = wages_up >= cut_public
    keep_ur = wages_ur >= cut_private

    up = wages_up[keep_up]
    ur = wages_ur[keep_ur]

    if len(up) < 10 or len(ur) < 10:
        raise ValueError(f"Too few observations after trimming: n_UP={len(up)}, n_UR={len(ur)}.")

    log_up = np.log(up)
    log_ur = np.log(ur)

    m_UP = len(up) / n_unemployed
    m_UR = len(ur) / n_unemployed

    return {
        "rows": raw["rows"],
        "n_people": raw["n_people"],
        "n_unemployed": n_unemployed,
        "delta_R": raw["delta_R"],
        "delta_P": raw["delta_P"],
        "m_UP": float(m_UP),
        "m_UR": float(m_UR),
        "cut_public": float(cut_public),
        "cut_private": float(cut_private),
        "n_UP": int(len(up)),
        "n_UR": int(len(ur)),
        "n_UP_raw": int(len(wages_up)),
        "n_UR_raw": int(len(wages_ur)),
        "share_kept_UP": float(len(up) / len(wages_up)),
        "share_kept_UR": float(len(ur) / len(wages_ur)),
        "Ew_UP": float(up.mean()),
        "Ew_UR": float(ur.mean()),
        "mean_log_UP": float(log_up.mean()),
        "mean_log_UR": float(log_ur.mean()),
        "var_log_UP": float(log_up.var(ddof=0)),
        "var_log_UR": float(log_ur.var(ddof=0)),
        "min_acc_UP": float(up.min()),
        "min_acc_UR": float(ur.min()),
    }


# -----------------------------------------------------------------------------
# Closed-form sector recovery
# -----------------------------------------------------------------------------

def recover_sector_from_calibration(mean_log_acc, var_log_acc, p_accept, sector_name="sector"):
    if not np.isfinite(mean_log_acc):
        raise ValueError(f"{sector_name}: mean_log_acc is not finite.")
    if not np.isfinite(var_log_acc) or var_log_acc <= 0:
        raise ValueError(f"{sector_name}: var_log_acc must be positive.")
    if not np.isfinite(p_accept) or p_accept <= 0 or p_accept >= 1:
        raise ValueError(f"{sector_name}: p_accept must lie strictly between 0 and 1. Got {p_accept}.")

    z = float(norm.isf(p_accept))
    imr = upper_tail_imr(z)
    denom = 1.0 + z * imr - imr ** 2
    if denom <= 1e-10:
        raise ValueError(f"{sector_name}: implied variance factor is non-positive ({denom}).")

    sigma = float(np.sqrt(var_log_acc / denom))
    mu = float(mean_log_acc - sigma * imr)
    c = float(mu + sigma * z)
    R = float(np.exp(c))

    return {
        "mu": mu,
        "sigma": sigma,
        "R": R,
        "offer_mean_model": float(lognormal_mean(mu, sigma)),
        "accepted_mean_model": float(lognormal_trunc_mean_above(mu, sigma, R)),
    }


# -----------------------------------------------------------------------------
# Estimation for one alpha/lambda pair
# -----------------------------------------------------------------------------

def estimate_one(m, alpha, lambda_0):
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie strictly between 0 and 1.")
    if lambda_0 <= 0:
        raise ValueError("lambda_0 must be positive.")

    p_P = m["m_UP"] / (lambda_0 * alpha)
    p_R = m["m_UR"] / (lambda_0 * (1.0 - alpha))

    if not (0 < p_P < 1):
        raise ValueError(
            f"Implied public acceptance probability is {p_P:.4f}, outside (0,1)."
        )
    if not (0 < p_R < 1):
        raise ValueError(
            f"Implied private acceptance probability is {p_R:.4f}, outside (0,1)."
        )

    fit_P = recover_sector_from_calibration(
        mean_log_acc=m["mean_log_UP"],
        var_log_acc=m["var_log_UP"],
        p_accept=p_P,
        sector_name="Public",
    )
    fit_R = recover_sector_from_calibration(
        mean_log_acc=m["mean_log_UR"],
        var_log_acc=m["var_log_UR"],
        p_accept=p_R,
        sector_name="Private",
    )

    return {
        "alpha": float(alpha),
        "lambda_0": float(lambda_0),
        "delta_P": float(m["delta_P"]),
        "delta_R": float(m["delta_R"]),
        "m_UP": float(m["m_UP"]),
        "m_UR": float(m["m_UR"]),
        "n_UP": int(m["n_UP"]),
        "n_UR": int(m["n_UR"]),
        "p_accept_P": float(p_P),
        "p_accept_R": float(p_R),
        "R_P": float(fit_P["R"]),
        "R_R": float(fit_R["R"]),
        "a_over_beta": float(fit_R["R"] - fit_P["R"]),
        "offer_mean_model_P": float(fit_P["offer_mean_model"]),
        "offer_mean_model_R": float(fit_R["offer_mean_model"]),
        "accepted_mean_model_P": float(fit_P["accepted_mean_model"]),
        "accepted_mean_model_R": float(fit_R["accepted_mean_model"]),
    }


# -----------------------------------------------------------------------------
# Cluster bootstrap by individual
# -----------------------------------------------------------------------------

def bootstrap_resample_by_person(df, id_col, rng, group_index=None):
    if group_index is None:
        group_index = df.groupby(id_col, sort=False).indices

    ids = np.array(list(group_index.keys()), dtype=object)
    draw = rng.choice(ids, size=len(ids), replace=True)

    parts = []
    for b, pid in enumerate(draw):
        block = df.iloc[group_index[pid]].copy()
        block[id_col] = f"{pid}__boot{b}"
        parts.append(block)

    return pd.concat(parts, ignore_index=True)


# -----------------------------------------------------------------------------
# Point estimates and bootstrap grid
# -----------------------------------------------------------------------------

def run_point_grid(m, combos):
    rows = []
    for alpha, lambda_0 in combos:
        row = {
            "alpha": float(alpha),
            "lambda_0": float(lambda_0),
            "status": "ok",
            "error": "",
        }
        try:
            row.update(estimate_one(m, alpha=alpha, lambda_0=lambda_0))
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_grid(df, combos, min_wage, reps=200, seed=42, id_col="id_pessoa", progress_every=25):
    rng = np.random.default_rng(seed)
    group_index = df.groupby(id_col, sort=False).indices
    out = []

    for b in range(reps):
        boot_df = bootstrap_resample_by_person(df, id_col=id_col, rng=rng, group_index=group_index)
        raw_b = extract_raw_objects(boot_df)
        m_b = build_trimmed_moments(raw_b, cut_public=min_wage, cut_private=min_wage)
        grid_b = run_point_grid(m_b, combos)
        grid_b = grid_b.copy()
        grid_b["bootstrap_rep"] = b
        out.append(grid_b)

        if progress_every and ((b + 1) % progress_every == 0 or (b + 1) == reps):
            print(f"Bootstrap replication {b + 1}/{reps}")

    return pd.concat(out, ignore_index=True)


def summarize_bootstrap_ci(boot_df, combos, params=("R_P", "R_R", "a_over_beta")):
    ok = boot_df[boot_df["status"] == "ok"].copy()
    rows = []

    for alpha, lambda_0 in combos:
        sub = ok[(np.isclose(ok["alpha"], alpha)) & (np.isclose(ok["lambda_0"], lambda_0))]
        row = {
            "alpha": float(alpha),
            "lambda_0": float(lambda_0),
            "n_boot_ok": int(len(sub)),
        }
        for p in params:
            vals = pd.to_numeric(sub[p], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                row[f"{p}_p2_5"] = np.nan
                row[f"{p}_p50"] = np.nan
                row[f"{p}_p97_5"] = np.nan
            else:
                q = np.quantile(vals, [0.025, 0.5, 0.975])
                row[f"{p}_p2_5"] = float(q[0])
                row[f"{p}_p50"] = float(q[1])
                row[f"{p}_p97_5"] = float(q[2])
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# LaTeX table builder
# -----------------------------------------------------------------------------

def build_latex_table(point_df, ci_df, combos, caption, label):
    point_map = {(float(r.alpha), float(r.lambda_0)): r for r in point_df.itertuples(index=False)}
    ci_map = {(float(r.alpha), float(r.lambda_0)): r for r in ci_df.itertuples(index=False)}

    def est_ci_cell(combo, param):
        prow = point_map.get(combo)
        crow = ci_map.get(combo)
        if prow is None or getattr(prow, "status", "failed") != "ok":
            return "--"
        est = getattr(prow, param)
        if crow is None:
            return fmt_num(est, 1)
        lo = getattr(crow, f"{param}_p2_5", np.nan)
        hi = getattr(crow, f"{param}_p97_5", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            return "\\makecell[c]{" + fmt_num(est, 1) + r" \\ [" + fmt_num(lo, 1) + ", " + fmt_num(hi, 1) + "]}"
        return fmt_num(est, 1)

    def scalar_cell(combo, key, digits=4, integer=False):
        prow = point_map.get(combo)
        if prow is None or getattr(prow, "status", "failed") != "ok":
            return "--"
        val = getattr(prow, key)
        return fmt_int(val) if integer else fmt_num(val, digits)

    headers = [rf"$\alpha={alpha:.2f},\ \lambda_0={lam:.2f}$" for alpha, lam in combos]
    ncols = 1 + len(combos)

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(rf"\caption{{{latex_escape(caption)}}}")
    lines.append(rf"\label{{{latex_escape(label)}}}")
    lines.append(r"\begin{tabular}{l" + "c" * len(combos) + "}")
    lines.append(r"\toprule")
    lines.append(" & " + " & ".join(headers) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"$R_R$" + " & " + " & ".join(est_ci_cell(c, "R_R") for c in combos) + r" \\")
    lines.append(r"$R_P$" + " & " + " & ".join(est_ci_cell(c, "R_P") for c in combos) + r" \\")
    lines.append(r"$a/\beta$" + " & " + " & ".join(est_ci_cell(c, "a_over_beta") for c in combos) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"$\alpha$" + " & " + " & ".join(scalar_cell(c, "alpha", digits=2) for c in combos) + r" \\")
    lines.append(r"$\lambda_0$" + " & " + " & ".join(scalar_cell(c, "lambda_0", digits=2) for c in combos) + r" \\")
    lines.append(r"$\delta_P$" + " & " + " & ".join(scalar_cell(c, "delta_P", digits=4) for c in combos) + r" \\")
    lines.append(r"$\delta_R$" + " & " + " & ".join(scalar_cell(c, "delta_R", digits=4) for c in combos) + r" \\")
    lines.append(r"$m_{UP}$" + " & " + " & ".join(scalar_cell(c, "m_UP", digits=4) for c in combos) + r" \\")
    lines.append(r"$m_{UR}$" + " & " + " & ".join(scalar_cell(c, "m_UR", digits=4) for c in combos) + r" \\")
    lines.append(r"$n_{UP}$" + " & " + " & ".join(scalar_cell(c, "n_UP", integer=True) for c in combos) + r" \\")
    lines.append(r"$n_{UR}$" + " & " + " & ".join(scalar_cell(c, "n_UR", integer=True) for c in combos) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{0.25em}")
    lines.append(r"\begin{minipage}{0.95\linewidth}")
    lines.append(r"\footnotesize Notes: The estimation sample keeps formal, full-time employees (formal $=1$ and usual weekly hours $\geq 30$) and the unemployed (condi\c{c}\~ao de ocupa\c{c}\~ao $=2$); informal, part-time, self-employed and out-of-labour-force person-quarters are excluded; wages are deflated to 2025q3 reais and accepted-wage moments drop observations below each year\'s statutory minimum wage (2023: R\$1{,}302; 2024: R\$1{,}412). Cells for $R_R$, $R_P$, and $a/\beta$ report point estimates; bracketed bootstrap percentile 95\% confidence intervals are shown when the bootstrap is run (cluster bootstrap at the individual level).")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=None, help="Optional prepared panel; overrides --sample.")
    add_sample_argument(parser)
    parser.add_argument("--no-diagnose", dest="diagnose", action="store_false",
                        default=True, help="Suppress the sample-funnel diagnostics.")
    parser.add_argument("--year-min", type=int, default=2023,
                        help="First year to include (default 2023). "
                             "The higher-ed sample spans 2016-2025; the full sample is 2023-2024 only.")
    parser.add_argument("--year-max", type=int, default=2024,
                        help="Last year to include (default 2024).")
    parser.add_argument("--alphas", type=float, nargs="+", required=True, help="List of calibrated alpha values.")
    parser.add_argument("--lambda0s", type=float, nargs="+", required=True, help="List of calibrated lambda_0 values.")
    parser.add_argument("--min-wage", type=float, default=-1.0,
                        help="Flat NOMINAL min-wage floor for all years; default (-1) uses "
                             "year-specific statutory minimums (2023:1302, 2024:1412).")
    parser.add_argument("--bootstrap", type=int, default=300, help="Number of cluster-bootstrap repetitions.")
    parser.add_argument("--boot-seed", type=int, default=42, help="Random seed for bootstrap.")
    parser.add_argument("--boot-progress-every", type=int, default=25, help="Print bootstrap progress every N replications.")
    parser.add_argument("--caption", type=str, default="Reservation wages and amenity estimates across calibration choices", help="LaTeX table caption.")
    parser.add_argument("--label", type=str, default="tab:v10e_grid", help="LaTeX table label.")
    parser.add_argument("--output-tex", type=Path, default=None, help="Optional path to save the LaTeX table.")
    parser.add_argument("--save-point-csv", type=Path, default=None, help="Optional path to save point estimates as CSV.")
    parser.add_argument("--save-boot-csv", type=Path, default=None, help="Optional path to save bootstrap draws as CSV.")
    parser.add_argument("--save-ci-csv", type=Path, default=None, help="Optional path to save bootstrap CIs as CSV.")
    args = parser.parse_args()

    combos = [(a, l) for a, l in product(args.alphas, args.lambda0s)]

    years = tuple(range(args.year_min, args.year_max + 1))
    df = (load_prepared_panel(args.input) if args.input is not None
          else load_harmonized_panel(args.sample, diagnose=args.diagnose,
                                      min_wage_override=args.min_wage, years=years))
    raw = extract_raw_objects(df)
    m = build_trimmed_moments(raw, cut_public=0.0, cut_private=0.0)  # floor applied upstream (per-year, real)

    point_df = run_point_grid(m, combos)

    print()
    print("=== Point estimates (formal, full-time sample) ===")
    with pd.option_context("display.max_columns", None, "display.width", 250, "display.float_format", lambda x: f"{x:,.4f}"):
        print(point_df.to_string(index=False))

    if args.save_point_csv is not None:
        point_df.to_csv(args.save_point_csv, index=False)
        print()
        print(f"Saved point estimates to: {args.save_point_csv}")

    if args.bootstrap > 0:
        print()
        print(f"=== Running cluster bootstrap (B={args.bootstrap}) ===")
        boot_df = bootstrap_grid(
            df=df,
            combos=combos,
            min_wage=args.min_wage,
            reps=args.bootstrap,
            seed=args.boot_seed,
            progress_every=args.boot_progress_every,
        )

        ci_df = summarize_bootstrap_ci(boot_df, combos, params=("R_P", "R_R", "a_over_beta"))

        print()
        print("=== Bootstrap percentile 95% confidence intervals ===")
        with pd.option_context("display.max_columns", None, "display.width", 250, "display.float_format", lambda x: f"{x:,.4f}"):
            print(ci_df.to_string(index=False))

        if args.save_boot_csv is not None:
            boot_df.to_csv(args.save_boot_csv, index=False)
            print()
            print(f"Saved bootstrap draws to: {args.save_boot_csv}")

        if args.save_ci_csv is not None:
            ci_df.to_csv(args.save_ci_csv, index=False)
            print()
            print(f"Saved bootstrap CIs to: {args.save_ci_csv}")
    else:
        print()
        print("=== Skipping bootstrap (--bootstrap 0): point estimates only, no CIs ===")
        ci_df = pd.DataFrame(columns=["alpha", "lambda_0"])

    latex_table = build_latex_table(
        point_df=point_df,
        ci_df=ci_df,
        combos=combos,
        caption=args.caption,
        label=args.label,
    )

    print()
    print("=== LaTeX table ===")
    print(latex_table)

    if args.output_tex is not None:
        args.output_tex.write_text(latex_table, encoding="utf-8")
        print()
        print(f"Saved LaTeX table to: {args.output_tex}")


if __name__ == "__main__":
    main()
