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
python 9_GMM.py --alphas 0.03 0.06 --lambda0s 0.50 0.80 --bootstrap 500 --output-tex table_v10E.tex
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "Cleaned Data" / "PNADC_prepared_for_GMM_v4.parquet"
DEFAULT_MIN_WAGE = 1302.0


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

def load_panel(path):
    df = pd.read_parquet(path)
    if "valid_transition_pair" in df.columns:
        df = df[df["valid_transition_pair"]].copy()
    else:
        df = df[df["sector_t1"].notna()].copy()
    df["wage_t"] = pd.to_numeric(df["wage_t"], errors="coerce")
    df["wage_t1"] = pd.to_numeric(df["wage_t1"], errors="coerce")
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
    lines.append(r"\footnotesize Notes: All estimates use a trimming cutoff equal to 100\% of the statutory minimum wage. Cells for $R_R$, $R_P$, and $a/\beta$ report point estimates with bootstrap percentile 95\% confidence intervals in brackets. Confidence intervals are obtained by cluster bootstrap at the individual level.")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Panel parquet with sector_t, sector_t1, wage_t1.")
    parser.add_argument("--alphas", type=float, nargs="+", required=True, help="List of calibrated alpha values.")
    parser.add_argument("--lambda0s", type=float, nargs="+", required=True, help="List of calibrated lambda_0 values.")
    parser.add_argument("--min-wage", type=float, default=DEFAULT_MIN_WAGE, help=f"Minimum wage anchor and trimming cutoff (default: {DEFAULT_MIN_WAGE:.0f}).")
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

    df = load_panel(args.input)
    raw = extract_raw_objects(df)
    m = build_trimmed_moments(raw, cut_public=args.min_wage, cut_private=args.min_wage)

    point_df = run_point_grid(m, combos)

    print()
    print("=== Point estimates at 100% minimum-wage cutoff ===")
    with pd.option_context("display.max_columns", None, "display.width", 250, "display.float_format", lambda x: f"{x:,.4f}"):
        print(point_df.to_string(index=False))

    if args.save_point_csv is not None:
        point_df.to_csv(args.save_point_csv, index=False)
        print()
        print(f"Saved point estimates to: {args.save_point_csv}")

    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive in v10E because the LaTeX table includes confidence intervals.")

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
