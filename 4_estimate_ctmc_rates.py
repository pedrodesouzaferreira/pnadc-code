"""
4_estimate_ctmc_rates.py  --  Continuous-time hazard rates from PNADC's quarterly panel
=====================================================================================
Python port of estimate_ctmc_rates.R (see HANDOFF_ctmc_rate_fix.md).

WHY: the structural estimation plugs discrete-time quarterly transition FRACTIONS
into a continuous-time model whose parameters are Poisson HAZARD rates. That is a
units error. The one-quarter transition matrix is P(Delta) = expm(Q*Delta); reading
Phat_jk as if it equalled q_jk*Delta is fine for rare events but badly wrong for the
churny U-exit flows. This script recovers the generator Q correctly.

States (model U/R/P; here ordered U, P, R to match the handoff):
    U = unemployed         : desocupado OR unpaid family worker (trab_familiar_aux)
    P = public sector       : public employee
    R = private sector       : private employee, self-employed, employer, domestic worker

DELIVERABLE OF THIS PASS (per handoff): print Phat, Route A (closed form), Route B
(CTMC MLE), Route C (matrix log, a check), optional Route D (tenure), and a
comparison table. Then STOP -- no .tex edits, no downstream reservation-wage changes.

Time unit = ONE QUARTER throughout (Delta = 1). Data are quarterly.

Usage
-----
    python 4_estimate_ctmc_rates.py                       # Routes A, B, C + comparison
    python 4_estimate_ctmc_rates.py --tenure              # also Route D (grouped exponential)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm, logm
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "Cleaned Data"
YEARS = [2023, 2024]
STATES = ["U", "P", "R"]                      # fixed ordering used everywhere below

COLS = [
    "id_pessoa", "ano", "trimestre",
    "empregado_setor_pub", "empregado_setor_priv", "desocupado",
    "conta_propria", "empregador", "trab_domestico", "trab_familiar_aux",
    "na_pea", "renda_habitual_principal",
    "tempo_nesse_trabalho_label",             # Route D (interval-censored tenure)
]


# ---------------------------------------------------------------------------
# Data + state construction
# ---------------------------------------------------------------------------
def _ind(df, col):
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0).eq(1)


def load_panel(cols=COLS):
    frames = []
    for year in YEARS:
        path = CLEANED_DIR / f"PNADC_limpo_{year}.dta"
        print(f"Loading {path.name} ...")
        # read the header once to intersect requested columns with what exists
        head = next(pd.read_stata(path, convert_categoricals=False, chunksize=1))
        use = [c for c in cols if c in head.columns]
        frames.append(pd.read_stata(path, columns=use, convert_categoricals=False))
    df = pd.concat(frames, ignore_index=True)
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["trimestre"] = pd.to_numeric(df["trimestre"], errors="coerce")
    base = int(df["ano"].min())
    df["time"] = (df["ano"] - base) * 4 + df["trimestre"]
    df = df.sort_values(["id_pessoa", "time"]).reset_index(drop=True)
    print(f"Panel: {len(df):,} person-quarter rows, "
          f"{df['id_pessoa'].nunique():,} individuals")
    return df


def state_UPR(df):
    """Collapsed U/P/R state (paper mapping). NaN for out-of-labor-force."""
    public = _ind(df, "empregado_setor_pub")
    private = (_ind(df, "empregado_setor_priv") | _ind(df, "conta_propria")
               | _ind(df, "empregador") | _ind(df, "trab_domestico"))
    unemp = _ind(df, "desocupado") | _ind(df, "trab_familiar_aux")
    out = pd.Series(np.select([unemp, public, private], ["U", "P", "R"], default=None),
                    index=df.index, dtype="object")
    if "na_pea" in df.columns:
        out = out.where(_ind(df, "na_pea"))
    return out


def consecutive_pairs(df):
    """Return (origin_state, dest_state) for every same-person consecutive-quarter pair.
    Uses the FULL sample (no minimum-wage filter on transitions -- see handoff Step 1)."""
    df = df.copy()
    df["state"] = state_UPR(df)
    g = df.groupby("id_pessoa")
    consec = (g["time"].shift(-1) - df["time"]).eq(1)
    df["state_next"] = g["state"].shift(-1).where(consec)
    pairs = df.loc[consec & df["state"].notna() & df["state_next"].notna(),
                   ["state", "state_next"]]
    return pairs, df


def empirical_matrices(pairs):
    counts = (pairs.groupby(["state", "state_next"]).size().unstack(fill_value=0)
              .reindex(index=STATES, columns=STATES, fill_value=0))
    Phat = counts.div(counts.sum(axis=1), axis=0)
    return counts, Phat


# ---------------------------------------------------------------------------
# Route A -- closed-form competing-risks patch
# ---------------------------------------------------------------------------
def cr_hazard(p_dest, delta=1.0):
    """p_dest: dict {destination: probability} (excludes staying). Returns hazards."""
    p_exit = sum(p_dest.values())
    total_rate = -np.log(1 - p_exit) / delta
    return {k: total_rate * v / p_exit for k, v in p_dest.items()}


def route_A(Phat, delta=1.0):
    Q = pd.DataFrame(0.0, index=STATES, columns=STATES)
    for origin in STATES:
        dests = {d: Phat.loc[origin, d] for d in STATES if d != origin}
        haz = cr_hazard(dests, delta)
        for d, q in haz.items():
            Q.loc[origin, d] = q
        Q.loc[origin, origin] = -sum(haz.values())
    return Q


# ---------------------------------------------------------------------------
# Route B -- CTMC MLE on the discretely observed chain (replaces R's msm)
#   L = prod over consecutive pairs of [expm(Q*Delta)]_{s_t, s_t1}
#   Six free off-diagonal rates, optimized on the log scale (positivity).
# ---------------------------------------------------------------------------
_OFFDIAG = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]   # (U->P,U->R,P->U,P->R,R->U,R->P)


def _Q_from_theta(theta):
    q = np.exp(theta)
    Q = np.zeros((3, 3))
    for val, (i, j) in zip(q, _OFFDIAG):
        Q[i, j] = val
    for i in range(3):
        Q[i, i] = -Q[i].sum()
    return Q


def route_B(counts, delta=1.0, init_from=None):
    C = counts.to_numpy(dtype=float)                 # 3x3 transition counts

    def negll(theta):
        Q = _Q_from_theta(theta)
        P = expm(Q * delta)
        P = np.clip(P, 1e-300, None)
        return -np.sum(C * np.log(P))

    if init_from is not None:
        start = np.log(np.clip([init_from.iloc[i, j] for i, j in _OFFDIAG], 1e-4, None))
    else:
        start = np.log(np.full(6, 0.1))
    res = minimize(negll, start, method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
    Qhat = pd.DataFrame(_Q_from_theta(res.x), index=STATES, columns=STATES)
    Pimplied = pd.DataFrame(expm(Qhat.to_numpy() * delta), index=STATES, columns=STATES)
    return Qhat, Pimplied, res


# ---------------------------------------------------------------------------
# Route C -- matrix log of Phat (a check on B; may give small negatives)
# ---------------------------------------------------------------------------
def route_C(Phat, delta=1.0):
    Q = logm(Phat.to_numpy()) / delta
    return pd.DataFrame(np.real(Q), index=STATES, columns=STATES)


# ---------------------------------------------------------------------------
# Route D -- grouped-exponential MLE on interval-censored tenure (optional)
# ---------------------------------------------------------------------------
_BINS = {          # normalized-label substring -> (lower, upper) in YEARS
    "menos de 1 mes": (0.0, 1 / 12),
    "1 mes a menos de 1 ano": (1 / 12, 1.0),
    "1 ano a menos de 2 anos": (1.0, 2.0),
    "2 anos ou mais": (2.0, np.inf),
}


def _tenure_bin(label):
    s = (str(label).lower()
         .replace("ê", "e").replace("é", "e").replace("ç", "c").replace("í", "i")
         .replace("á", "a").replace("ã", "a").replace("ó", "o"))
    if "nao aplicavel" in s:
        return None
    for key, rng in _BINS.items():
        if key in s:
            return rng
    return None


def grouped_exp_nll(logq, counts):
    q = np.exp(logq)                                  # per-YEAR exit rate
    p = np.array([
        1 - np.exp(-q / 12),
        np.exp(-q / 12) - np.exp(-q),
        np.exp(-q) - np.exp(-2 * q),
        np.exp(-2 * q),
    ])
    p = np.clip(p, 1e-300, None)
    return -np.sum(counts * np.log(p))


def route_D(df):
    """Grouped-exponential separation-rate estimate per sector from tenure bins.
    Reports per-quarter q_j (= q_year/4). Fit within sector; the handoff also
    recommends restricting to top-wage workers so q_j ~= delta_j -- reported when
    a wage column is present."""
    if "tempo_nesse_trabalho_label" not in df.columns:
        print("  (no tenure label column -- skipping Route D)")
        return
    d = df.copy()
    d["state"] = state_UPR(d)
    d["bin"] = d["tempo_nesse_trabalho_label"].map(_tenure_bin)
    order = [(0.0, 1 / 12), (1 / 12, 1.0), (1.0, 2.0), (2.0, np.inf)]
    print("\nRoute D -- grouped-exponential tenure fit (per quarter):")
    for sector, name in [("P", "Public (delta_P)"), ("R", "Private (delta_R)")]:
        sub = d[(d["state"] == sector) & d["bin"].notna()]
        counts = np.array([(sub["bin"] == b).sum() for b in order], float)
        if counts.sum() < 40:
            print(f"  {name}: too few tenure obs ({int(counts.sum())})")
            continue
        res = minimize(grouped_exp_nll, x0=[np.log(0.5)], args=(counts,),
                       method="Nelder-Mead")
        q_yr = float(np.exp(res.x[0]))
        print(f"  {name}: pooled q = {q_yr:.4f}/yr = {q_yr / 4:.4f}/qtr  "
              f"(n={int(counts.sum()):,}, bins={counts.astype(int).tolist()})")


# ---------------------------------------------------------------------------
# Comparison table (the deliverable)
# ---------------------------------------------------------------------------
def comparison_table(Phat, QA, QB, PB):
    flows = [("U->P", "U", "P"), ("U->R", "U", "R"),
             ("P->U", "P", "U"), ("P->R", "P", "R"),
             ("R->U", "R", "U"), ("R->P", "R", "P")]
    rows = []
    for name, i, j in flows:
        rows.append({
            "flow": name,
            "raw_fraction": Phat.loc[i, j],
            "routeA_hazard": QA.loc[i, j],
            "routeB_hazard": QB.loc[i, j],
            "routeB_impliedP": PB.loc[i, j],
        })
    return pd.DataFrame(rows).set_index("flow")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tenure", action="store_true", help="Also run Route D.")
    ap.add_argument("--delta", type=float, default=1.0, help="Obs period in quarters.")
    args = ap.parse_args()

    df = load_panel()
    pairs, panel = consecutive_pairs(df)
    counts, Phat = empirical_matrices(pairs)

    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("\n=== Raw transition COUNTS (rows=origin, cols=dest) ===")
    print(counts.to_string())
    print("\n=== Empirical one-quarter transition matrix Phat (rows sum to 1) ===")
    print(Phat.to_string())

    QA = route_A(Phat, args.delta)
    print("\n=== Route A: closed-form competing-risks hazards (per quarter) ===")
    print(QA.to_string())

    QB, PB, res = route_B(counts, args.delta, init_from=QA)
    print(f"\n=== Route B: CTMC MLE generator Q (per quarter)  [converged={res.success}] ===")
    print(QB.to_string())
    print("\n--- Route B implied one-quarter P (compare to Phat) ---")
    print(PB.to_string())

    QC = route_C(Phat, args.delta)
    print("\n=== Route C: matrix-log of Phat (check; small negatives => trust B) ===")
    print(QC.to_string())

    print("\n" + "=" * 70)
    print("COMPARISON TABLE (the deliverable)")
    print("=" * 70)
    print(comparison_table(Phat, QA, QB, PB).to_string())

    print("\nImplied separation rates:  delta_P = qPU (Route B) = "
          f"{QB.loc['P', 'U']:.4f}/qtr,  delta_R = qRU = {QB.loc['R', 'U']:.4f}/qtr")
    print("Implied lambda0 (U-exit rate, acceptance~1) = "
          f"{QB.loc['U', 'P'] + QB.loc['U', 'R']:.4f}/qtr  "
          "(cf. handoff Flag 2: recalibrate lambda0 from this).")

    if args.tenure:
        route_D(panel)

    print("\nSTOP. Per handoff: no .tex edits and no downstream reservation-wage "
          "changes until Routes A and B are reviewed.")


if __name__ == "__main__":
    main()
