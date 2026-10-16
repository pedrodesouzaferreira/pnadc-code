#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_earnings_age_profile_beliefs.py  —  The PNADC earnings-by-age figures of 12_earnings_age_profile.py with
                                       the UnB pilot respondents' own wage beliefs overlaid, for benchmarking.

INPUT   Output/Tables/pnadc_earnings_age_profile_<year>.csv           (run 12_earnings_age_profile.py first)
        ../PilotoUnB/Clean Data/piloto_clean.dta                       (pilot survey, cleaned by 02_clean.do)
        ../PilotoUnB/Raw Data/matriculas_planilha_mestra.csv           (enrolment roster, for the sample filter)
OUTPUT  Output/Tables/pilot_wage_beliefs_summary.csv    aggregate belief statistics only (no identifiers)
        Output/Figures/pnadc_earnings_age_profile_<year>_beliefs_{median,mean}.{pdf,png}
        --single: reads pnadc_earnings_age_profile_<year>_single.csv (run 12 with --half-window 0 first) and writes
        ..._<year>_single_beliefs_{median,mean}.{pdf,png}: PNADC lines at single years of age, no smoothing
        --years 2016-2019,2022-2024 --prices 2024: reads the pooled table written by
        12 --years 2016-2019,2022-2024 --prices 2024 [--half-window 0] (tag and suffix must match)

PILOT SAMPLE — exactly the analysis sample of PilotoUnB/Code/05_analysis.do (section 0c): Piloto 2
(sample == 3), reached the prize question (f8_sorteio_escolha non-missing), matched to the enrolment roster.
Beliefs are PRE-treatment (block E3B, asked before any information screen), so all arms are pooled.

BELIEF ITEMS (from Code/variable_crosswalk.csv)
  e3b_{pub,priv}_wage_med_2224   believed MEDIAN wage of college-educated 22-24 y.o., public concursado / private
                                 -> drawn at ages 22-24 (diamond, horizontal bar = the age range)
  e3b_{pub,priv}_wage_med_all    believed MEDIAN wage of college-educated workers, all ages
                                 -> dashed horizontal line across 22-60
  e3b_own_wage_{publico,privado} own most-likely gross wage if concursado / private
                                 -> hollow circle at x = 26 (the question carries no age; set apart from the
                                    22-24 diamond so the markers do not overlap)
The survey asks about MEDIAN wages, so the median figure is the like-for-like benchmark. The mean figure is
provided because it was asked for: there the belief markers are the mean (across respondents) of what each
respondent believes the median is, winsorised at p1/p99 as in 05_analysis.do; it is not a belief about the mean.
Across respondents: median figure uses the median belief, mean figure the winsorised mean; vertical bars are
90% bootstrap intervals (2,000 draws), matching the deck's 90% CIs.
Beliefs were elicited in Aug-Sep 2026 reais, PNADC is in 2025-Q3 reais (or average-<prices> reais with --prices);
no adjustment is made for the gap.
Beliefs concern college-educated workers in general (not DF-specific); the same markers are drawn in both
panels. 77% of respondents went to high school in DF.
The PNADC private line is private employees with carteira (see 12_earnings_age_profile.py); the PNADC "truth"
shown in the pilot used all private employees, so the private line here sits slightly above that number.

Never writes respondent-level data. Re-runnable; overwrites its outputs. Does not modify any existing script.
"""
import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PILOT = ROOT.parent / "PilotoUnB"
PILOT_DTA = PILOT / "Clean Data" / "piloto_clean.dta"
ROSTER = PILOT / "Raw Data" / "matriculas_planilha_mestra.csv"
OUT_TAB = ROOT / "Output" / "Tables"
OUT_FIG = ROOT / "Output" / "Figures"

AGE_MIN, AGE_MAX = 22, 60
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
COLOR = {"Public": BLUE, "Private": ORANGE}
ITEMS = {  # (sector, object) -> variable
    ("Public", "2224"): "e3b_pub_wage_med_2224", ("Private", "2224"): "e3b_priv_wage_med_2224",
    ("Public", "all"): "e3b_pub_wage_med_all", ("Private", "all"): "e3b_priv_wage_med_all",
    ("Public", "own"): "e3b_own_wage_publico", ("Private", "own"): "e3b_own_wage_privado",
}
N_BOOT, SEED = 2000, 20261015
OWN_X = 26          # x-position of the own-expectation markers (the question carries no age)


def pilot_sample():
    d = pd.read_stata(PILOT_DTA, convert_categoricals=False)
    d = d[(d["sample"] == 3) & d.f8_sorteio_escolha.notna()].copy()
    roster = pd.read_csv(ROSTER, dtype=str)
    roster = roster.rename(columns={c: c.strip().lower() for c in roster.columns})
    mats = set(roster["matricula"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True))
    d["c0_matricula"] = d["c0_matricula"].astype(str).str.strip()
    return d[d.c0_matricula.isin(mats)]


def winsor(x):
    lo, hi = np.percentile(x, [1, 99])
    return np.clip(x, lo, hi)


def summarize(d):
    rng = np.random.default_rng(SEED)
    rows = []
    for (sec, obj), v in ITEMS.items():
        x = d[v].dropna().to_numpy(float)
        xw = winsor(x)
        idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
        bmed = np.median(x[idx], axis=1)
        bmean = winsor_rows(x[idx])
        rows.append({"sector": sec, "object": obj, "variable": v, "n": len(x),
                     "median": np.median(x), "median_lo90": np.percentile(bmed, 5), "median_hi90": np.percentile(bmed, 95),
                     "mean_w": xw.mean(), "mean_w_lo90": np.percentile(bmean, 5), "mean_w_hi90": np.percentile(bmean, 95)})
    return pd.DataFrame(rows)


def winsor_rows(m):
    lo = np.percentile(m, 1, axis=1, keepdims=True)
    hi = np.percentile(m, 99, axis=1, keepdims=True)
    return np.clip(m, lo, hi).mean(axis=1)


def brl(x, _=None):
    return f"R$ {x:,.0f}".replace(",", ".")


def plot(p, b, year, stat, n_resp, name, single=False, prices="2025q3"):
    plab = "2025-Q3" if prices == "2025q3" else f"average-{prices}"
    year = str(year).replace("_", ", ")
    s_col = "median" if stat == "median" else "mean_w"
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharey=True)
    for ax, reg in zip(axes, ["Brazil", "Distrito Federal"]):
        q = p[p.region == reg]
        ends = {}
        for sec in ("Public", "Private"):
            col = COLOR[sec]
            r = q[q.sector == sec]
            if single:
                ax.plot(r.age, r[f"{stat}_window"], color=col, linewidth=1.5, marker="o", ms=2.5, zorder=3)
            else:
                ax.plot(r.age, r[f"{stat}_window"], color=col, linewidth=2, zorder=3)
                ax.scatter(r.age, r[f"{stat}_age"], s=7, color=col, alpha=0.25, linewidths=0, zorder=2)
            bb = b[b.sector == sec].set_index("object")
            # all ages: dashed line
            ax.hlines(bb.loc["all", s_col], AGE_MIN, AGE_MAX, colors=col, linestyles=(0, (5, 3)), linewidth=1.3, zorder=2)
            # 22-24: diamond spanning the age range, 90% CI
            y = bb.loc["2224", s_col]
            ax.errorbar(23, y, xerr=1, yerr=[[y - bb.loc["2224", f"{s_col}_lo90"]], [bb.loc["2224", f"{s_col}_hi90"] - y]],
                        fmt="D", ms=7, color=col, mec="white", mew=1.0, elinewidth=1.4, capsize=0, zorder=5)
            # own expectation: hollow circle, set to the right of the 22-24 diamond so they do not overlap
            y = bb.loc["own", s_col]
            ax.errorbar(OWN_X, y, yerr=[[y - bb.loc["own", f"{s_col}_lo90"]], [bb.loc["own", f"{s_col}_hi90"] - y]],
                        fmt="o", ms=7, mfc="white", mec=col, mew=1.6, color=col, elinewidth=1.0, capsize=0, zorder=4)
            ends[sec] = r[f"{stat}_window"].iloc[-1]
        ymax = max(p[f"{stat}_window"].max(), b[s_col].max()) * 1.05
        gap = 0.05 * ymax                                # keep the two end labels apart
        if abs(ends["Public"] - ends["Private"]) < gap:
            mid = (ends["Public"] + ends["Private"]) / 2
            hi, lo = ("Public", "Private") if ends["Public"] >= ends["Private"] else ("Private", "Public")
            ends[hi], ends[lo] = mid + gap / 2, mid - gap / 2
        for sec in ("Public", "Private"):
            ax.text(AGE_MAX + 0.6, ends[sec], sec, color=INK, fontsize=8.5, va="center", ha="left")
        ax.set_title(reg, loc="left", fontsize=10.5, color=INK)
        ax.set_xlim(AGE_MIN - 1.5, AGE_MAX + 6)
        ax.set_xticks(range(25, AGE_MAX + 1, 5))
        ax.set_xlabel("age", color=INK2)
        ax.yaxis.set_major_formatter(FuncFormatter(brl))
        ax.tick_params(colors=INK2, labelsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
    axes[0].set_ylim(bottom=0)
    lab = "Median" if stat == "median" else "Mean"
    axes[0].set_ylabel(f"{lab.lower()} monthly earnings (R$)", color=INK2)
    handles = [Line2D([], [], color=INK2, lw=2, label=f"PNADC {year}, {lab.lower()} by age" + (" (single years)" if single else " (5-year window)")),
               Line2D([], [], color=INK2, lw=1.3, ls=(0, (5, 3)), label="belief: median wage, college-educated, all ages"),
               Line2D([], [], color=INK2, marker="D", ls="none", ms=6, label="belief: median wage, college-educated aged 22–24"),
               Line2D([], [], color=INK2, marker="o", mfc="white", ls="none", ms=6, label="own expected wage in that sector (undated; drawn at 26)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.915), ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"{lab} earnings by age vs. UnB students' beliefs: college-educated full-time formal workers",
                 x=0.01, ha="left", fontsize=12, color=INK)
    across = ("median across respondents" if stat == "median"
              else "mean across respondents, winsorised at p1/p99 (of beliefs about the MEDIAN wage — the like-for-like "
                   "benchmark is the median figure)")
    note = (f"PNADC: IBGE PNADC {year}, completed tertiary, usual hours ≥ 35, positive habitual main-job earnings (R$ of {plab}), "
            "public = statutory/military/public CLT excluding temporary contracts, private = employees with carteira; first "
            "interview per person; weights V1028 (see 12_earnings_age_profile.py). Beliefs: UnB pilot (Piloto 2, analysis sample "
            f"of 05_analysis.do, N = {n_resp}), pre-treatment, all arms pooled; markers are the {across}; vertical bars are "
            "90% bootstrap intervals. Beliefs are about college-educated workers in general (same markers in both panels) and "
            "are in Aug–Sep 2026 reais. Blue = public, orange = private.")
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(note, 205)), fontsize=6.3, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.08, right=0.95, top=0.78, bottom=0.2, wspace=0.12)
    for ext, dpi in (("pdf", None), ("png", 200)):
        fig.savefig(OUT_FIG / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--year", default="2024", help="year of the 12_earnings_age_profile.py table to use")
    ap.add_argument("--years", default=None, help="pooled-years tag used in 12 --years (overrides --year)")
    ap.add_argument("--prices", default="2025q3", help="price base used in 12 --prices")
    ap.add_argument("--single", action="store_true", help="use the single-year-of-age table (12 --half-window 0)")
    a = ap.parse_args()
    if a.years:
        a.year = a.years.replace(",", "_")
    sfx = ("_single" if a.single else "") + ("" if a.prices == "2025q3" else f"_p{a.prices}")
    tab = OUT_TAB / f"pnadc_earnings_age_profile_{a.year}{sfx}.csv"
    if not tab.exists():
        raise SystemExit(f"{tab} not found: run 12_earnings_age_profile.py with the matching --year/--years/--prices"
                         + (" --half-window 0" if a.single else "") + " first")
    p = pd.read_csv(tab)
    d = pilot_sample()
    b = summarize(d)
    b.insert(0, "n_respondents", len(d))
    b.to_csv(OUT_TAB / "pilot_wage_beliefs_summary.csv", index=False)
    for stat in ("median", "mean"):
        plot(p, b, a.year, stat, len(d), f"pnadc_earnings_age_profile_{a.year}{sfx}_beliefs_{stat}", a.single, a.prices)
    # benchmark printout: PNADC at 22-24 vs beliefs
    for reg in ("Brazil", "Distrito Federal"):
        for sec in ("Public", "Private"):
            r = p[(p.region == reg) & (p.sector == sec) & (p.age == 23)].iloc[0]
            bb = b[(b.sector == sec)].set_index("object")
            print(f"{reg:16s} {sec:7s} age 23{' ' if a.single else ' window'}: PNADC median {r.median_window:8,.0f} mean {r.mean_window:8,.0f} | "
                  f"belief 22-24 median {bb.loc['2224', 'median']:8,.0f} | own {bb.loc['own', 'median']:8,.0f}")
    print(b.round(0).to_string(index=False))
    print(f"N respondents = {len(d)}\n-> {OUT_TAB / 'pilot_wage_beliefs_summary.csv'}\n-> "
          f"{OUT_FIG}/pnadc_earnings_age_profile_{a.year}{sfx}_beliefs_{{median,mean}}.pdf (+png)")


if __name__ == "__main__":
    main()
