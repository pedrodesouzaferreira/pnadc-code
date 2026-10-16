#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_earnings_age_profile.py  —  Earnings-by-age profile of college-educated, full-time formal workers,
                               public vs private, Brazil vs Distrito Federal, one PNADC year.

INPUT   Cleaned Data/PNADC_harmonized_VD3004_7_2016_2025_light.csv
          (output of 2_harmonize.py + 2a_harmonize_makelighter.py; respondents with a completed
          tertiary degree only, VD3004 == 7; wages already deflated to 2025-Q3 prices)
OUTPUT  Output/Tables/pnadc_earnings_age_profile_<year>.csv
          one row per region x sector x age (22-60): weighted mean and median earnings for the single age
          and for a centred 5-year window, approximate 95% band for the window mean, unweighted N, sum of weights
        Output/Figures/pnadc_earnings_age_profile_<year>_mean.{pdf,png}
        Output/Figures/pnadc_earnings_age_profile_<year>_median.{pdf,png}

Usage   python 12_earnings_age_profile.py                # latest year with all four quarters (2024 now)
        python 12_earnings_age_profile.py --year 2023 --obs all
        python 12_earnings_age_profile.py --half-window 0         # single-year ages, no smoothing ("_single" files)
        python 12_earnings_age_profile.py --years 2016-2019,2022-2024 --prices 2024 --half-window 0
                                     # pool several years, wages in average-2024 reais

POOLING (--years)  Every listed year is read; "first interview of each person" is then taken over the pooled
panel (the .do's first_obs: bys id_pessoa (ano trimestre)), so a person interviewed across a year boundary
counts once. Weights stay peso. Output tag = the --years string (e.g. 2016-2019_2022-2024).
PRICES (--prices)  The input wages are in 2025-Q3 reais (IBGE CO2 deflator by UF; 2_harmonize.py). --prices 2024
rescales them to average-2024 reais with one national factor k = weighted mean over all 2024 interviews with
positive earnings of nominal / real_2025q3 (k ≈ 0.948). A single scalar, so shapes and gaps are unchanged.

SAMPLE CUT — follows the PILOT 2 globals in 7_pnadc_fillins_survey.do ($pub_first_2022 / $pri_first_2022),
restricted to one calendar year:
  * college educated: the input file only has VD3004 == 7 (completed tertiary);
  * full time: horas_habituais_principal >= 35 (non-missing);
  * positive earnings: renda_habitual_principal_real_2025q3 > 0 (habitual monthly earnings, main job,
    2025-Q3 reais) — "earnings conditional on employment";
  * public = empregado_setor_pub_refined, built exactly as in the .do (and in 10_fillins_pnadc_fulltime.do):
        servidor_publico_estatutario == 1
        | (V4010 in {5412, 511, 512, 411, 412} & posicao_trab_principal == 2)     [military codes]
        | (posicao_trab_principal == 4 & carteira_assinada == 1)                  [public CLT]
      and set to 0 when V4025 == 1 (temporary contract);
  * private = empregado_setor_priv == 1 & formal == 1, i.e. VD4009 == 1, private employee WITH carteira.
    (The .do's $pri global uses empregado_setor_priv alone, which also includes private employees without
    carteira; restricting to carteira matches "emprego formal". Use --private-all to drop the restriction.)
  * one observation per person: the person's first interview in the chosen year (the .do's first_obs, taken
    within the year). --obs all keeps every quarterly interview instead (each quarter is a full sample, so
    weights are divided by the number of quarters).
  * weights: peso (V1028), as in the .do's [aweight = peso].
  * ages 22-60; DF = id_uf == 53.

ESTIMATES  For each age a, the plotted line is the weighted mean (or median) over people aged a-h..a+h
(h = --half-window, default 2, i.e. a centred 5-year window); the dots are the single-age values. With
--half-window 0 the line IS the single-age value (no dots; outputs get the suffix "_single").
Band: mean ± 1.96 × sd_w / sqrt(n_eff), n_eff = (Σw)² / Σw² (ignores the PNADC cluster/strata design, so it
is too narrow). DF cells are small (~20 people per age per sector), which is why the default smooths.

No existing script is modified. Re-runnable; overwrites its outputs. Python + pandas + matplotlib only.
"""
import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IN = ROOT / "Cleaned Data" / "PNADC_harmonized_VD3004_7_2016_2025_light.csv"
OUT_TAB = ROOT / "Output" / "Tables"
OUT_FIG = ROOT / "Output" / "Figures"

COLS = ["ano", "trimestre", "id_uf", "id_pessoa", "peso", "idade", "servidor_publico_estatutario",
        "carteira_assinada", "V4010", "V4025", "posicao_trab_principal", "horas_habituais_principal",
        "renda_habitual_principal_real_2025q3", "formal", "empregado_setor_priv"]
WAGE = "renda_habitual_principal_real_2025q3"
AGE_MIN, AGE_MAX, HALF_WINDOW = 22, 60, 2
MILITARY_V4010 = [5412, 511, 512, 411, 412]
DF_CODE = 53

BLUE, ORANGE = "#2a78d6", "#eb6834"          # public, private (validated categorical pair)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SECTOR_COLOR = {"Public": BLUE, "Private": ORANGE}


def parse_years(spec):
    out = []
    for part in str(spec).split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def load(year=None, years=None):
    """year: one year (None = latest with 4 quarters); years: list of years to pool (overrides year)."""
    chunks = []
    for c in pd.read_csv(IN, usecols=COLS + ["renda_habitual_principal"], chunksize=500_000, low_memory=False):
        chunks.append(c)
    d = pd.concat(chunks, ignore_index=True)
    if years:
        return d[d.ano.isin(years)].copy(), d
    if year is None:                                   # latest year with all four quarters
        q = d.groupby("ano").trimestre.nunique()
        year = int(q[q == 4].index.max())
    return d[d.ano == year].copy(), d


def price_factor(full, prices):
    """Multiply 2025-Q3 reais by this to express them in average-<prices> reais."""
    if prices == "2025q3":
        return 1.0
    y = full[(full.ano == int(prices)) & (full.renda_habitual_principal > 0) & full[WAGE].notna()]
    return float(np.average(y.renda_habitual_principal / y[WAGE], weights=y.peso))


def build_sample(d, obs="first", private_all=False, hw=HALF_WINDOW):
    d = d.copy()
    public_rule = ((d.servidor_publico_estatutario == 1)
                   | (d.V4010.isin(MILITARY_V4010) & (d.posicao_trab_principal == 2))
                   | ((d.posicao_trab_principal == 4) & (d.carteira_assinada == 1)))
    d["public"] = public_rule & ~(d.V4025 == 1)
    d["private"] = (d.empregado_setor_priv == 1) & (True if private_all else d.formal == 1)
    n_q = d.groupby("ano").trimestre.nunique().sum()          # quarters in the sample (all years)
    if obs == "first":
        d = d.sort_values(["id_pessoa", "ano", "trimestre"]).groupby("id_pessoa", sort=False).head(1)
        d["w"] = d.peso
    else:
        d["w"] = d.peso / n_q
    keep = ((d.horas_habituais_principal >= 35) & d.horas_habituais_principal.notna()
            & (d[WAGE] > 0) & d.idade.between(AGE_MIN - hw, AGE_MAX + hw)
            & (d.public | d.private))
    d = d[keep].copy()
    d["sector"] = np.where(d.public, "Public", "Private")
    return d


def wstats(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    if len(x) == 0:
        return dict(mean=np.nan, median=np.nan, se=np.nan, n=0, sumw=0.0)
    m = np.average(x, weights=w)
    var = np.average((x - m) ** 2, weights=w)
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    med = x[o][np.searchsorted(cw, 0.5)]
    return dict(mean=m, median=med, se=np.sqrt(var / n_eff), n=len(x), sumw=w.sum())


def profile(s, hw=HALF_WINDOW):
    rows = []
    regions = {"Brazil": s, "Distrito Federal": s[s.id_uf == DF_CODE]}
    for reg, r in regions.items():
        for sec in ("Public", "Private"):
            g = r[r.sector == sec]
            for a in range(AGE_MIN, AGE_MAX + 1):
                one = g[g.idade == a]
                win = g[(g.idade - a).abs() <= hw]
                s1, sw = wstats(one[WAGE], one.w), wstats(win[WAGE], win.w)
                rows.append({"region": reg, "sector": sec, "age": a,
                             "mean_age": s1["mean"], "median_age": s1["median"], "n_age": s1["n"],
                             "mean_window": sw["mean"], "median_window": sw["median"],
                             "ci_lo_window": sw["mean"] - 1.96 * sw["se"], "ci_hi_window": sw["mean"] + 1.96 * sw["se"],
                             "n_window": sw["n"], "sumw_window": sw["sumw"]})
    return pd.DataFrame(rows)


def brl(x, _=None):
    return f"R$ {x:,.0f}".replace(",", ".")


def plot(p, year, stat, obs, private_all, name, hw=HALF_WINDOW, prices="2025q3"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for ax, reg in zip(axes, ["Brazil", "Distrito Federal"]):
        q = p[p.region == reg]
        ends = {}
        for sec in ("Public", "Private"):
            r = q[q.sector == sec]
            col = SECTOR_COLOR[sec]
            if stat == "mean":
                ax.fill_between(r.age, r.ci_lo_window, r.ci_hi_window, color=col, alpha=0.15, linewidth=0)
            if hw > 0:
                ax.scatter(r.age, r[f"{stat}_age"], s=9, color=col, alpha=0.35, linewidths=0, zorder=2)
            ax.plot(r.age, r[f"{stat}_window"], color=col, linewidth=2 if hw > 0 else 1.6, zorder=3,
                    marker=None if hw > 0 else "o", ms=3)
            ends[sec] = r[f"{stat}_window"].iloc[-1]
        gap = 0.05 * p[f"{stat}_window"].max()           # keep the two end labels apart
        if abs(ends["Public"] - ends["Private"]) < gap:
            mid = (ends["Public"] + ends["Private"]) / 2
            hi, lo = ("Public", "Private") if ends["Public"] >= ends["Private"] else ("Private", "Public")
            ends[hi], ends[lo] = mid + gap / 2, mid - gap / 2
        for sec in ("Public", "Private"):
            ax.text(AGE_MAX + 0.6, ends[sec], sec, color=INK, fontsize=8.5, va="center", ha="left")
        n_pub = int(q[q.sector == "Public"].n_age.sum())
        n_pri = int(q[q.sector == "Private"].n_age.sum())
        fmt = lambda n: f"{n:,}".replace(",", ".")
        ax.set_title(f"{reg}   (N = {fmt(n_pub)} public, {fmt(n_pri)} private)",
                     loc="left", fontsize=10, color=INK)
        ax.set_xlim(AGE_MIN - 1, AGE_MAX + 5)
        ax.set_xticks(range(25, AGE_MAX + 1, 5))
        ax.set_xlabel("age", color=INK2)
        ax.yaxis.set_major_formatter(FuncFormatter(brl))
        ax.tick_params(colors=INK2, labelsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
    axes[0].set_ylim(bottom=0)
    lab = "Mean" if stat == "mean" else "Median"
    plab = "2025-Q3" if prices == "2025q3" else f"average {prices}"
    axes[0].set_ylabel(f"{lab.lower()} habitual monthly earnings, main job (R$ of {plab})", color=INK2)
    ylab = year if str(year).isdigit() else str(year).replace("_", ", ")
    fig.suptitle(f"{lab} earnings by age, college-educated full-time formal workers, PNADC {ylab}",
                 x=0.01, ha="left", fontsize=12, color=INK)
    priv = "private employees (with or without carteira)" if private_all else "private employees with carteira (VD4009 = 1)"
    pooled = not str(year).isdigit()
    unit = (("first interview of each person in the pooled years" if pooled else "first interview of each person in the year")
            if obs == "first" else "all quarterly interviews (weights / number of quarters)")
    note = (f"Source: IBGE PNADC {ylab}, Base dos Dados extract with completed tertiary education (VD3004 = 7). Sample: usual "
            f"hours in main job ≥ 35, positive habitual earnings, {unit}, weights V1028. Public: statutory servants, "
            "military, or public employees with carteira, excluding temporary contracts (as in 7_pnadc_fillins_survey.do). "
            f"Private: {priv}. "
            + (f"Lines: {lab.lower()} over a centred {2 * hw + 1}-year age window; dots: single-year ages"
               if hw > 0 else f"Lines: {lab.lower()} at each single year of age (no smoothing)")
            + (f"; band: ±1.96 s.e. of the {'window ' if hw > 0 else ''}mean, ignoring the survey design."
               if stat == "mean" else ".")
            + (" Earnings deflated to 2025-Q3 prices." if prices == "2025q3" else
               f" Earnings in average-{prices} reais (IBGE CO2 deflator to 2025-Q3, rescaled by one national factor)."))
    fig.text(0.01, 0.01, "\n".join(textwrap.wrap(note, 200)), fontsize=6.4, color=INK2, va="bottom")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.86, bottom=0.2, wspace=0.12)
    for ext, dpi in (("pdf", None), ("png", 200)):
        fig.savefig(OUT_FIG / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--year", type=int, default=None, help="PNADC year (default: latest with 4 quarters)")
    ap.add_argument("--obs", choices=["first", "all"], default="first")
    ap.add_argument("--private-all", action="store_true", help="include private employees without carteira")
    ap.add_argument("--years", default=None, help="pool years, e.g. 2016-2019,2022-2024 (overrides --year)")
    ap.add_argument("--prices", default="2025q3", help="price base: 2025q3 (input) or a year, e.g. 2024")
    ap.add_argument("--half-window", type=int, default=HALF_WINDOW,
                    help="smoothing half-width in years (default 2 = 5-year window; 0 = single-year ages)")
    a = ap.parse_args()
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    years = parse_years(a.years) if a.years else None
    d, full = load(a.year, years)
    year = a.years.replace(",", "_") if years else int(d.ano.iloc[0])
    k = price_factor(full, a.prices)
    del full
    d[WAGE] = d[WAGE] * k
    hw = a.half_window
    s = build_sample(d, a.obs, a.private_all, hw)
    p = profile(s, hw)
    p.insert(0, "year", year)
    suffix = "" if (a.obs == "first" and not a.private_all) else f"_{a.obs}" + ("_privall" if a.private_all else "")
    suffix += "" if hw == HALF_WINDOW else ("_single" if hw == 0 else f"_hw{hw}")
    suffix += "" if a.prices == "2025q3" else f"_p{a.prices}"
    out = OUT_TAB / f"pnadc_earnings_age_profile_{year}{suffix}.csv"
    p.to_csv(out, index=False)
    for stat in ("mean", "median"):
        plot(p, year, stat, a.obs, a.private_all, f"pnadc_earnings_age_profile_{year}{suffix}_{stat}", hw, a.prices)

    s22 = s[s.idade.between(AGE_MIN, AGE_MAX)]
    print(f"price factor 2025q3 -> {a.prices}: {k:.4f}")
    if years:
        print("people by year of first interview:", s.groupby("ano").size().to_dict())
    print(f"PNADC {year}: {len(d):,} interviews, {d.id_pessoa.nunique():,} people; sample ages {AGE_MIN}-{AGE_MAX}: "
          f"{len(s22):,} ({(s22.sector == 'Public').sum():,} public, {(s22.sector == 'Private').sum():,} private; "
          f"DF {(s22.id_uf == DF_CODE).sum():,})")
    show = p[p.age.isin([22, 25, 30, 35, 40, 45, 50, 55, 60])].pivot_table(
        index="age", columns=["region", "sector"], values="mean_window").round(0)
    print(show.to_string())
    print(f"-> {out}\n-> {OUT_FIG}/pnadc_earnings_age_profile_{year}{suffix}_{{mean,median}}.pdf (+png)")


if __name__ == "__main__":
    main()
