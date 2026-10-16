#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_pnadc_fillins_wages_df.py  —  The wage fill-ins of 7_pnadc_fillins_survey.do (p50 and p90 of habitual
                                 earnings, public vs private, all ages and 22-24), for Distrito Federal.

INPUT   Cleaned Data/PNADC_harmonized_VD3004_7_2016_2025_light.csv   (same file the .do imports)
OUTPUT  Output/Tables/pnadc_fillins_wages_DF.csv
          one row per pilot definition x region x group: p50, p90, count (unweighted N), as in the .do's
          esttab cells("p50 p90 count"). Brazil rows are kept as a check: they must equal
          Output/pnadc_pilot1_wages.csv and Output/pnadc_pilot2_wages.csv.

DEFINITIONS — copied from 7_pnadc_fillins_survey.do, nothing added:
  earnings   renda_habitual_principal_real_2025q3 (Stata: renda_habitual_principal_real_20), 2025-Q3 reais
  public     empregado_setor_pub_refined: servidor_publico_estatutario == 1
             | (v4010 in 5412 511 512 411 412 & posicao_trab_principal == 2)
             | (posicao_trab_principal == 4 & carteira_assinada == 1); set to 0 if v4025 == 1
  private    empregado_setor_priv == 1   (with or without carteira, as in the .do)
  PILOT 1    $pub_3qrt_2022 / $pri_3qrt_2022: horas_habituais_principal >= 35, ano >= 2022, trimestre == 3,
             earnings > 0; every 3rd-quarter interview. These are the numbers shown in the pilot survey.
  PILOT 2    $pub_first_2022 / $pri_first_2022: horas >= 35, ano >= 2022, first_obs == 1 (person's first
             interview in the whole 2016-2025 panel: bys id_pessoa (ano trimestre): _n == 1); no earnings > 0
             condition (as in the .do).
  22-24      idade in 22..24.   DF = id_uf == 53 (the only line not in the .do).
  weights    [aweight = peso]; percentiles follow Stata's summarize, detail rule for weighted data
             (first observation whose cumulative weight exceeds p% of the total; average of two neighbours
             on an exact tie).
LOOSENED VERSIONS (second output, pnadc_fillins_wages_DF_loosened.csv) — NOT the .do's definitions. The DF
public 22-24 cell has 6 (PILOT 1) / 8 (PILOT 2) people, so the young group is re-estimated with:
  years  "2022+"  (as in the .do)   or  "2016+" (all years in the file: 2016-2019, 2022-2025Q3);
  ages   22-24 (as in the .do)       or  22-26;
everything else as in PILOT 1 / PILOT 2 (for "2016+", first_obs is still the first interview in the panel).
Earnings stay in 2025-Q3 reais (already deflated), so pooling years needs no extra adjustment.
Re-runnable; overwrites its outputs. Does not modify any existing script.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IN = ROOT / "Cleaned Data" / "PNADC_harmonized_VD3004_7_2016_2025_light.csv"
OUT = ROOT / "Output" / "Tables" / "pnadc_fillins_wages_DF.csv"
OUT_LOOSE = ROOT / "Output" / "Tables" / "pnadc_fillins_wages_DF_loosened.csv"
W = "renda_habitual_principal_real_2025q3"
COLS = ["ano", "trimestre", "id_uf", "id_pessoa", "peso", "idade", "servidor_publico_estatutario",
        "carteira_assinada", "V4010", "V4025", "posicao_trab_principal", "horas_habituais_principal", W,
        "empregado_setor_priv", "empregado_setor_pub"]


def stata_pct(x, w, p):
    o = np.argsort(x, kind="mergesort")
    x, w = x[o], w[o]
    c = np.cumsum(w)
    target = c[-1] * p / 100
    i = np.searchsorted(c, target, side="right")
    if i > 0 and np.isclose(c[i - 1], target):
        return (x[i - 1] + x[i]) / 2
    return x[i]


def main():
    d = pd.read_csv(IN, usecols=COLS, low_memory=False)
    d = d.sort_values(["id_pessoa", "ano", "trimestre"], kind="mergesort")
    d["first_obs"] = ~d.duplicated("id_pessoa")
    rule = ((d.servidor_publico_estatutario == 1)
            | (d.V4010.isin([5412, 511, 512, 411, 412]) & (d.posicao_trab_principal == 2))
            | ((d.posicao_trab_principal == 4) & (d.carteira_assinada == 1)))
    d["pub_refined"] = np.where(d.empregado_setor_pub.notna(), (rule & ~(d.V4025 == 1)).astype(float), np.nan)
    ft = (d.horas_habituais_principal >= 35) & d.horas_habituais_principal.notna()
    post = d.ano >= 2022
    defs = {
        "PILOT 1 (3rd quarter, shown in survey)": ((d.pub_refined == 1) & ft & post & (d.trimestre == 3) & (d[W] > 0),
                                                   (d.empregado_setor_priv == 1) & ft & post & (d.trimestre == 3) & (d[W] > 0)),
        "PILOT 2 (first observation)": ((d.pub_refined == 1) & ft & post & d.first_obs,
                                        (d.empregado_setor_priv == 1) & ft & post & d.first_obs),
    }
    regions = {"Brazil": pd.Series(True, index=d.index), "Distrito Federal": d.id_uf == 53}
    young = d.idade.between(22, 24)
    rows = []
    for pilot, (pub, pri) in defs.items():
        for reg, r in regions.items():
            for grp, m in [("Publico", pub), ("Privado", pri), ("Publico 22-24", pub & young), ("Privado 22-24", pri & young)]:
                s = d[m & r & d[W].notna()]
                x, w = s[W].to_numpy(float), s.peso.to_numpy(float)
                rows.append({"definition": pilot, "region": reg, "group": grp,
                             "p50": round(stata_pct(x, w, 50), 2) if len(s) else np.nan,
                             "p90": round(stata_pct(x, w, 90), 2) if len(s) else np.nan,
                             "count": len(s)})
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    with pd.option_context("display.width", 160):
        print(out.to_string(index=False))
    print(f"-> {OUT}")

    # ---- loosened young-group versions (not the .do's definitions)
    rows = []
    years = {"2022+": d.ano >= 2022, "2016+": d.ano >= 2016}
    ages = {"22-24": d.idade.between(22, 24), "22-26": d.idade.between(22, 26)}
    for pilot in ("PILOT 1", "PILOT 2"):
        for yl, ym in years.items():
            base = ft & ym & ((d.trimestre == 3) & (d[W] > 0) if pilot == "PILOT 1" else d.first_obs)
            for al, am in ages.items():
                for reg, r in regions.items():
                    for sec, sm in (("Publico", d.pub_refined == 1), ("Privado", d.empregado_setor_priv == 1)):
                        s = d[base & am & sm & r & d[W].notna()]
                        x, w = s[W].to_numpy(float), s.peso.to_numpy(float)
                        rows.append({"definition": pilot, "years": yl, "ages": al, "region": reg, "group": sec,
                                     "p50": round(stata_pct(x, w, 50), 2) if len(s) else np.nan,
                                     "p90": round(stata_pct(x, w, 90), 2) if len(s) else np.nan,
                                     "count": len(s),
                                     "same_as_do": int(yl == "2022+" and al == "22-24")})
    loose = pd.DataFrame(rows)
    loose.to_csv(OUT_LOOSE, index=False)
    with pd.option_context("display.width", 160):
        print(loose[loose.region == "Distrito Federal"].to_string(index=False))
    print(f"-> {OUT_LOOSE}")


if __name__ == "__main__":
    main()
