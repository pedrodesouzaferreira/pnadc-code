"""Tabulate the three headline PNADC labor-market statistics, by year.

For the harmonized panel, reports each year's:

  separation   : share of respondents employed (public or private sector) at
                 time t who are unemployed 4 quarters (12 months) later.
  job_finding  : among respondents unemployed at time t, the share employed
                 in the public sector 4 quarters later, and separately the
                 share employed in the private sector.
  wages        : average real (2025 Q3 BRL) usual monthly earnings among
                 respondents employed (public or private) at time t.

All statistics use the PNADC person weight (`peso`). Each table also reports
N, the unweighted respondent count entering that year's statistic, and
pop_N, its weighted sum -- so a thin year is visible next to its point
estimate.

These are simplified, unconditional versions of the sector/age/search-method
breakdowns computed by `10_fillins_pnadc_fulltime.do`; use that do-file for
the pooled I4 treatment-text numbers.

Usage
-----
    python 7_pnadc_fillins_survey.py                      # all 3 tables, higher-ed 2016-2025
    python 7_pnadc_fillins_survey.py --stat wages          # one table only
    python 7_pnadc_fillins_survey.py --sample full         # 2023-2024 only
    python 7_pnadc_fillins_survey.py --csv-dir "Cleaned Data/summary_stats_i4"

Requires the harmonized Parquet file produced by 2_harmonize.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from harmonized_data import add_sample_argument, load_harmonized

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "Cleaned Data"
LEAD_QUARTERS = 4

COLUMNS = [
    "id_pessoa", "ano", "trimestre", "peso",
    "empregado_setor_pub", "empregado_setor_priv",
    "conta_propria", "empregador", "trab_domestico",
    "desocupado", "trab_familiar_aux",
    "renda_habitual_principal_real_2025q3",
]

STATS = ("separation", "job_finding", "wages")


def _flag(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0).eq(1)


def build_panel(sample: str) -> pd.DataFrame:
    df = load_harmonized(CLEANED_DIR, sample, columns=COLUMNS)
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df["trimestre"] = pd.to_numeric(df["trimestre"], errors="coerce")
    df["peso"] = pd.to_numeric(df["peso"], errors="coerce")
    df["time"] = df["ano"] * 4 + df["trimestre"]

    df["employed_pub"] = _flag(df, "empregado_setor_pub")
    df["employed_priv"] = (
        _flag(df, "empregado_setor_priv")
        | _flag(df, "conta_propria")
        | _flag(df, "empregador")
        | _flag(df, "trab_domestico")
    )
    df["employed"] = df["employed_pub"] | df["employed_priv"]
    df["unemployed"] = _flag(df, "desocupado") | _flag(df, "trab_familiar_aux")
    df["wage_real"] = pd.to_numeric(
        df.get("renda_habitual_principal_real_2025q3"), errors="coerce"
    )

    df = df.sort_values(["id_pessoa", "time"]).reset_index(drop=True)
    print(f"Panel: {len(df):,} person-quarter rows, {df['id_pessoa'].nunique():,} individuals")
    return df


def with_lead(df: pd.DataFrame, columns: list[str], periods: int = LEAD_QUARTERS) -> pd.DataFrame:
    """Attach `<col>_f4`: the same person's value `periods` quarters later,
    only when the two observations are exactly `periods` quarters apart."""
    df = df.copy()
    grouped = df.groupby("id_pessoa")
    gap = grouped["time"].shift(-periods) - df["time"]
    on_time = gap.eq(periods)
    for column in columns:
        lead = grouped[column].shift(-periods)
        df[f"{column}_f4"] = lead.where(on_time)
    return df


def weighted_share(flag: pd.Series, weight: pd.Series) -> tuple[float, int, float]:
    flag = flag.astype(float)
    mask = flag.notna() & weight.notna() & weight.gt(0)
    if not mask.any():
        return np.nan, 0, 0.0
    return (
        float(np.average(flag[mask], weights=weight[mask])) * 100,
        int(mask.sum()),
        float(weight[mask].sum()),
    )


def weighted_mean(value: pd.Series, weight: pd.Series) -> tuple[float, int, float]:
    mask = value.notna() & weight.notna() & weight.gt(0)
    if not mask.any():
        return np.nan, 0, 0.0
    return (
        float(np.average(value[mask], weights=weight[mask])),
        int(mask.sum()),
        float(weight[mask].sum()),
    )


def yearly_separation(panel: pd.DataFrame) -> pd.DataFrame:
    leaded = with_lead(panel, ["unemployed"])
    base = leaded[leaded["employed"]]
    rows = []
    for year, group in base.groupby(base["ano"].astype("Int64")):
        value, n, pop_n = weighted_share(group["unemployed_f4"], group["peso"])
        rows.append({"ano": int(year), "separation_pct": value, "N": n, "pop_N": pop_n})
    return pd.DataFrame(rows).sort_values("ano").reset_index(drop=True)


def yearly_job_finding(panel: pd.DataFrame) -> pd.DataFrame:
    leaded = with_lead(panel, ["employed_pub", "employed_priv"])
    base = leaded[leaded["unemployed"]]
    rows = []
    for year, group in base.groupby(base["ano"].astype("Int64")):
        pub_value, pub_n, pub_pop_n = weighted_share(group["employed_pub_f4"], group["peso"])
        priv_value, priv_n, priv_pop_n = weighted_share(group["employed_priv_f4"], group["peso"])
        rows.append({
            "ano": int(year),
            "public_pct": pub_value, "public_N": pub_n, "public_pop_N": pub_pop_n,
            "private_pct": priv_value, "private_N": priv_n, "private_pop_N": priv_pop_n,
        })
    return pd.DataFrame(rows).sort_values("ano").reset_index(drop=True)


def yearly_wages(panel: pd.DataFrame) -> pd.DataFrame:
    base = panel[panel["employed"] & panel["wage_real"].gt(0)]
    rows = []
    for year, group in base.groupby(base["ano"].astype("Int64")):
        value, n, pop_n = weighted_mean(group["wage_real"], group["peso"])
        rows.append({"ano": int(year), "real_wage_2025q3": value, "N": n, "pop_N": pop_n})
    return pd.DataFrame(rows).sort_values("ano").reset_index(drop=True)


TABLE_BUILDERS = {
    "separation": yearly_separation,
    "job_finding": yearly_job_finding,
    "wages": yearly_wages,
}

TITLES = {
    "separation": "Separation: employed -> unemployed 12 months later (%)",
    "job_finding": "Job finding: unemployed -> employed 12 months later, by sector (%)",
    "wages": "Real usual monthly earnings among the employed (2025 Q3 BRL)",
}


def _pct(value: float) -> str:
    return f"{value:.1f}" if pd.notna(value) else "NA"


def _money(value: float) -> str:
    return f"{value:,.0f}" if pd.notna(value) else "NA"


def print_separation(table: pd.DataFrame) -> None:
    print(f"\n=== {TITLES['separation']} ===")
    fmt = table.assign(
        separation_pct=table["separation_pct"].map(_pct),
        N=table["N"].map("{:,d}".format),
        pop_N=table["pop_N"].map("{:,.0f}".format),
    )
    print(fmt.to_string(index=False))


def print_job_finding(table: pd.DataFrame) -> None:
    print(f"\n=== {TITLES['job_finding']} ===")
    fmt = table.assign(
        public_pct=table["public_pct"].map(_pct),
        public_N=table["public_N"].map("{:,d}".format),
        public_pop_N=table["public_pop_N"].map("{:,.0f}".format),
        private_pct=table["private_pct"].map(_pct),
        private_N=table["private_N"].map("{:,d}".format),
        private_pop_N=table["private_pop_N"].map("{:,.0f}".format),
    )
    print(fmt.to_string(index=False))


def print_wages(table: pd.DataFrame) -> None:
    print(f"\n=== {TITLES['wages']} ===")
    fmt = table.assign(
        real_wage_2025q3=table["real_wage_2025q3"].map(_money),
        N=table["N"].map("{:,d}".format),
        pop_N=table["pop_N"].map("{:,.0f}".format),
    )
    print(fmt.to_string(index=False))


TABLE_PRINTERS = {
    "separation": print_separation,
    "job_finding": print_job_finding,
    "wages": print_wages,
}


def maybe_write_csv(stat: str, table: pd.DataFrame, csv_dir: Path | None) -> None:
    if csv_dir is None:
        return
    csv_dir.mkdir(parents=True, exist_ok=True)
    path = csv_dir / f"pnadc_fillins_survey_{stat}.csv"
    table.to_csv(path, index=False)
    print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_sample_argument(parser, default="higher-ed")
    parser.add_argument("--stat", choices=("all",) + STATS, default="all")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Also write each table as a tidy CSV in this directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = build_panel(args.sample)
    stats = STATS if args.stat == "all" else (args.stat,)
    for stat in stats:
        table = TABLE_BUILDERS[stat](panel)
        TABLE_PRINTERS[stat](table)
        maybe_write_csv(stat, table, args.csv_dir)


if __name__ == "__main__":
    main()
