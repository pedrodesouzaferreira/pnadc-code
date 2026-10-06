# pnadc-code

Data pipeline and estimation code for **"The Value of Government Jobs"** (Pedro Ferreira, Harvard Kennedy School, dissertation in progress). The project studies how Brazilian workers choose between preparing for civil-service exams (*concursos*) and searching for private-sector jobs, using the PNADC household survey (IBGE) and the RAIS employer-employee registry (Ministry of Labor).

Everything here runs on public data pulled from [Base dos Dados](https://basedosdados.org) (Google BigQuery). The raw and cleaned data are not in the repository; the scripts recreate them.

## Pipeline

Scripts are numbered in the order they run. Python scripts take `--help` where they have options.

| Step | Script | What it does |
|---|---|---|
| 0 | `0_import.py`, `0_import_sample.py`, `0_download_vd3004_7.py` | Pull PNADC microdata from BigQuery (`basedosdados.br_ibge_pnadc.microdados`): a full year, a 10% household sample, or only respondents with a completed tertiary degree (`VD3004 == 7`) to keep downloads manageable. |
| 0 | `0_import_dicionario.py`, `0_import_educacao.py` | Pull the IBGE variable dictionary (code to label mappings) and the education supplement table. |
| 1 | `1_clean.py`, `1_clean_educacao.py` | Apply dictionary labels, build the person identifier (`id_domicilio` + `V2003`), and save cleaned yearly files. |
| 2 | `2_harmonize.py`, `2a_harmonize_makelighter.py`, `harmonized_data.py` | Merge the cleaned waves (2016–2019, 2022–2025), align column names across years, deflate all wage and income variables to 2025-Q3 prices, and export Parquet / Stata / CSV. `harmonized_data.py` is the shared loader used by every later script (`--sample full` or `--sample higher-ed`). |
| 3 | `3_transitions_sample10pct.py`, `3_paper_analysis.py` | Quarter-to-quarter labor-market transitions (unemployed / public / private) and every descriptive figure and table used in the paper. |
| 4 | `4_estimate_ctmc_rates.py`, `estimate_ctmc_rates.R` | Recover continuous-time Poisson hazard rates from the discrete quarterly transition matrix: closed-form patch, CTMC maximum likelihood, and a matrix-log check. Python port of the R original. |
| 5–7 | `5_descriptives.py`, `6_model_log_q20.py`, `6_analysis.do`, `7_pnadc_fillins_survey.py` / `.do`, `10_fillins_pnadc_fulltime.do` | Descriptive tabulations, wage-quantile models, and the headline statistics used as fill-ins in the survey instrument. |
| 9 | `9_GMM.py` | Structural estimation by GMM over a grid of calibrated (alpha, lambda_0) values, with percentile-bootstrap confidence intervals and LaTeX table output. |
| 11 | `11a_rais_import.py`, `11a_rais_import_parallel.py`, `11_rais_descriptives.py` + `.sbatch` files | Download RAIS linked employer-employee records for all 27 states and 2010–2024 (parallel, resumable, atomic writes), and tabulate public vs. private formal employment. The `.sbatch` files run these on a Slurm cluster (Harvard FASRC). |
| lib | `pnadc_superpc/` | Reusable helpers (config, I/O, cleaning, variable mappings, transition matrices) shared by the scripts above. |

## Requirements

- Python 3.10+ with `pandas`, `numpy`, `scipy`, `pyarrow`, `matplotlib`, `seaborn`, `basedosdados`
- A Google Cloud project with BigQuery enabled (`basedosdados` bills query bytes to it; set it with `--billing-project-id` or in the script header)
- R 4.x with `expm` and `msm` for `estimate_ctmc_rates.R`
- Stata 16+ for the `.do` files

```bash
pip install pandas numpy scipy pyarrow matplotlib seaborn basedosdados
```

Note from `11a_rais_import_parallel.py`: `basedosdados` pins old versions of `click`, `pandas` and `pyarrow` and conflicts with `geopandas`; keep them in separate environments.

## Running

```bash
# 1. download one year of PNADC (edit YEAR at the top of the script)
python 0_import.py
# 2. clean and harmonize
python 1_clean.py
python 2_harmonize.py --sample full
# 3. transitions, hazard rates, GMM
python 4_estimate_ctmc_rates.py --sample full
python 9_GMM.py --alphas 0.03 0.06 --lambda0s 0.50 0.80 --bootstrap 300
```

Paths: the scripts assume the layout `<project>/Code/` (this repo), `<project>/Raw Data/` and `<project>/Cleaned Data/`, resolved relative to the script's location. A few older scripts still carry an absolute path at the top; edit it before running.

## Data sources

- IBGE, *Pesquisa Nacional por Amostra de Domicílios Contínua* (PNADC), via `basedosdados.br_ibge_pnadc`
- Ministério do Trabalho, *Relação Anual de Informações Sociais* (RAIS), via `basedosdados.br_me_rais`
