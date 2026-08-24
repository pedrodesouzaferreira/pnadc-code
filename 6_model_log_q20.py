import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


N_QUANTILES = 20


# Setting cd
os.chdir("C:/Users/ped205/Dropbox/MY PROJECTS/CONCURSOS PUBLICOS/Data/PNADC")

# Only load the columns we actually use
columns_to_keep = [
    "id_pessoa",
    "ano",
    "trimestre",
    "renda_habitual_principal",
    "empregado_setor_pub",
    "empregado_setor_priv",
]

dtypes = {
    "id_pessoa": "int64",
    "ano": "int16",
    "trimestre": "int8",
    "renda_habitual_principal": "float32",
    "empregado_setor_pub": "int8",
    "empregado_setor_priv": "int8",
}

# Loading data
df = pd.read_csv("Cleaned Data/PNADC_limpo_2024.csv", usecols=columns_to_keep, dtype=dtypes)
df_2023 = pd.read_csv("Cleaned Data/PNADC_limpo_2023.csv", usecols=columns_to_keep, dtype=dtypes)
df = pd.concat([df, df_2023], ignore_index=True)

# Generating variables that match the code
df["wage"] = df["renda_habitual_principal"]
df["sector"] = np.where(
    df["empregado_setor_pub"] == 1,
    "public",
    np.where(df["empregado_setor_priv"] == 1, "private", "unemployed"),
)

# Generating a variable for time
df["time"] = (df["ano"] - 2023) * 4 + df["trimestre"]

# Setting panel structure
df = df.sort_values(["id_pessoa", "time"]).reset_index(drop=True)

# Defining wage_origin as the wage in the previous ano trimestre for each worker
df["wage_origin"] = df.groupby("id_pessoa")["wage"].shift(1)

# Defining wage_dest as the wage in the current ano trimestre for each worker
df["wage_dest"] = df["wage"]

# Defining previous sector
df["sector_origin"] = df.groupby("id_pessoa")["sector"].shift(1)

# Defining transition from private to public
df["transition"] = np.where(
    (df["sector_origin"] == "private") & (df["sector"] == "public"),
    "private_to_public",
    "other",
)

# Step 1: estimate mu_P and sigma_P from public-sector wages in log space
public_wages = df.loc[(df["sector"] == "public") & (df["wage"] > 0), "wage"]
public_logwages = np.log(public_wages)
mu_P, sig_P = public_logwages.mean(), public_logwages.std()

# Step 2: compute observed truncated mean in log space
switchers = df[df["transition"] == "private_to_public"].copy()
switchers = switchers[
    (switchers["wage_origin"] > 0) & (switchers["wage_dest"] > 0)
].copy()

if switchers.empty:
    raise ValueError("No private-to-public switchers with positive origin and destination wages.")

switchers["log_wage_origin"] = np.log(switchers["wage_origin"])
switchers["log_wage_dest"] = np.log(switchers["wage_dest"])

# Bin by log origin wages using quantiles
switchers["log_w_bin"] = pd.qcut(
    switchers["log_wage_origin"],
    q=N_QUANTILES,
    duplicates="drop",
)

bin_stats = (
    switchers.groupby("log_w_bin", observed=True)
    .agg(
        log_w_origin_mean=("log_wage_origin", "mean"),
        w_origin_mean=("wage_origin", "mean"),
        truncated_mean=("log_wage_dest", "mean"),
        n=("wage_dest", "count"),
    )
    .reset_index()
)

bin_stats["log_bin_left"] = bin_stats["log_w_bin"].astype(object).map(lambda interval: float(interval.left))
bin_stats["log_bin_right"] = bin_stats["log_w_bin"].astype(object).map(lambda interval: float(interval.right))
bin_stats["log_bin_mid"] = (bin_stats["log_bin_left"] + bin_stats["log_bin_right"]) / 2

# Convert the log-bin endpoints back to BRL for easier interpretation
bin_stats["w_bin_left"] = np.exp(bin_stats["log_bin_left"])
bin_stats["w_bin_right"] = np.exp(bin_stats["log_bin_right"])
bin_stats["w_bin_mid"] = np.exp(bin_stats["log_bin_mid"])
bin_stats = bin_stats.sort_values("log_bin_left").reset_index(drop=True)


# Step 3: invert the truncated normal for each bin
def truncated_normal_mean(t, mu, sigma):
    """E[X | X >= t] where X ~ N(mu, sigma^2)."""
    alpha = (t - mu) / sigma
    log_mills = norm.logpdf(alpha) - norm.logsf(alpha)
    return mu + sigma * np.exp(log_mills)


def invert_truncated_mean(observed_mean, mu, sigma):
    """Solve for t (log scale) such that E[X | X >= t] = observed_mean."""
    if observed_mean < mu:
        return np.nan
    lower = mu - 10 * sigma
    upper = mu + 10 * sigma
    f = lambda t: truncated_normal_mean(t, mu, sigma) - observed_mean
    try:
        return brentq(f, lower, upper, xtol=1e-8)
    except ValueError:
        return np.nan


bin_stats["log_tilde_w_P"] = bin_stats["truncated_mean"].apply(
    lambda obs: invert_truncated_mean(obs, mu_P, sig_P)
)
bin_stats["tilde_w_P"] = np.exp(bin_stats["log_tilde_w_P"])
bin_stats["truncated_mean_level"] = np.exp(bin_stats["truncated_mean"])

# Step 4: recover a/beta from the threshold
valid_bins = bin_stats.dropna(subset=["tilde_w_P"]).copy()
if valid_bins.empty:
    raise ValueError("Could not recover any valid threshold from the log-wage quantile bins.")

bottom_bin = valid_bins.iloc[0]
a_over_beta = bottom_bin["w_origin_mean"] - bottom_bin["tilde_w_P"]
print(f"Estimated a/beta = {a_over_beta:.1f} BRL/month")

print("\nFirst bins used in the estimation:")
print(
    bin_stats[
        [
            "log_w_bin",
            "w_bin_left",
            "w_bin_right",
            "w_origin_mean",
            "truncated_mean_level",
            "tilde_w_P",
            "n",
        ]
    ].head(10)
)

# Step 5a: plot in BRL space
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(
    bin_stats["w_bin_mid"],
    bin_stats["truncated_mean_level"],
    "o--",
    label="Observed mean destination wage",
)
ax.plot(
    bin_stats["w_bin_mid"],
    bin_stats["tilde_w_P"],
    "s-",
    label="Estimated threshold $\\tilde{w}_P(w)$",
)
ax.plot(
    bin_stats["w_bin_mid"],
    bin_stats["w_bin_mid"],
    "k:",
    label="45 degree line (no pay cut)",
)
ax.set_xlabel("Origin wage bin midpoint (private sector, BRL)")
ax.set_ylabel("Destination wage (public sector, BRL)")
ax.legend()
ax.set_title(f"Threshold vs. destination wage\nLog-origin-wage bins, {N_QUANTILES} quantiles")
plt.tight_layout()
plt.savefig("Output/Figures/threshold_plot_log_q20_levels.pdf")

# Step 5b: plot in log space
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(
    bin_stats["log_bin_mid"],
    bin_stats["truncated_mean"],
    "o--",
    label="Observed mean log destination wage",
)
ax.plot(
    bin_stats["log_bin_mid"],
    bin_stats["log_tilde_w_P"],
    "s-",
    label="Estimated log threshold $\\log \\tilde{w}_P(w)$",
)
ax.plot(
    bin_stats["log_bin_mid"],
    bin_stats["log_bin_mid"],
    "k:",
    label="45 degree line (no pay cut)",
)
ax.set_xlabel("Origin log wage bin midpoint (private sector)")
ax.set_ylabel("Destination log wage (public sector)")
ax.legend()
ax.set_title(f"Threshold vs. destination log wage\nLog-origin-wage bins, {N_QUANTILES} quantiles")
plt.tight_layout()
plt.savefig("Output/Figures/threshold_plot_log_q20_logs.pdf")
plt.show()
