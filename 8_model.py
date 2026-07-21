from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


BIN_WIDTH = 500
PNADC_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = PNADC_DIR / "Cleaned Data"
FIGURES_DIR = PNADC_DIR / "Output" / "Figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Loading data
columns_to_keep = [
    "id_pessoa",
    "ano",
    "trimestre",
    "renda_habitual_principal",
    "empregado_setor_pub",
    "empregado_setor_priv",
]
df = pd.read_csv(CLEANED_DIR / "PNADC_limpo_2024.csv", usecols=columns_to_keep)
df_2023 = pd.read_csv(CLEANED_DIR / "PNADC_limpo_2023.csv", usecols=columns_to_keep)
df = pd.concat([df, df_2023], ignore_index=True)

# Keeping needed columns
columns_to_keep = [
    "id_pessoa",
    "ano",
    "trimestre",
    "renda_habitual_principal",
    "empregado_setor_pub",
    "empregado_setor_priv",
]
df = df[columns_to_keep]

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

# Step 1: estimate mu_P and sigma_P from public-sector wages
public_wages = df.loc[df["sector"] == "public", "wage"]

# Optional but recommended: estimate on log wages
public_logwages = np.log(public_wages[public_wages > 0])
mu_P, sig_P = public_logwages.mean(), public_logwages.std()

# Step 2: compute observed truncated mean in log space
switchers = df[df["transition"] == "private_to_public"].copy()
switchers = switchers[
    (switchers["wage_origin"] > 0) & (switchers["wage_dest"] > 0)
].copy()
switchers["log_wage_dest"] = np.log(switchers["wage_dest"])

if switchers.empty:
    raise ValueError("No private-to-public switchers with positive origin and destination wages.")

bin_min = np.floor(switchers["wage_origin"].min() / BIN_WIDTH) * BIN_WIDTH
bin_max = np.ceil(switchers["wage_origin"].max() / BIN_WIDTH) * BIN_WIDTH
bin_edges = np.arange(bin_min, bin_max + BIN_WIDTH, BIN_WIDTH)

switchers["w_bin"] = pd.cut(
    switchers["wage_origin"],
    bins=bin_edges,
    right=False,
    include_lowest=True,
)

bin_stats = (
    switchers.groupby("w_bin", observed=True)
    .agg(
        w_origin_mean=("wage_origin", "mean"),
        truncated_mean=("log_wage_dest", "mean"),
        n=("wage_dest", "count"),
    )
    .reset_index()
)

bin_stats["bin_left"] = bin_stats["w_bin"].apply(lambda interval: interval.left)
bin_stats["bin_right"] = bin_stats["w_bin"].apply(lambda interval: interval.right)
bin_stats["w_bin_mid"] = (bin_stats["bin_left"] + bin_stats["bin_right"]) / 2
bin_stats = bin_stats.sort_values("bin_left").reset_index(drop=True)


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
    raise ValueError("Could not recover any valid threshold from the wage bins.")

bottom_bin = valid_bins.iloc[0]
a_over_beta = bottom_bin["w_origin_mean"] - bottom_bin["tilde_w_P"]
print(f"Estimated a/beta = {a_over_beta:.1f} BRL/month")

print("\nFirst bins used in the estimation:")
print(
    bin_stats[
        ["w_bin", "bin_left", "bin_right", "w_origin_mean", "truncated_mean_level", "tilde_w_P", "n"]
    ].head(10)
)

# Step 5: plot threshold vs origin wage
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
ax.set_title(f"Switching threshold vs. observed destination wages\nBins of BRL {BIN_WIDTH:,.0f}")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "threshold_plot.pdf")
