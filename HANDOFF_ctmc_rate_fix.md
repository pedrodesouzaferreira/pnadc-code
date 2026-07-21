# Handoff: fix the continuous-time-rate estimation in "The Value of Public Employment"

## PRIME DIRECTIVE (read first)

The paper's structural estimation currently plugs **discrete-time transition
probabilities** (quarter-to-quarter fractions) into a **continuous-time model**
whose parameters are **Poisson hazard rates**. This is a units error. Your job is
to fix the rate estimation and, **as the first and only deliverable of this pass**,
produce the numeric output of two estimators — **Route A** (closed-form patch) and
**Route B** (`msm` CTMC MLE) — and a comparison table against the raw fractions.

**Do NOT edit the `.tex` yet. Do NOT touch the downstream reservation-wage /
amenity code yet.** Compute Routes A and B, print the comparison, and stop for
review. Pedro wants to see A and B before anything is written up.

A ready-to-use, commented R script with all four estimators lives at
`estimate_ctmc_rates.R` — reuse its functions rather than rewriting them.

---

## 1. Project context (so the changes make sense)

Pedro Ferreira, *The Value of Public Employment*. A partial-equilibrium,
continuous-time, on-the-job-search model (Burdett–Mortensen style) of public vs.
private employment in Brazil. Three labor-market states:

- `U` = unemployed
- `P` = employed, public sector
- `R` = employed, private sector

Key structural parameters: separation rates `delta_P`, `delta_R`; offer-arrival
rates `lambda0` (unemployed) and `lambda1` (employed); `alpha` = probability an
offer is public; discount rate `rho`; lognormal offer distributions
`F_P, F_R` with reservation wages `R_P, R_R`; amenity value `a/beta`.

Two identification targets:
1. `R_R - R_P = a/beta` (dollar value of public amenities).
2. Slope of the cross-sector switching threshold,
   `wtilde_P'(w) = 1 - (delta_R - delta_P)/q_R(w)` (job-stability channel).

**Data (PNADC).** Rotating household panel, 2023–2024, ~1.23M individuals,
each followed for **up to 5 quarters**. You observe the state each quarter and
accepted wages. You also have an interval-censored **tenure** question
(categories: "Não aplicável", "Menos de 1 mês", "De 1 mês a menos de 1 ano",
"De 1 ano a menos de 2 anos", "2 anos ou mais"). You do **not** observe exact
spell durations.

Data-mapping conventions already in the paper: self-employed/employers -> `R`;
unpaid family workers -> `U`; out-of-labor-force transitions disregarded.

---

## 2. The object being estimated: the CTMC generator

The correct object is the intensity matrix (generator) `Q` of the continuous-time
Markov chain on `{U, P, R}`. Off-diagonal `q_jk` = instantaneous hazard of `j->k`;
each diagonal = minus the row sum. In Pedro's notation, ordering `(U, P, R)`:

```
         U                          P                     R
 U   -(qUP+qUR)                 lambda0*alpha*pP    lambda0*(1-alpha)*pR
 P    delta_P                   -(delta_P+piPR)     piPR
 R    delta_R                   piRP                -(delta_R+piRP)
```

where `p_s = 1 - F_s(R_s)` are acceptance probabilities, and the cross-sector
poaching intensities are wage-averaged pieces of the model's `q_j(w)`:

```
piPR = lambda1*(1-alpha) * E_{w~G_P}[ 1 - F_R(wtilde_R(w)) ]
piRP = lambda1*alpha     * E_{w~G_R}[ 1 - F_P(wtilde_P(w)) ]
```

Crucial facts:

- **Within-sector upgrades (public->better public) are `P->P` self-transitions**
  and do NOT appear in this state generator. So `qPU = delta_P` and `qRU = delta_R`
  are clean reads of the separation rates, uncontaminated by on-the-job search.
- The identity `qUP = lambda0*alpha*pP` is exactly the paper's `m_UP` equation —
  but it refers to the **generator entry `qUP`**, NOT the raw quarterly fraction
  `Phat_UP`. That single substitution is the fix.
- Discrete <-> continuous link: the one-quarter transition matrix is
  `P(Delta) = expm(Q * Delta)`. The bug is treating `Phat_jk(Delta)` as if it
  equaled `q_jk * Delta`. Fine for rare events; badly wrong for U-exit, where the
  row sums to ~0.47 and `exp`/`log` are strongly nonlinear.

---

## 3. Severity (what actually changes)

Illustrative Route-A recompute from the paper's reported aggregate fractions,
with the label swap (see Flag 1) corrected, Delta = 1 quarter:

| Flow            | Raw fraction | Route A hazard (/qtr) | correction |
|-----------------|-------------:|----------------------:|-----------:|
| U -> P          |        0.028 |                0.0377 |      +35%  |
| U -> R          |        0.439 |                0.5915 |      +35%  |
| P -> U (=delta_P)|       0.012 |                0.0126 |      +5.1% |
| P -> R          |        0.084 |                0.0883 |      +5.1% |
| R -> U (=delta_R)|       0.034 |                0.0349 |      +2.7% |
| R -> P          |        0.018 |                0.0185 |      +2.7% |

Takeaway: separation rates barely move (delta's were ~fine), but the
U->employment rates inflate ~35%. Those rates feed the acceptance probabilities
`p_s`, hence `R_P, R_R`, hence `a/beta`. This is why the fix matters.

**These illustrative numbers are from reported aggregates only.** Recompute
everything from the microdata.

---

## 4. TASK — implement Routes A and B, then STOP

### Step 0. Locate the current code

Find where the analysis currently:
- computes `m_{P->U}` and `m_{R->U}` as fractions and sets `delta_P`, `delta_R`
  equal to them (paper Step 1 / eq. `m_{P->U}=delta_P`);
- computes `m_UP`, `m_UR` as fractions of unemployed who transition, then backs
  out `p_P = m_UP/(lambda0*alpha)`, `p_R = m_UR/(lambda0*(1-alpha))`
  (paper eq. `acceptance_probs`);
- feeds `p_P, p_R` into the truncated-lognormal inversion for `R_P, R_R, a/beta`.

Report the file/function/line locations back before changing anything downstream.

### Step 1. Build the empirical one-quarter transition matrix `Phat`

- Restrict to consecutive-wave pairs (wave t, wave t+1) for the same person.
- States `U, P, R` using the paper's mapping (self-employed/employers -> R;
  unpaid family -> U; drop OLF transitions).
- IMPORTANT: use the **full sample** for transition counts. The minimum-wage
  filter that drops sub-minimum wages is applied ONLY to wage moments, NOT to
  transitions (preserve this existing logic).
- Produce the 3x3 `Phat` (rows sum to 1) and print it, plus the raw counts.

### Step 2. Route A — closed-form competing-risks patch

Use `cr_hazard()` from `estimate_ctmc_rates.R`. For each origin state, convert the
destination PROBABILITIES into hazards:

```
q_dest = -log(1 - p_exit)/Delta * (p_dest / p_exit),   p_exit = sum(p_dest)
```

Output `qUP, qUR, qPU(=delta_P), qPR, qRU(=delta_R), qRP` per quarter.
Also report implied `lambda0 = qUP + qUR` (under acceptance ~ 1) and the corrected
acceptance probabilities `p_s = q_{Us}/(lambda0*alpha_s)` using the calibrated
`lambda0, alpha`.

### Step 3. Route B — `msm` CTMC MLE

Use the `msm` block in `estimate_ctmc_rates.R`. Requirements:
- Long format: one row per person-wave with `id`, numeric `time` (quarter index,
  consistent unit), `state` in `{1=U, 2=P, 3=R}`.
- Allow all six transitions; let diagonals be filled automatically; `gen.inits=TRUE`.
- Print `qmatrix.msm(fit)` (the estimated `Q`) and `pmatrix.msm(fit, t=1)`
  (implied one-quarter `P`, to compare against `Phat`).

### Step 4. Comparison table (the deliverable)

Produce one table with, for each of the six flows:
`raw fraction | Route A hazard | Route B hazard | (Route B implied P vs Phat)`.

Then STOP and hand back for review. No tex, no downstream edits yet.

Expected pattern: A and B should be close. Large A-vs-B divergence on the churny
flows (e.g. U<->R) signals within-quarter multiple transitions (which B handles
via `expm(Q*Delta)` and A approximates) or nonstationarity across waves.

---

## 5. Flags to verify while doing this

**Flag 1 — LABEL SWAP.** With `alpha in {0.03, 0.06}` and `lambda0 = 0.5`, the model
caps `m_UP = lambda0*alpha*pP <= 0.03`. The paper reports U->public = 43.9%, which
is impossible for the public flow — it is almost certainly U->PRIVATE, with 2.8%
being U->public. The separation sentence in the draft is also internally
inconsistent ("public lower ... 3.4% vs 1.2%"). **Verify the true labels in the
microdata before trusting any output.** Under the economically sensible reading
public is the more stable / more rationed sector.

**Flag 2 — lambda0 recalibration.** Corrected quarterly U-exit prob ~0.467 implies a
RATE `-log(1-0.467) ~ 0.63/qtr`, so `lambda0 ~ 0.63`, not 0.5. Recompute the
calibration once the rates are in hand. (For reference, Postel-Vinay–Robin 2002
estimate annualized `lambda0 ~ 1.5–2.1`.)

**Flag 3 — TIME UNITS.** Pick ONE unit (quarters recommended, since data are
quarterly) and use it for `delta`'s, `lambda`'s, AND `rho`. A quarterly `delta`
with an annual `rho` is inconsistent. If any parameter is currently annual,
convert.

---

## 6. Route D — tenure cross-check (optional this pass; describe, don't block on it)

The tenure question is the only duration info available and gives an INDEPENDENT
read on separation rates. Under a constant exit hazard, the elapsed tenure of an
ongoing spell is Exponential(`q_j`) in steady state. Fit a grouped/interval-censored
exponential to the tenure bins (function `grouped_exp_nll()` in the R script),
bins in years:

```
"Menos de 1 mes"            -> [0, 1/12)
"De 1 mes a menos de 1 ano" -> [1/12, 1)
"De 1 ano a menos de 2 anos"-> [1, 2)
"2 anos ou mais"            -> [2, Inf)
"Nao aplicavel"             -> drop
```

- Fit on **top-wage** workers in each sector -> `q_j ~ delta_j` (clean separation
  rate, because at the top of the wage distribution both within-sector upgrades and
  cross-sector poaching vanish). Cross-check against Route B's `qPU, qRU`.
- The gap between pooled `q_j` (all wages) and top-wage `delta_j` identifies
  `lambda1`, which the paper currently backs out only informally.
- Do NOT pool all wages into a single exponential fit (mixture-of-exponentials
  bias / spurious negative duration dependence); fit within wage bins.
- Convert per-year to per-quarter (`/4`) to match Routes A/B.

If time allows, report the top-wage tenure `delta_P, delta_R` next to Routes A/B.

---

## 7. Downstream edits (LATER — not this pass, listed so you know where it lands)

Once Pedro approves the A/B output, the only pipeline changes are at the INPUTS:
- Set `delta_P = qPU`, `delta_R = qRU` (from Route B, or A) instead of raw fractions.
- Set `p_s = q_{Us}/(lambda0*alpha_s)` using the rate `q_{Us}` instead of the
  fraction `m_{Us}`.
- Recalibrate `lambda0` per Flag 2; confirm units per Flag 3.
The truncated-lognormal inversion for `R_P, R_R, a/beta` is unchanged — only its
inputs change. The `.tex` (Steps 1 and the identification/empirics sections) is
edited only after the numbers are reviewed.

---

## 8. Modeling caveat to keep visible

`msm` assumes piecewise-constant hazards over each quarter and a locally stationary
chain — the same steady-state assumption the reservation-wage inversion already
relies on, so it is coherent with the rest of the paper. But if 2023->2024 has real
business-cycle movement in `lambda0`, allow the generator to depend on wave
(`msm` supports covariates on `Q`). Note this in the report; don't act on it yet.
