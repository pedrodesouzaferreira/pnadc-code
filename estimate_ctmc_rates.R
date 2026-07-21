################################################################################
# Recovering continuous-time hazard rates (delta_P, delta_R, lambda_0*alpha, etc.)
# from PNADC's discrete quarterly panel + interval-censored tenure.
#
# States: U = unemployed, P = public, R = private.
# Model generator Q (rows = origin, cols = destination), all entries are HAZARDS:
#
#        U                              P                    R
#  U  -(qUP+qUR)                       lambda0*alpha*pP     lambda0*(1-alpha)*pR
#  P   delta_P                         -(delta_P+piPR)      piPR
#  R   delta_R                         piRP                 -(delta_R+piRP)
#
# The quarterly transition matrix is P(Delta) = expm(Q*Delta).
# Reading raw quarterly FRACTIONS as if they were RATES is the units error.
#
# Keep ONE time unit everywhere (rates below are per QUARTER unless noted) and
# make sure rho uses the same unit.
################################################################################


## ----------------------------------------------------------------------------
## ROUTE A. Two-line closed-form competing-risks patch.
## Turns one-period transition PROBABILITIES out of a state into destination
## hazards. Exact if there are no within-period bounce-backs (good enough for a
## quarter); msm (Route B) relaxes that.
## ----------------------------------------------------------------------------

cr_hazard <- function(p_dest, Delta = 1) {
  # p_dest : named numeric vector of destination PROBABILITIES (exclude staying)
  # Delta  : length of one observation period, in your chosen time unit
  p_exit     <- sum(p_dest)
  total_rate <- -log(1 - p_exit) / Delta
  total_rate * p_dest / p_exit          # destination-specific hazards
}

# EXAMPLE (quarter = 1 time unit). *** CHECK YOUR LABELS FIRST (see flag below) ***
# Suppose corrected fractions are:
cr_hazard(c(P = 0.028, R = 0.439))       # U-exit -> qUP, qUR
cr_hazard(c(U = 0.012, R = 0.084))       # P-exit -> qPU (=delta_P), qPR
cr_hazard(c(U = 0.034, P = 0.018))       # R-exit -> qRU (=delta_R), qRP

# Then acceptance probabilities (with calibrated lambda0, alpha):
# pP = qUP / (lambda0 * alpha);  pR = qUR / (lambda0 * (1 - alpha))


## ----------------------------------------------------------------------------
## ROUTE B. msm  (RECOMMENDED). Exact discretely-observed CTMC likelihood:
##   L_i = prod_k [ expm(Q * Delta_ik) ]_{s_ik, s_i,k+1}
## Handles competing risks, unequal spacing, and within-quarter multiple moves.
## ----------------------------------------------------------------------------

# install.packages("msm")
library(msm)

# Data must be LONG: one row per person-wave, with
#   id    : person identifier
#   time  : NUMERIC time in a consistent unit (e.g. quarter index 0,1,2,3,4)
#   state : 1 = U, 2 = P, 3 = R
# dat <- read.csv("pnadc_long.csv")

# Inspect the raw transition counts msm will use:
# statetable.msm(state, id, data = dat)

# Allowed transitions (0 = forbidden; diagonals filled automatically).
# All six flows allowed here:
Q0 <- rbind(
  c(0,    0.03, 0.50),   # from U
  c(0.01, 0,    0.08),   # from P
  c(0.01, 0.02, 0)       # from R
)
rownames(Q0) <- colnames(Q0) <- c("U", "P", "R")

fit <- msm(state ~ time, subject = id, data = dat,
           qmatrix = Q0, gen.inits = TRUE)   # gen.inits = crude rate starting values

# Estimated generator: read off the hazards you need
Qhat <- qmatrix.msm(fit)     # Qhat["P","U"] = delta_P ; Qhat["R","U"] = delta_R
Qhat                         # Qhat["P","R"] = piPR    ; Qhat["U","P"] = lambda0*alpha*pP, etc.

# Sanity check: implied one-quarter P vs the raw fractions
pmatrix.msm(fit, t = 1)


## ----------------------------------------------------------------------------
## ROUTE C. Matrix-log of the empirical one-quarter matrix (a check on B).
## May return small NEGATIVE off-diagonals (the embedding problem) -> then trust
## msm instead.
## ----------------------------------------------------------------------------

# install.packages("expm")
library(expm)

# Phat: 3x3 empirical one-quarter transition matrix, rows summing to 1,
#       states ordered U, P, R.
# Phat <- prop.table(statetable.msm(state, id, data = dat), 1)

Delta <- 1                      # one quarter, in your time unit
Q_ml  <- logm(Phat) / Delta
Q_ml                            # off-diagonals should be >= 0; rows sum ~ 0


## ----------------------------------------------------------------------------
## ROUTE D. Tenure question -> grouped-exponential MLE of the job-spell exit rate.
## Elapsed tenure of an ongoing spell is Exponential(q_j) in steady state.
## Fit on TOP-WAGE workers -> q_j ~= delta_j (clean separation rate).
## Fit on all -> pooled q_j; the gap to delta_j identifies lambda_1.
## Do NOT pool across wages into a single exponential (mixture bias); fit by wage bin.
##
## PNADC bins (converted to YEARS; note this fit is per-year):
##   "Menos de 1 mes"          -> [0, 1/12)
##   "De 1 mes a menos de 1 ano"-> [1/12, 1)
##   "De 1 ano a menos de 2 anos"-> [1, 2)
##   "2 anos ou mais"          -> [2, Inf)
##   "Nao aplicavel"           -> drop (not employed)
## ----------------------------------------------------------------------------

grouped_exp_nll <- function(logq, counts) {
  q <- exp(logq)                       # per-year exit rate (bins are in years)
  p <- c(
    lt1m  = 1 - exp(-q / 12),
    m1_y1 = exp(-q / 12) - exp(-q),
    y1_y2 = exp(-q)      - exp(-2 * q),
    ge2y  = exp(-2 * q)
  )
  -sum(counts * log(p))
}

# counts must be in this exact order:  c(<1m, 1m-1y, 1y-2y, >=2y)
# Example: high-wage PUBLIC workers only (top wage decile) -> estimates delta_P
counts_P_top <- c(lt1m = 20, m1_y1 = 300, y1_y2 = 500, ge2y = 4000)   # placeholder
opt   <- optimize(grouped_exp_nll, interval = c(-5, 3), counts = counts_P_top)
q_yr  <- exp(opt$minimum)              # per-YEAR hazard
q_qtr <- q_yr / 4                      # convert to per-QUARTER to match msm/Route A
cat(sprintf("delta_P  ~  %.4f /yr  =  %.4f /qtr\n", q_yr, q_qtr))

# Repeat with high-wage PRIVATE workers -> delta_R.
# Repeat pooled (all wages) per sector -> q_j; then lambda_1 falls out of q_j - delta_j.


################################################################################
# FLAGS
# 1) LABEL SWAP: with alpha in {0.03,0.06} the model caps m_UP <= 0.03, so a
#    43.9% U->public flow is almost certainly U->PRIVATE. Fix labels before anything.
# 2) lambda0: corrected U-exit prob ~0.467/qtr implies a RATE -log(1-0.467)~0.63/qtr,
#    so lambda0 ~ 0.63 (not 0.5). Recalibrate.
# 3) TIME UNITS: pick quarters OR years and use it for delta's, lambda's, AND rho.
################################################################################
