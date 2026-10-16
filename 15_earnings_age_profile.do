*==============================================================================
* 15_earnings_age_profile.do
*
* Mean earnings by age (22-60), public vs private, Brazil vs Distrito Federal,
* PNADC college-educated full-time formal workers -- the Stata version of
* 12_earnings_age_profile.py -- with the UnB pilot respondents' own expected
* wage right after graduation (E3B.5 / E3B.6) overlaid.
*
* INPUT   Cleaned Data/PNADC_harmonized_VD3004_7_2016_2025_light.csv
*         ../PilotoUnB/Clean Data/piloto_clean.dta
*         ../PilotoUnB/Raw Data/matriculas_planilha_mestra.csv
* OUTPUT  Output/Tables/pnadc_earnings_age_profile_stata_<tag>_<single|hwK>_p<year>.csv
*         Output/Figures/pnadc_earnings_age_profile_stata_<tag>_<single|hwK>_p<year>.{pdf,png}
*         Output/Figures/pnadc_earnings_age_profile_stata_<tag>_<single|hwK>_p<year>_nobeliefs.{pdf,png}
*           (same graph without the students' expected wage); both 16:9
*         Code/15_earnings_age_profile.log
*
* PNADC SAMPLE (same cut as 12_earnings_age_profile.py; public sector exactly as
* in 7_pnadc_fillins_survey.do):
*   - years in `years' (default 2016-2019 and 2022-2024), pooled;
*   - first interview of each person in those years
*     (bys id_pessoa (ano trimestre): _n == 1, as first_obs in the .do);
*   - completed tertiary education (the input file only has VD3004 == 7);
*   - usual hours in main job >= 35; habitual main-job earnings > 0;
*   - public  = empregado_setor_pub_refined (block copied from the .do);
*   - private = empregado_setor_priv == 1 & formal == 1 (private employee WITH
*               carteira; set priv_all = 1 to drop the carteira condition and
*               match the .do's $pri);
*   - weights [aweight = peso]; DF = id_uf == 53.
* EARNINGS  renda_habitual_principal_real_2025q3 (2025-Q3 reais), rescaled to
*   average-`priceyear' reais by one national factor k = weighted mean, over all
*   `priceyear' interviews with positive earnings, of nominal / real (k ~ 0.948
*   for 2024). A single scalar: shapes and gaps do not change.
* ESTIMATES For each age a: weighted mean over people aged a-hw..a+hw (hw = 0:
*   single years). Band: mean +/- 1.96 * sd_w / sqrt(n_eff), n_eff =
*   (sum w)^2 / sum w^2; ignores the survey design, so it is too narrow.
* BELIEFS   Analysis sample of 05_analysis.do (Piloto 2, reached the prize
*   question, matched to the enrolment roster), pre-treatment. Items:
*   e3b_own_wage_publico (E3B.6) and e3b_own_wage_privado (E3B.5): "logo apos
*   concluir sua graduacao ... trabalhando em tempo integral como concursado /
*   no setor privado ... qual seria, mais provavelmente, seu salario mensal
*   bruto?". Mean across respondents, winsorised at p1/p99 as in 05_analysis.do
*   (_belstat); drawn as a dashed horizontal line across ages `agemin'-`agemax'
*   in each sector's colour (its 90% CI is printed in the log, not plotted).
*   In Aug-Sep 2026 reais; no price adjustment. The same lines appear in both
*   panels.
* Expected values (Python 12_ with the same options, for checking the log):
*   k = 0.9477; mean at age 23: Brazil public 3,234 / private 2,995;
*   DF public 7,207 / private 2,832; age 30: Brazil public 5,710 / private 4,531.
*   Pilot N = 249; own expected wage, winsorised mean [90% CI]: public 9,377
*   [8,806, 9,949] (N = 247); private 5,203 [4,834, 5,573] (N = 248).
*
* Re-runnable; overwrites its outputs. Does not modify any existing file.
*==============================================================================
clear all
set more off

global pnadc  "/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC"
global piloto "/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PilotoUnB"
cd "$pnadc"

capture log close
log using "Code/15_earnings_age_profile.log", replace text

*------------------------------------------------------------------------------
* Options
*------------------------------------------------------------------------------
local years      "2016 2017 2018 2019 2022 2023 2024"
local tag        "2016-2019_2022-2024"
local priceyear  2024
local hw         0          // smoothing half-window in years: 0 = single years, 2 = 5-year window
local priv_all   0          // 1 = private employees with or without carteira (as $pri in the .do)
local agemin     22
local agemax     60

local ftag = cond(`hw' == 0, "single", "hw`hw'")
local stem "pnadc_earnings_age_profile_stata_`tag'_`ftag'_p`priceyear'"

*------------------------------------------------------------------------------
* 0. Pilot: own expected wage right after graduation (winsorised mean, 90% CI)
*------------------------------------------------------------------------------
tempfile roster
import delimited "$piloto/Raw Data/matriculas_planilha_mestra.csv", clear
rename matricula c0_matricula
tostring c0_matricula, replace force format(%15.0f)
replace c0_matricula = trim(c0_matricula)
isid c0_matricula
save `roster'

use "$piloto/Clean Data/piloto_clean.dta", clear
keep if !missing(f8_sorteio_escolha) & sample == 3
merge 1:1 c0_matricula using `roster', keep(1 3)
keep if _merge == 3                      // enrolment filter, as in 05_analysis.do
drop _merge
local nresp = _N
di as result "Pilot analysis sample: N = `nresp'"

foreach s in publico privado {
    quietly summarize e3b_own_wage_`s', detail
    local p1  = r(p1)
    local p99 = r(p99)
    gen double ow_`s' = min(max(e3b_own_wage_`s', `p1'), `p99') if !missing(e3b_own_wage_`s')
    quietly mean ow_`s', level(90)
    matrix T = r(table)
    local own_`s'    = T[1,1]
    local own_`s'_lo = T[5,1]
    local own_`s'_hi = T[6,1]
    local own_`s'_n  = e(N)
    di as result "Own expected wage, `s': mean (winsorised p1/p99) = " %9.0fc `own_`s'' ///
        "  90% CI [" %9.0fc `own_`s'_lo' ", " %9.0fc `own_`s'_hi' "]  N = `own_`s'_n'"
}

*------------------------------------------------------------------------------
* 1. PNADC sample
*------------------------------------------------------------------------------
import delimited "Cleaned Data/PNADC_harmonized_VD3004_7_2016_2025_light.csv", clear
rename renda_habitual_principal_real_20 wage     // = renda_habitual_principal_real_2025q3

* price factor: 2025-Q3 reais -> average-`priceyear' reais
gen double ratio = renda_habitual_principal / wage ///
    if ano == `priceyear' & renda_habitual_principal > 0 & !missing(renda_habitual_principal) & !missing(wage)
summarize ratio [aweight = peso], meanonly
local k = r(mean)
di as result "Price factor 2025q3 -> `priceyear': " %6.4f `k'
drop ratio

gen byte inyears = 0
foreach y of local years {
    replace inyears = 1 if ano == `y'
}
keep if inyears
drop inyears
replace wage = wage * `k'

* first interview of each person in the pooled years (first_obs in the .do)
bysort id_pessoa (ano trimestre): gen byte first_obs = _n == 1
keep if first_obs

* public sector: copied from 7_pnadc_fillins_survey.do
gen byte public_universe = !missing(empregado_setor_pub)
gen byte public_rule = ///
    servidor_publico_estatutario == 1 | ///
    (inlist(v4010, 5412, 511, 512, 411, 412) & posicao_trab_principal == 2) | ///
    (posicao_trab_principal == 4 & carteira_assinada == 1)
gen byte empregado_setor_pub_refined = 0 if public_universe
replace empregado_setor_pub_refined = 1 if public_universe & public_rule
replace empregado_setor_pub_refined = 0 if public_universe & v4025 == 1
drop public_universe public_rule

gen byte private = empregado_setor_priv == 1
if `priv_all' == 0 replace private = 0 if formal != 1

gen byte sector = .
replace sector = 2 if private == 1
replace sector = 1 if empregado_setor_pub_refined == 1
keep if !missing(sector)
keep if horas_habituais_principal >= 35 & !missing(horas_habituais_principal)
keep if wage > 0 & !missing(wage)
keep if inrange(idade, `agemin' - `hw', `agemax' + `hw')
keep ano trimestre id_uf idade peso wage sector
gen double w2 = peso^2

count if inrange(idade, `agemin', `agemax')
di as result "PNADC sample, ages `agemin'-`agemax': " r(N)
tab sector if inrange(idade, `agemin', `agemax')
tab sector if inrange(idade, `agemin', `agemax') & id_uf == 53

*------------------------------------------------------------------------------
* 2. Mean by age
*------------------------------------------------------------------------------
tempname h
tempfile prof
postfile `h' byte(region sector) int age double(mean se) long n using `prof', replace
forvalues r = 1/2 {
    local rcond = cond(`r' == 1, "1", "id_uf == 53")
    forvalues s = 1/2 {
        forvalues a = `agemin'/`agemax' {
            local sel "`rcond' & sector == `s' & abs(idade - `a') <= `hw'"
            quietly summarize wage [aweight = peso] if `sel'
            local N = r(N)
            if `N' == 0 {
                post `h' (`r') (`s') (`a') (.) (.) (0)
                continue
            }
            local m = r(mean)
            local v = cond(`N' > 1, r(Var) * (`N' - 1) / `N', 0)   // sum w (x-m)^2 / sum w
            quietly summarize peso if `sel'
            local sw = r(sum)
            quietly summarize w2 if `sel'
            local sw2 = r(sum)
            local se = sqrt(`v' / (`sw'^2 / `sw2'))
            post `h' (`r') (`s') (`a') (`m') (`se') (`N')
        }
    }
}
postclose `h'

use `prof', clear
gen double lo = mean - 1.96 * se
gen double hi = mean + 1.96 * se
label define region 1 "Brazil" 2 "Distrito Federal"
label values region region
label define sector 1 "Public" 2 "Private"
label values sector sector
export delimited using "Output/Tables/`stem'.csv", replace

di as result "Mean earnings at selected ages (R$ of average `priceyear'):"
list region sector age mean n if inlist(age, 23, 30, 40, 50, 60), sepby(region sector) noobs

*------------------------------------------------------------------------------
* 3. Add the belief markers and plot
*------------------------------------------------------------------------------
* 8 extra rows: region (2) x sector (2) x {agemin, agemax}, so the own expected
* wage is drawn as a horizontal dashed line across the whole age range
local nprof = _N
set obs `=`nprof' + 8'
gen int kk = _n - `nprof' if _n > `nprof'
replace region = cond(kk <= 4, 1, 2) if _n > `nprof'
replace sector = cond(mod(kk - 1, 4) < 2, 1, 2) if _n > `nprof'
gen double own  = cond(sector == 1, `own_publico', `own_privado') if _n > `nprof'
gen int own_age = cond(mod(kk, 2) == 1, `agemin', `agemax') if _n > `nprof'
drop kk

local cpub "42 120 214"
local cpri "235 104 52"
local ylab = cond(`hw' == 0, "single years of age", "centred `=2*`hw'+1'-year age window")
local years_lbl = subinstr("`tag'", "_", ", ", .)
local privtxt = cond(`priv_all' == 1, "private employees with or without carteira", ///
                                       "private employees with carteira (VD4009 = 1)")

* Two versions: with the students' own expected wage (dashed lines) and
* without it (PNADC profile only, suffix _nobeliefs). Both 16:9.
local pnote "PNADC wages inflated to `priceyear' reais (IBGE deflator)."
foreach ver in beliefs nobeliefs {
    if "`ver'" == "beliefs" {
        local ownplots `"(line own own_age if sector == 1, lcolor("`cpub'") lpattern(dash) lwidth(medthick)) (line own own_age if sector == 2, lcolor("`cpri'") lpattern(dash) lwidth(medthick))"'
        local ttl   "Mean earnings by age vs. students' expected wage after graduation"
        local nt    `""`pnote'" "Dashed lines: UnB pilot (N = `nresp'), own most likely gross wage right after graduation" "in each sector (E3B.5/E3B.6), mean winsorised at p1/p99; Aug-Sep 2026 reais.""'
        local lord  `"3 "Public, PNADC mean" 4 "Private, PNADC mean" 5 "Own exp. wage, public" 6 "Own exp. wage, private""'
        local lrows 2
        local vsuf  ""
    }
    else {
        local ownplots ""
        local ttl   "Mean earnings by age"
        local nt    `""`pnote'""'
        local lord  `"3 "Public, PNADC mean" 4 "Private, PNADC mean""'
        local lrows 1
        local vsuf  "_nobeliefs"
    }

    twoway ///
        (rarea lo hi age if sector == 1 & !missing(mean), color("`cpub'%20") lwidth(none)) ///
        (rarea lo hi age if sector == 2 & !missing(mean), color("`cpri'%20") lwidth(none)) ///
        (connected mean age if sector == 1, lcolor("`cpub'") mcolor("`cpub'") msymbol(o) msize(vsmall) lwidth(medthick)) ///
        (connected mean age if sector == 2, lcolor("`cpri'") mcolor("`cpri'") msymbol(o) msize(vsmall) lwidth(medthick)) ///
        `ownplots' ///
        , by(region, ///
             title("`ttl'", size(medsmall)) ///
             subtitle("PNADC `years_lbl', college-educated full-time formal workers", size(small)) ///
             note(`nt', size(vsmall)) ///
             graphregion(color(white)) legend(position(6))) ///
        legend(order(`lord') rows(`lrows') size(small) region(lcolor(white))) ///
        xlabel(25(5)60) xtitle("Age") ///
        ylabel(, format(%9.0fc) angle(0) grid glcolor(gs14)) ytitle("Mean monthly earnings (R$)") ///
        plotregion(color(white)) scheme(s1color) xsize(16) ysize(9)

    graph export "Output/Figures/`stem'`vsuf'.pdf", replace
    graph export "Output/Figures/`stem'`vsuf'.png", replace width(2200)
    di as result "-> Output/Figures/`stem'`vsuf'.pdf (+png)"
}

log close
