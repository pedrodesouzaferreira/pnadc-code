***********************
* Analysis of PNADC data
***********************

cd "C:\Users\ped205\Dropbox\MY PROJECTS\CONCURSOS PUBLICOS\Data\PNADC"


************************************
* 0. Data loading and exploration 
************************************

*import delimited "Cleaned Data/PNADC_limpo_2024.csv", clear 
*save "Cleaned Data/PNADC_limpo_2024.dta", replace

*import delimited "Cleaned Data/PNADC_limpo_2023.csv", clear
*save "Cleaned Data/PNADC_limpo_2023.dta", replace

use "Cleaned Data/PNADC_limpo_2023.dta", clear
drop v40582_label
g v40582_label = ""
drop  v40592_label 
g  v40592_label = ""
drop v405921_label
g v405921_label =""
append using "Cleaned Data/PNADC_limpo_2024.dta"

* Creating a single time variable based on ano and trimestre
gen time = (ano - 2023) * 4 + trimestre

* Setting xtset
xtset id_pessoa time

* Generating dummy variable for transition from  empregado_setor_priv to empregado_setor_pub
g transition_priv_to_pub = l.empregado_setor_priv == 1 & empregado_setor_pub == 1
g transition_pub_to_priv = l.empregado_setor_pub == 1 & empregado_setor_priv == 1

* Regressing delta wage on past wage
reg D.renda_habitual_principal l.renda_habitual_principal if transition_priv_to_pub == 1
reg D.log_renda l.log_renda if transition_priv_to_pub == 1
reg D.renda_habitual_principal l.renda_habitual_principal if transition_pub_to_priv == 1

* Discretizing the wage variable into 500 BRL bins 
gen wage_bin = floor(renda_habitual_principal / 500) 
reg D.wage_bin wage_bin if transition_priv_to_pub == 1

* Log renda 
gen log_renda_habitual_principal = log(renda_habitual_principal)


************************************
* 1. Defining variables 
************************************

* 1.1 Sector of employment
g setor = 0 if desocupado == 1 | trab_familiar_aux == 1
replace setor = 1 if empregado_setor_priv == 1 | conta_propria == 1 | empregador == 1 | trab_domestico == 1
replace setor = 2 if empregado_setor_pub == 1
replace setor = . if na_pea == 0
lab def lsetor 0 "Unemployed" 1 "Private Sector" 2 "Public Sector"
lab val setor lsetor

g setor_t1 = f.setor

* 1.1bis Sector of employment 
g setor2 = 0 if desocupado == 1 
replace setor2 = 1 if empregado_setor_priv == 1 
replace setor2 = 2 if empregado_setor_pub == 1
lab def lsetor2 0 "Unemployed" 1 "Private Sector" 2 "Public Sector"
lab val setor2 lsetor2

g setor2_t1 = f.setor2

* 1.2 Transition variables
drop transition_priv_to_pub transition_pub_to_priv
g transition_priv_to_pub = l.empregado_setor_priv == 1 & empregado_setor_pub == 1
g transition_pub_to_priv = l.empregado_setor_pub == 1 & empregado_setor_priv == 1
g transition_priv_to_pub2 = setor2 == 1 & setor2_t1 == 2
g transition_pub_to_priv2 = setor2 == 2 & setor2_t1 == 1

* 1.3 Wage variable
drop log_renda 
gen renda = renda_habitual_principal
// winsorizing renda variable 
levelsof ano, local(anos)
foreach a of local anos {
    sum renda if ano == `a', detail
    local p99_`a' = r(p99)
    local p1_`a' = r(p1)
    replace renda = `p99_`a'' if renda > `p99_`a'' & !missing(renda) & ano == `a'
    replace renda = `p1_`a'' if renda < `p1_`a'' & !missing(renda) & ano == `a'
}

sum renda, detail
local p99 = r(p99)
local p1 = r(p1)
replace renda = `p99' if renda > `p99' & !missing(renda)
replace renda = `p1' if renda < `p1' & !missing(renda)
gen renda_USD = renda / 5.0
gen renda_t1 = f.renda
gen renda_t1_USD = renda_t1 / 5.0
gen log_renda = log(renda)
gen log_renda_t1 = log(renda_t1)
gen delta_log_renda = log_renda_t1 - log_renda
gen delta_renda = renda_t1 - renda
gen delta_renda_relative = ((renda_t1 - renda) / renda)*100
// winsorizing
levelsof ano, local(anos)
foreach vars in delta_renda_relative delta_log_renda {
g `vars'_w = `vars'
    foreach a of local anos {
        sum `vars' [weight= peso] if ano == `a', detail
        local p99_`vars'_`a' = r(p99)
        local p1_`vars'_`a' = r(p1)
        replace `vars'_w = `p99_`vars'_`a'' if `vars' > `p99_`vars'_`a'' & !missing(`vars') & ano == `a'
        replace `vars'_w = `p1_`vars'_`a'' if `vars' < `p1_`vars'_`a'' & !missing(`vars') & ano == `a'
    }
}

* 1.4 Residualizing wages
reg renda i.time idade i.sexo i.raca_cor i.nivel_instrucao i.zona
predict renda_resid, resid
gen renda_t1_resid = f.renda_resid
reg log_renda i.time idade i.sexo i.raca_cor i.nivel_instrucao i.zona
predict log_renda_resid, resid
gen log_renda_t1_resid = f.log_renda_resid
gen delta_log_renda_resid = log_renda_t1 - log_renda_resid
gen delta_renda_resid = renda_t1 - renda_resid




************************************
* 2. Plots
************************************

* Plotting the log distribution of wages
foreach vars in delta_log_renda_w delta_renda_relative_w delta_renda_relative delta_log_renda {
       if "`vars'" == "delta_log_renda_w" local x_title = "Change in Log Wage"
       else if "`vars'" == "delta_renda_relative_w"  local x_title = "Change in Wage (%)"
       sum `vars' [weight= peso] if transition_priv_to_pub == 1, detail
       local mean_priv_to_pub: display %9.2f r(mean)
       local mean_priv_to_pub = stritrim(string(`mean_priv_to_pub'))
       local N_priv_to_pub: display %9.0f r(N)
       local N_priv_to_pub = stritrim(string(`N_priv_to_pub'))
       sum `vars' [weight= peso] if transition_pub_to_priv == 1, detail
       local mean_pub_to_priv: display %9.2f r(mean)
       local N_pub_to_priv: display %9.0f r(N)
       local mean_pub_to_priv = stritrim(string(`mean_pub_to_priv'))
       local N_pub_to_priv = stritrim(string(`N_pub_to_priv'))
    twoway (kdensity `vars' [weight= peso] if transition_priv_to_pub == 1, lcolor(orange) lwidth(medium)) ///
           (kdensity `vars' [weight= peso] if transition_pub_to_priv == 1, lcolor(midblue) lwidth(medium)), ///
           legend(row(1) label(1 "Private to Public") label(2 "Public to Private") pos(6)) ///
           xtitle("`x_title'") ytitle("Density") ///
           xline(`mean_priv_to_pub', lcolor(orange) lwidth(medium) lpattern(dash)) ///
           xline(`mean_pub_to_priv', lcolor(midblue) lwidth(medium) lpattern(dash)) ///
           note("Dashed lines represent the mean change for each transition type. Pub to Priv: `mean_pub_to_priv' (N: `N_pub_to_priv'), Priv to Pub: `mean_priv_to_pub' (N: `N_priv_to_pub')",  size(tiny)) 
    graph export "Output/Figures/figure_2_`vars'.pdf", replace
}

* Plotting the distribution of wages for incumbents vs. switchers 
foreach vars in renda_USD log_renda { 
       if "`vars'" == "renda_USD" local x_title = "Wage (USD)"
       else if "`vars'" == "log_renda"  local x_title = "Log Wage (BRL)"
       sum `vars' [weight= peso] if transition_priv_to_pub == 1, detail
       local mean_priv_to_pub: display %9.2f r(mean)
       local N_priv_to_pub: display %9.0f r(N)
       sum `vars' [weight= peso] if transition_pub_to_priv == 1, detail
       local mean_pub_to_priv: display %9.2f r(mean)
       local N_pub_to_priv: display %9.0f r(N)
       sum `vars' [weight= peso] if setor == 2, detail
       local mean_public_incumbents: display %9.2f r(mean)
       local N_public_incumbents: display %9.0f r(N)
       sum `vars' [weight= peso] if setor == 1, detail
       local mean_private_incumbents: display %9.2f r(mean)
       local N_private_incumbents: display %9.0f r(N)
       foreach lev in priv_to_pub pub_to_priv public_incumbents private_incumbents {
              local mean_`lev' = stritrim(string(`mean_`lev''))
              local N_`lev' = stritrim(string(`N_`lev''))
       }
       twoway (kdensity `vars' if transition_priv_to_pub == 1, lcolor(orange%40) lwidth(medium) lpattern(dash)) ///
              (kdensity `vars' if transition_pub_to_priv == 1, lcolor(midblue%40) lwidth(medium) lpattern(dash)) ///
              (kdensity `vars' if setor == 2, lcolor(orange)  lwidth(medium)) ///
              (kdensity `vars' if setor == 1, lcolor(blue) lwidth(medium)), ///
              legend(row(2) label(1 "Private to Public") label(2 "Public to Private") ///
              label(3 "Public Sector Incumbents") label(4 "Private Sector Incumbents") pos(6)) ///
              xtitle("`x_title'") ytitle("Density") ///
              note("Dashed lines represent switchers, solid lines represent incumbents (in job for more than a year)." "Private to Public: `mean_priv_to_pub' (N: `N_priv_to_pub'), Public to Private: `mean_pub_to_priv' (N: `N_pub_to_priv'), Public Incumbents: `mean_public_incumbents' (N: `N_public_incumbents'), Private Incumbents: `mean_private_incumbents' (N: `N_private_incumbents')", size(tiny))
       graph export "Output/Figures/figure_3_`vars'.pdf", replace
}
ssssss
* Plotting the log distribution of wages by sector
twoway (kdensity log_renda if setor == 1 & ano == 2023, lcolor(orange%40) lwidth(medium)) ///
    (kdensity log_renda if setor == 1 & ano == 2024, lcolor(orange) lwidth(medium)) ///
    (kdensity log_renda if setor == 2 & ano == 2023, lcolor(midblue%40) lwidth(medium)) ///
    (kdensity log_renda if setor == 2 & ano == 2024, lcolor(midblue) lwidth(medium)), ///
    legend(label(1 "Private Sector 2023") label(2 "Private Sector 2024") label(3 "Public Sector 2023") label(4 "Public Sector 2024")) ///
    title("Log Wage Distribution by Sector and Year") xtitle("Log Wage") ytitle("Density")
graph export "Output/Figures/log_wage_distribution_by_sector_and_year.pdf", replace

* Plotting Delta w for public and private sector transitoins
twoway (kdensity delta_log_renda if transition_priv_to_pub == 1, lcolor(orange) lwidth(medium)) ///
       (kdensity delta_log_renda if transition_pub_to_priv == 1, lcolor(midblue) lwidth(medium)), ///
       legend(label(1 "Private to Public") label(2 "Public to Private")) ///
       title("Distribution of Log Wage Changes by Transition Type") xtitle("Change in Log Wage") ytitle("Density")
graph export "Output/Figures/log_wage_change_distribution_by_transition_type.pdf", replace

twoway (kdensity delta_renda if transition_priv_to_pub == 1, lcolor(orange) lwidth(medium)) ///
       (kdensity delta_renda if transition_pub_to_priv == 1, lcolor(midblue) lwidth(medium)), ///
       legend(label(1 "Private to Public") label(2 "Public to Private")) ///
       title("Distribution of Wage Changes by Transition Type") xtitle("Change in Wage (BRL)") ytitle("Density")
graph export "Output/Figures/wage_change_distribution_by_transition_type.pdf", replace

* Plotting w_t1 vs w_t for public and private sector transitions
twoway (lpolyci renda_t1_USD renda_USD if transition_priv_to_pub == 1, bwidth(500) color(orange%25) degree(2)) ///
        (lpolyci renda_t1_USD renda_USD if transition_pub_to_priv == 1, bwidth(500) lcolor(midblue%25) lwidth(medium) degree(2)) ///
        (function y=x, range(0 4000) lcolor(gray%50) lwidth(thin) lpattern(dash)), ///
       legend(label(1 "Private to Public") label(4 "Public to Private")) ///
       title("Wage at t+1 vs Wage at t by Transition Type") xtitle("Wage at t (BRL)") ytitle("Wage at t+1 (BRL)")
graph export "Output/Figures/wage_t1_vs_wage_t_by_transition_type.pdf", replace

twoway (lpolyci log_renda_t1 log_renda if transition_priv_to_pub == 1, bwidth(0.5) color(orange%25) degree(2)) ///
        (lpolyci log_renda_t1 log_renda if transition_pub_to_priv == 1, bwidth(0.5) lcolor(midblue%25) lwidth(medium) degree(2)) ///
        (function y=x, range(0 10) lcolor(gray%50) lwidth(thin) lpattern(dash)), ///
       legend(label(1 "Private to Public") label(4 "Public to Private")) ///
       title("Log Wage at t+1 vs Log Wage at t by Transition Type") xtitle("Log Wage at t") ytitle("Log Wage at t+1")
graph export "Output/Figures/log_wage_t1_vs_log_wage_t_by_transition_type.pdf", replace

* Plotting residualized wages at w_{t+1} vs w_{t} for public and private sector transitions
twoway (lpolyci renda_t1_resid renda_resid if transition_priv_to_pub == 1, bwidth(500) color(orange%25) degree(2)) ///
        (lpolyci renda_t1_resid renda_resid if transition_pub_to_priv == 1, bwidth(500) lcolor(midblue%25) lwidth(medium) degree(2)) ///
        (function y=x, range(-2000 2000) lcolor(gray%50) lwidth(thin) lpattern(dash)), ///
       legend(label(1 "Private to Public") label(4 "Public to Private")) ///
       title("Residualized Wage at t+1 vs Residualized Wage at t by Transition Type") xtitle("Residualized Wage at t (BRL)") ytitle("Residualized Wage at t+1 (BRL)")