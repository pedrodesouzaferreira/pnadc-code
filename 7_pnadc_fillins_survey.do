cd "/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC/"

import delimited "Cleaned Data/PNADC_harmonized_VD3004_7_2016_2025_light.csv", clear

sum renda_efetiva_principal_real_202 if empregado_setor_priv == 1 & horas_habituais_principal >= 30, d

// Generating variables
bys id_pessoa (ano trimestre): gen first_obs = _n == 1
g idade_22_24 = inrange(idade, 22, 24)
g time = (ano - 2016) * 4 + trimestre
xtset id_pessoa time
sort id_pessoa time
g desocupado_f4 = f4.desocupado if !missing(f4.desocupado)
br id_pessoa time ano trimestre desocupado_f4 desocupado 

// Alternative public-sector definition, replicating define_public_sector
// from 10_fillins_pnadc_fulltime.do. The harmonized empregado_setor_pub
// flag is broader (raw VD4009 posicao_emprego codes 5/6/7 -- public with or
// without carteira, or militar/estatutario). This redefinition instead
// requires statutory-servant status, a small set of occupation codes under
// posicao_trab_principal == 2, or a CLT government position (== 4) with
// carteira_assinada == 1 -- and excludes temporary (V4025 == 1) workers.
// Needs V4010, V4025, posicao_trab_principal, servidor_publico_estatutario,
// and carteira_assinada in the light CSV (2a_harmonize_makelighter.py).
gen byte public_universe = !missing(empregado_setor_pub)
gen byte public_rule = ///
    servidor_publico_estatutario == 1 | ///
    (inlist(v4010, 5412, 511, 512, 411, 412) & posicao_trab_principal == 2) | ///
    (posicao_trab_principal == 4 & carteira_assinada == 1)

gen byte empregado_setor_pub_refined = 0 if public_universe
replace empregado_setor_pub_refined = 1 if public_universe & public_rule
replace empregado_setor_pub_refined = 0 if public_universe & v4025 == 1

drop public_universe public_rule

// Generating leads for sector
g priv_f4 = f4.empregado_setor_priv if !missing(f4.empregado_setor_priv)
g pub_f4 = f4.empregado_setor_pub_refined if !missing(f4.empregado_setor_pub_refined)
g estat_f4 = f4.servidor_publico_estatutario == 1 if !missing(f4.empregado_setor_pub_refined)
g horas_habituais_principal_f4 = f4.horas_habituais_principal if !missing(f4.horas_habituais_principal)
g byte fulltime_f4 = horas_habituais_principal_f4 >= 35 if !missing(horas_habituais_principal_f4)
g byte pub_ft_f4 = 0 if !missing(pub_f4)
replace pub_ft_f4 = 1 if pub_f4 == 1 & fulltime_f4 == 1
replace pub_ft_f4 = . if pub_f4 == 1 & missing(fulltime_f4)
g byte priv_ft_f4 = 0 if !missing(priv_f4)
replace priv_ft_f4 = 1 if priv_f4 == 1 & fulltime_f4 == 1
replace priv_ft_f4 = . if priv_f4 == 1 & missing(fulltime_f4)
g byte estat_ft_f4 = 0 if !missing(estat_f4)
replace estat_ft_f4 = 1 if estat_f4 == 1 & fulltime_f4 == 1
replace estat_ft_f4 = . if estat_f4 == 1 & missing(fulltime_f4)

// How much the two public-sector definitions disagree
tab empregado_setor_pub empregado_setor_pub_refined, m

* Job-search groups for Treatment C. With microdados only, "regularly studies
* for exams" is proxied by the PNADC search-method flag V4072A == 5:
* buscando_via_concurso == 1.
// gen byte job_search = (desocupado == 1) if !missing(desocupado)
gen byte job_search = (tomou_providencia_busca == 1) 
gen byte exam_search = (job_search == 1  & buscando_via_concurso == 1)

* "Focuses on private vacancies" proxy: unemployed higher-ed respondents whose
* main search method is direct employer contact, ads, private agency/syndicate,
* or relatives/friends. This excludes concurso, own business, other, and
* no effective search action.
gen byte private_focus = (job_search == 1  & ///
    inlist(metodo_busca_emprego, 1, 2, 3, 4, 6))


// Global for conditions
gl pri_first empregado_setor_priv == 1 & horas_habituais_principal >= 35 & first_obs == 1 & !missing(horas_habituais_principal)
gl pri_all empregado_setor_priv == 1 & horas_habituais_principal >= 35 & !missing(horas_habituais_principal)
gl pub_first empregado_setor_pub_refined == 1 & horas_habituais_principal >= 35 & first_obs == 1 & !missing(horas_habituais_principal)
gl pub_all empregado_setor_pub_refined == 1 & horas_hab
gl pub_first_2022 empregado_setor_pub_refined == 1 & horas_habituais_principal >= 35 & ano >= 2022 & first_obs == 1 & !missing(horas_habituais_principal)
gl pri_first_2022 empregado_setor_priv == 1 & horas_habituais_principal >= 35 & ano >= 2022 & first_obs == 1 & !missing(horas_habituais_principal)
gl pub_3qrt_2022 empregado_setor_pub_refined == 1 & horas_habituais_principal >= 35 & ano >= 2022 & trimestre == 3 & !missing(horas_habituais_principal) & renda_habitual_principal_real_20 > 0
gl pri_3qrt_2022 empregado_setor_priv == 1 & horas_habituais_principal >= 35 & ano >= 2022 & trimestre == 3 & !missing(horas_habituais_principal) & renda_habitual_principal_real_20 > 0
gl pub_2022 empregado_setor_pub_refined == 1 & horas_habituais_principal >= 35 & ano >= 2022 & !missing(horas_habituais_principal) & renda_habitual_principal_real_20 > 0
gl pri_2022 empregado_setor_priv == 1 & horas_habituais_principal >= 35 & ano >= 2022 & !missing(horas_habituais_principal) & renda_habitual_principal_real_20 > 0
gl pub empregado_setor_pub_refined == 1 & horas_habituais_principal >= 35 & !missing(horas_habituais_principal) & renda_habitual_principal_real_20 > 0
gl pri empregado_setor_priv == 1 & horas_habituais_principal >= 35 & !missing(horas_habituais_principal) & renda_habitual_principal_real_20 > 0
gl estat_3qrt_2022 servidor_publico_estatutario == 1 & horas_habituais_principal >= 35 & ano >= 2022 & trimestre == 3 & !missing(horas_habituais_principal) & renda_habitual_principal_real_20 > 0
gl estat servidor_publico_estatutario == 1 & horas_habituais_principal >= 35 & !missing(horas_habituais_principal) & renda_habitual_principal_real_20 > 0


// Requires the estout package: ssc install estout
capture which esttab
if _rc ssc install estout

// (PILOT 1) Wages: Third quarter 2022 and nonwinsorized
estpost summarize renda_habitual_principal_real_20 [aweight = peso] if $pub_3qrt_2022, detail
eststo wage_estat
estpost summarize renda_habitual_principal_real_20 [aweight = peso] if $pri_3qrt_2022, detail
eststo wage_pri
estpost summarize renda_habitual_principal_real_20 [aweight = peso] if idade_22_24 == 1 & $pub_3qrt_2022, detail
eststo wage_estat_2224
estpost summarize renda_habitual_principal_real_20 [aweight = peso] if idade_22_24 == 1 & $pri_3qrt_2022, detail
eststo wage_pri_2224

esttab wage_estat wage_pri wage_estat_2224 wage_pri_2224 using "Output/pnadc_pilot1_wages.csv", ///
    replace cells("p50(fmt(2)) p90(fmt(2)) count(fmt(0))") mtitles("Publico" "Privado" "Publico 22-24" "Privado 22-24") noobs nonumber


// (PILOT 1) Stability
estpost summarize desocupado_f4 [aweight = peso] if $pub
eststo stab_estat
estpost summarize desocupado_f4 [aweight = peso] if $pri
eststo stab_pri
estpost summarize desocupado_f4 [aweight = peso] if idade_22_24 == 1 & $pub
eststo stab_estat_2224
estpost summarize desocupado_f4 [aweight = peso] if idade_22_24 == 1 & $pri
eststo stab_pri_2224

esttab stab_estat stab_pri stab_estat_2224 stab_pri_2224 using "Output/pnadc_pilot1_stability.csv", ///
    replace cells("mean(fmt(4)) count(fmt(0))") mtitles("Publico" "Privado" "Publico 22-24" "Privado 22-24") noobs nonumber


// (PILOT 1) Job finding rates
estpost summarize pub_ft_f4 [aweight = peso] if private_focus == 1
eststo jf_estat_privfocus
estpost summarize priv_ft_f4 [aweight = peso] if private_focus == 1
eststo jf_pri_privfocus
estpost summarize pub_ft_f4 [aweight = peso] if idade_22_24 == 1 & private_focus == 1
eststo jf_estat_privfocus_2224
estpost summarize priv_ft_f4 [aweight = peso] if idade_22_24 == 1 & private_focus == 1
eststo jf_pri_privfocus_2224

estpost summarize pub_ft_f4 [aweight = peso] if exam_search == 1
eststo jf_estat_exam
estpost summarize priv_ft_f4 [aweight = peso] if exam_search == 1
eststo jf_pri_exam
estpost summarize pub_ft_f4 [aweight = peso] if idade_22_24 == 1 & exam_search == 1
eststo jf_estat_exam_2224
estpost summarize priv_ft_f4 [aweight = peso] if idade_22_24 == 1 & exam_search == 1
eststo jf_pri_exam_2224

esttab jf_estat_privfocus jf_pri_privfocus jf_estat_privfocus_2224 jf_pri_privfocus_2224 ///
    jf_estat_exam jf_pri_exam jf_estat_exam_2224 jf_pri_exam_2224 using "Output/pnadc_pilot1_jobfinding.csv", ///
    replace cells("mean(fmt(4)) count(fmt(0))") ///
    mtitles("Publico privfocus" "Privado privfocus" "Publico privfocus 22-24" "Privado privfocus 22-24" ///
        "Publico exam" "Privado exam" "Publico exam 22-24" "Privado exam 22-24") noobs nonumber


* In PILOT 2, we will use the first wage observation for each individual after 2022, rather than the third quarter.


// (PILOT 2) Wages: All quarters post 2022 and nonwinsorized
estpost summarize renda_habitual_principal_real_20 [aweight = peso] if $pub_first_2022, detail
eststo wage_pub
estpost summarize renda_habitual_principal_real_20 [aweight = peso] if $pri_first_2022, detail
eststo wage_pri
estpost summarize renda_habitual_principal_real_20 [aweight = peso] if idade_22_24 == 1 & $pub_first_2022, detail
eststo wage_pub_2224
estpost summarize renda_habitual_principal_real_20 [aweight = peso] if idade_22_24 == 1 & $pri_first_2022, detail
eststo wage_pri_2224

esttab wage_pub wage_pri wage_pub_2224 wage_pri_2224 using "Output/pnadc_pilot2_wages.csv", ///
    replace cells("p50(fmt(2)) p90(fmt(2)) count(fmt(0))") mtitles("Publico" "Privado" "Publico 22-24" "Privado 22-24") noobs nonumber


// (PILOT 2) Stability
estpost summarize desocupado_f4 [aweight = peso] if $pub
eststo stab_pub
estpost summarize desocupado_f4 [aweight = peso] if $pri
eststo stab_pri
estpost summarize desocupado_f4 [aweight = peso] if idade_22_24 == 1 & $pub
eststo stab_pub_2224
estpost summarize desocupado_f4 [aweight = peso] if idade_22_24 == 1 & $pri
eststo stab_pri_2224

esttab stab_pub stab_pri stab_pub_2224 stab_pri_2224 using "Output/pnadc_pilot2_stability.csv", ///
    replace cells("mean(fmt(4)) count(fmt(0))") mtitles("Estatutario" "Privado" "Estatutario 22-24" "Privado 22-24") noobs nonumber


// (PILOT 2) Job finding rates
estpost summarize pub_ft_f4 [aweight = peso] if private_focus == 1
eststo jf_pub_privfocus
estpost summarize priv_ft_f4 [aweight = peso] if private_focus == 1
eststo jf_pri_privfocus
estpost summarize pub_ft_f4 [aweight = peso] if idade_22_24 == 1 & private_focus == 1
eststo jf_pub_privfocus_2224
estpost summarize priv_ft_f4 [aweight = peso] if idade_22_24 == 1 & private_focus == 1
eststo jf_pri_privfocus_2224

estpost summarize pub_ft_f4 [aweight = peso] if exam_search == 1
eststo jf_pub_exam
estpost summarize priv_ft_f4 [aweight = peso] if exam_search == 1
eststo jf_pri_exam
estpost summarize pub_ft_f4 [aweight = peso] if idade_22_24 == 1 & exam_search == 1
eststo jf_pub_exam_2224
estpost summarize priv_ft_f4 [aweight = peso] if idade_22_24 == 1 & exam_search == 1
eststo jf_pri_exam_2224

esttab jf_pub_privfocus jf_pri_privfocus jf_pub_privfocus_2224 jf_pri_privfocus_2224 ///
    jf_pub_exam jf_pri_exam jf_pub_exam_2224 jf_pri_exam_2224 using "Output/pnadc_pilot2_jobfinding.csv", ///
    replace cells("mean(fmt(4)) count(fmt(0))") ///
    mtitles("Estatutario privfocus" "Privado privfocus" "Estatutario privfocus 22-24" "Privado privfocus 22-24" ///
        "Estatutario exam" "Privado exam" "Estatutario exam 22-24" "Privado exam 22-24") noobs nonumber





