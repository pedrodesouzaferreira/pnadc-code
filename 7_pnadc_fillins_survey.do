cd "/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC/Cleaned Data/"

import delimited "PNADC_harmonized_VD3004_7_2016_2025_light.csv", clear

sum renda_efetiva_principal_real_202 if empregado_setor_priv == 1 & horas_habituais_principal >= 30, d

// Generating variables
bys id_pessoa (ano trimestre): gen first_obs = _n == 1
g idade_22_24 = inrange(idade, 22, 24)

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
    (inlist(occ_code, 5412, 511, 512, 411, 412) & posicao_trab_principal == 2) | ///
    (posicao_trab_principal == 4 & carteira_assinada == 1)

gen byte empregado_setor_pub_refined = 0 if public_universe
replace empregado_setor_pub_refined = 1 if public_universe & public_rule
replace empregado_setor_pub_refined = 0 if public_universe & V4025 == 1

drop occ_code public_universe public_rule

// How much the two public-sector definitions disagree
tab empregado_setor_pub empregado_setor_pub_refined, m

// Global for conditions
gl pri_first empregado_setor_priv == 1 & horas_habituais_principal >= 35 & first_obs == 1 & !missing(horas_habituais_principal)
gl pri_all empregado_setor_priv == 1 & horas_habituais_principal >= 35 & !missing(horas_habituais_principal)
gl pub_first empregado_setor_pub == 1 & horas_habituais_principal >= 35 & first_obs == 1 & !missing(horas_habituais_principal)
gl pub_all empregado_setor_pub == 1 & horas_hab
gl pub_first_2022 empregado_setor_pub == 1 & horas_habituais_principal >= 35 & ano >= 2022 & first_obs == 1 & !missing(horas_habituais_principal)
gl pri_first_2022 empregado_setor_priv == 1 & horas_habituais_principal >= 35 & ano >= 2022 & first_obs == 1 & !missing(horas_habituais_principal)
gl pub_3qrt_2022 empregado_setor_pub == 1 & horas_habituais_principal >= 35 & ano >= 2022 & trimestre == 3 & !missing(horas_habituais_principal)
gl pri_3qrt_2022 empregado_setor_priv == 1 & horas_habituais_principal >= 35 & ano >= 2022 & trimestre == 3 & !missing(horas_habituais_principal)


// First obs
sum renda_habitual_principal_winsor_ [weight = peso]  if $pri_first, d
sum renda_habitual_principal_winsor_ [weight = peso]  if $pub_first, d

// First obs post 2022
sum renda_habitual_principal_winsor_ [weight = peso] if $pri_first_2022, d
sum renda_habitual_principal_winsor_ [weight = peso] if $pub_first_2022, d

// Third quarter 2022 (very close)
sum renda_habitual_principal_winsor_ [weight = peso] if $pub_3qrt_2022, d
sum renda_habitual_principal_winsor_ [weight = peso] if $pri_3qrt_2022, d
sum renda_habitual_principal_winsor_ [weight = peso] if idade_22_24 == 1 & $pub_3qrt_2022, d
sum renda_habitual_principal_winsor_ [weight = peso] if idade_22_24 == 1 & $pri_3qrt_2022, d

// Third quarter 2022 and nonwinsorized
sum renda_habitual_principal_real_20 [weight = peso] if $pub_3qrt_2022, d
sum renda_habitual_principal_real_20 [weight = peso] if $pri_3qrt_2022, d
