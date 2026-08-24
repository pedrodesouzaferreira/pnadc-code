cd "/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC/Cleaned Data/"

import delimited "PNADC_harmonized_VD3004_7_2016_2025_light.csv", clear

sum renda_efetiva_principal_real_202 if empregado_setor_priv == 1 & formal == 1 & horas_habituais_principal >= 30, d

// Keeping first observation of individual
bys id_pessoa (ano trimestre): gen first_obs = _n == 1
g idade_22_24 = inrange(idade, 22, 24)

// Global for conditions 
gl pri_first empregado_setor_priv == 1 & formal == 1 & horas_habituais_principal >= 35 & first_obs == 1 & !missing(horas_habituais_principal)
gl pri_all empregado_setor_priv == 1 & formal == 1 & horas_habituais_principal >= 35 & !missing(horas_habituais_principal)
gl pub_first empregado_setor_pub == 1 & formal == 1 & horas_habituais_principal >= 35 & first_obs == 1 & !missing(horas_habituais_principal)
gl pub_all empregado_setor_pub == 1 & formal == 1 & horas_hab
gl pub_first_2022 empregado_setor_pub == 1 & formal == 1 & horas_habituais_principal >= 35 & ano >= 2022 & first_obs == 1 & !missing(horas_habituais_principal)
gl pri_first_2022 empregado_setor_priv == 1 & formal == 1 & horas_habituais_principal >= 35 & ano >= 2022 & first_obs == 1 & !missing(horas_habituais_principal)
gl pub_3qrt_2022 empregado_setor_pub == 1 & formal == 1 & horas_habituais_principal >= 35 & ano >= 2022 & trimestre == 3 & !missing(horas_habituais_principal)
gl pri_3qrt_2022 empregado_setor_priv == 1 & formal == 1 & horas_habituais_principal >= 35 & ano >= 2022 & trimestre == 3 & !missing(horas_habituais_principal)


// First obs
sum renda_habitual_principal_winsor_ [weight = peso]  if $pri_first, d
sum renda_habitual_principal_winsor_ [weight = peso]  if $pub_first, d

// First obs post 2022
sum renda_habitual_principal_winsor_ [weight = peso] if $pri_first_2022, d
sum renda_habitual_principal_winsor_ [weight = peso] if $pub_first_2022, d

// Third quarter 2022
sum renda_habitual_principal_winsor_ [weight = peso] if $pub_3qrt_2022, d
sum renda_habitual_principal_winsor_ [weight = peso] if $pri_3qrt_2022, d
sum renda_habitual_principal_winsor_ [weight = peso] if idade_22_24 == 1 & $pub_3qrt_2022, d
sum renda_habitual_principal_winsor_ [weight = peso] if idade_22_24 == 1 & $pri_3qrt_2022, d

// Third quarter 2022 and nonwinsorized
sum renda_habitual_principal_real_20 [weight = peso] if $pub_3qrt_2022, d
sum renda_habitual_principal_real_20 [weight = peso] if $pri_3qrt_2022, d
