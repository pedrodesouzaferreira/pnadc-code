cd "/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC/Cleaned Data/"

import delimited "PNADC_harmonized_VD3004_7_2016_2025_light.csv", clear

sum renda_efetiva_principal_real_202 if empregado_setor_priv == 1 & formal == 1 & horas_habituais_principal >= 30, d

// Keeping first observation of individual
bys id_pessoa (ano trimestre): gen first_obs = _n == 1

// Summarizing 
sum renda_habitual_principal_winsor_ [weight = peso] ///
if empregado_setor_priv == 1 & ///
formal == 1 & ///
horas_habituais_principal >= 35 & ///
first_obs == 1 & /// 
!missing(horas_habituais_principal), d

// Summarizing 
sum renda_habitual_principal_winsor_ [weight = peso] ///
if empregado_setor_priv == 1 & ///
formal == 1 & ///
horas_habituais_principal >= 35 & ///
!missing(horas_habituais_principal), d

// Summarizing 
sum renda_habitual_principal_winsor_ [weight = peso] ///
if empregado_setor_pub == 1 & ///
formal == 1 & ///
horas_habituais_principal >= 35 & ///
first_obs == 1 & /// 
!missing(horas_habituais_principal), d

// Summarizing 
sum renda_habitual_principal_winsor_ [weight = peso] ///
if empregado_setor_pub == 1 & ///
formal == 1 & ///
horas_habituais_principal >= 35 & ///
!missing(horas_habituais_principal), d