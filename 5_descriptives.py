# -*- coding: utf-8 -*-
# ============================================================================
# 5_descriptives.py
# Tabulações simples da PNADC Contínua do 3º trimestre de 2024,
# SÓ para pessoas NA FORÇA DE TRABALHO (ocupadas OU desocupadas procurando):
#   - gênero
#   - faixa de renda
#   - escolaridade
#   - região
#
# Para que serve: descrever essa população para mandar pra empresa de pesquisa
# montar uma amostra representativa. As porcentagens usam o peso da PNADC,
# então representam a POPULAÇÃO (não só a amostra).
#
# Só imprime as tabelas no terminal (não salva arquivo nenhum).
# Código simples e direto (sem funções): ler -> filtrar -> rotular -> contar.
# ============================================================================

import os
import pandas as pd

# mostra os números com 1 casa decimal (fica mais limpo no terminal)
pd.set_option("display.float_format", lambda x: f"{x:.1f}")

# ----------------------------------------------------------------------------
# 1) Caminho dos dados, relativo a ESTE script (funciona em qualquer máquina)
#    Estrutura esperada:  .../PNADC/Code/5_descriptives.py   (este arquivo)
#                         .../PNADC/Cleaned Data/PNADC_limpo_2024.dta
# ----------------------------------------------------------------------------
aqui = os.path.dirname(os.path.abspath(__file__))
caminho = os.path.join(aqui, "..", "Cleaned Data", "PNADC_limpo_2024.dta")

colunas = ["ano", "trimestre", "na_pea", "sexo", "nivel_instrucao_label",
           "renda_efetiva_principal", "peso", "id_uf"]

# ----------------------------------------------------------------------------
# 2) Ler os dados de uma vez só (o superPC tem RAM de sobra) e filtrar:
#    - 3º trimestre de 2024
#    - na_pea == 1  -> pessoas na força de trabalho (ocupadas ou desocupadas).
#      (na_pea vem de VD4001/condicao_forca_trabalho == "Força de trabalho";
#       confere exatamente com ocupado == 1 OU desocupado == 1.)
# ----------------------------------------------------------------------------
dados = pd.read_stata(caminho, columns=colunas)
dados = dados[(dados["ano"] == 2024) & (dados["trimestre"] == 3) & (dados["na_pea"] == 1)]

print("Pessoas na amostra (2024 T3, na força de trabalho):", len(dados))
print("População representada (soma dos pesos):", round(dados["peso"].sum()))
print()

# ----------------------------------------------------------------------------
# 3) Criar rótulos fáceis de entender
# ----------------------------------------------------------------------------
# Gênero: na PNADC, 1 = Homem, 2 = Mulher
dados["genero"] = dados["sexo"].map({1: "Homem", 2: "Mulher"})

# Escolaridade: conserta acento (o arquivo vem com texto "torto", ex.: MÃ©dio)
dados["escolaridade"] = (dados["nivel_instrucao_label"]
                         .str.encode("latin-1", "ignore")
                         .str.decode("utf-8", "ignore"))
dados["escolaridade"] = dados["escolaridade"].replace("", "(sem informação)")

# Região: o primeiro dígito do código da UF (id_uf) indica a região
# 1 = Norte, 2 = Nordeste, 3 = Sudeste, 4 = Sul, 5 = Centro-Oeste
dados["regiao"] = (dados["id_uf"] // 10).map({1: "Norte", 2: "Nordeste",
                                              3: "Sudeste", 4: "Sul",
                                              5: "Centro-Oeste"})

# Faixa de renda (renda do trabalho principal, R$ por mês).
# Aqui os desocupados entram como "Sem renda de trabalho" (não têm trabalho principal).
com_renda = dados[dados["renda_efetiva_principal"] > 0].copy()
limites = [0, 1000, 2000, 3000, 5000, 10000, 100000000]
nomes   = ["Até 1.000", "1.000 a 2.000", "2.000 a 3.000",
           "3.000 a 5.000", "5.000 a 10.000", "Mais de 10.000"]
com_renda["faixa_renda"] = pd.cut(com_renda["renda_efetiva_principal"],
                                  bins=limites, labels=nomes)

# ----------------------------------------------------------------------------
# 4) GÊNERO
#    n_amostra   = quantas pessoas na amostra
#    pct_pop (%) = participação na POPULAÇÃO (usa o peso da PNADC)
# ----------------------------------------------------------------------------
n_amostra = dados.groupby("genero")["peso"].count()
soma_peso = dados.groupby("genero")["peso"].sum()
tabela_genero = pd.DataFrame({"n_amostra": n_amostra,
                              "pct_pop": 100 * soma_peso / soma_peso.sum()})
print("=== GÊNERO ===")
print(tabela_genero)
print()

# ----------------------------------------------------------------------------
# 5) FAIXA DE RENDA (só quem tem renda do trabalho principal > 0)
# ----------------------------------------------------------------------------
n_amostra = com_renda.groupby("faixa_renda")["peso"].count()
soma_peso = com_renda.groupby("faixa_renda")["peso"].sum()
tabela_renda = pd.DataFrame({"n_amostra": n_amostra,
                             "pct_pop": 100 * soma_peso / soma_peso.sum()})
print("=== FAIXA DE RENDA (trabalho principal, R$/mês) ===")
print(tabela_renda)
print("(entre os ocupados com renda do trabalho principal maior que zero)")
print()

# ----------------------------------------------------------------------------
# 6) ESCOLARIDADE
# ----------------------------------------------------------------------------
n_amostra = dados.groupby("escolaridade")["peso"].count()
soma_peso = dados.groupby("escolaridade")["peso"].sum()
tabela_educ = pd.DataFrame({"n_amostra": n_amostra,
                            "pct_pop": 100 * soma_peso / soma_peso.sum()})
print("=== ESCOLARIDADE ===")
print(tabela_educ)
print()

# ----------------------------------------------------------------------------
# 7) REGIÃO
# ----------------------------------------------------------------------------
n_amostra = dados.groupby("regiao")["peso"].count()
soma_peso = dados.groupby("regiao")["peso"].sum()
tabela_regiao = pd.DataFrame({"n_amostra": n_amostra,
                              "pct_pop": 100 * soma_peso / soma_peso.sum()})
print("=== REGIÃO ===")
print(tabela_regiao)
print()
