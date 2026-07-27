# -*- coding: utf-8 -*-
# ============================================================================
# 5_descriptives.py
# Tabulações simples da PNADC Contínua do 3º trimestre de 2024 (POPULAÇÃO TODA):
#   - gênero
#   - faixa de renda
#   - escolaridade
#   - região
#
# Para que serve: descrever a população para mandar pra empresa de pesquisa
# montar uma amostra representativa. As porcentagens usam o peso da PNADC,
# então representam a POPULAÇÃO (não só a amostra).
#
# Código de propósito simples e direto (sem funções), fácil de ler:
# ler os dados -> filtrar o trimestre -> criar rótulos -> contar.
# ============================================================================

import pandas as pd

# ----------------------------------------------------------------------------
# 1) Ler os dados
#    O arquivo é grande (~8,5 GB), então lemos em PEDAÇOS e já jogamos fora
#    tudo que não é o 3º trimestre de 2024. Assim não estoura a memória.
#    (Se a sua máquina tiver bastante RAM, dá pra trocar tudo isso por uma
#     linha só:  dados = pd.read_stata(caminho, columns=colunas)  e depois
#     filtrar.)
# ----------------------------------------------------------------------------
caminho = "/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC/Cleaned Data/PNADC_limpo_2024.dta"

colunas = ["ano", "trimestre", "sexo", "nivel_instrucao_label",
           "renda_efetiva_principal", "peso", "id_uf"]

pedacos = []
for pedaco in pd.read_stata(caminho, columns=colunas, chunksize=100000):
    so_q3 = pedaco[(pedaco["ano"] == 2024) & (pedaco["trimestre"] == 3)]
    pedacos.append(so_q3)

dados = pd.concat(pedacos)

print("Pessoas na amostra (2024 T3):", len(dados))
print("População representada (soma dos pesos):", round(dados["peso"].sum()))
print()

# ----------------------------------------------------------------------------
# 2) Criar rótulos fáceis de entender
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
# Olhamos só quem tem renda positiva no trabalho principal.
com_renda = dados[dados["renda_efetiva_principal"] > 0].copy()
limites = [0, 1000, 2000, 3000, 5000, 10000, 100000000]
nomes   = ["Até 1.000", "1.000 a 2.000", "2.000 a 3.000",
           "3.000 a 5.000", "5.000 a 10.000", "Mais de 10.000"]
com_renda["faixa_renda"] = pd.cut(com_renda["renda_efetiva_principal"],
                                  bins=limites, labels=nomes)

# Onde salvar as tabelas
pasta_saida = "/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/PNADC/Output/Tables/"

# ----------------------------------------------------------------------------
# 3) GÊNERO
#    n_amostra   = quantas pessoas na amostra
#    pct_pop (%) = participação na POPULAÇÃO (usa o peso da PNADC)
# ----------------------------------------------------------------------------
n_amostra = dados.groupby("genero")["peso"].count()
soma_peso = dados.groupby("genero")["peso"].sum()
tabela_genero = pd.DataFrame({"n_amostra": n_amostra,
                              "pct_pop": round(100 * soma_peso / soma_peso.sum(), 1)})
print("=== GÊNERO ===")
print(tabela_genero)
print()
tabela_genero.to_csv(pasta_saida + "desc_genero_2024T3.csv")

# ----------------------------------------------------------------------------
# 4) FAIXA DE RENDA (só quem tem renda do trabalho principal > 0)
# ----------------------------------------------------------------------------
n_amostra = com_renda.groupby("faixa_renda")["peso"].count()
soma_peso = com_renda.groupby("faixa_renda")["peso"].sum()
tabela_renda = pd.DataFrame({"n_amostra": n_amostra,
                             "pct_pop": round(100 * soma_peso / soma_peso.sum(), 1)})
print("=== FAIXA DE RENDA (trabalho principal, R$/mês) ===")
print(tabela_renda)
print("(entre quem tem renda do trabalho principal maior que zero)")
print()
tabela_renda.to_csv(pasta_saida + "desc_renda_2024T3.csv")

# ----------------------------------------------------------------------------
# 5) ESCOLARIDADE
# ----------------------------------------------------------------------------
n_amostra = dados.groupby("escolaridade")["peso"].count()
soma_peso = dados.groupby("escolaridade")["peso"].sum()
tabela_educ = pd.DataFrame({"n_amostra": n_amostra,
                            "pct_pop": round(100 * soma_peso / soma_peso.sum(), 1)})
print("=== ESCOLARIDADE ===")
print(tabela_educ)
print()
tabela_educ.to_csv(pasta_saida + "desc_escolaridade_2024T3.csv")

# ----------------------------------------------------------------------------
# 6) REGIÃO
# ----------------------------------------------------------------------------
n_amostra = dados.groupby("regiao")["peso"].count()
soma_peso = dados.groupby("regiao")["peso"].sum()
tabela_regiao = pd.DataFrame({"n_amostra": n_amostra,
                              "pct_pop": round(100 * soma_peso / soma_peso.sum(), 1)})
print("=== REGIÃO ===")
print(tabela_regiao)
print()
tabela_regiao.to_csv(pasta_saida + "desc_regiao_2024T3.csv")

print("Pronto! As quatro tabelas também foram salvas em Output/Tables/ como .csv")
