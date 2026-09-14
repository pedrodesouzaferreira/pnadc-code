# -*- coding: utf-8 -*-
# ============================================================================
# 11_rais_descriptives.py
#
# OBJETIVO
#   Mostrar como o mercado de trabalho FORMAL se divide entre emprego PÚBLICO e
#   PRIVADO entre trabalhadores COM ENSINO SUPERIOR e em tempo integral, por
#   Região Geográfica Imediata (RGI).
#
#   Saída: um mapa das RGIs (coroplético pela participação do emprego público)
#   com pizzas saindo das RGIs destacadas, mais uma pizza do agregado.
#
# DUAS RODADAS (constante RODADA, logo abaixo)
#   "AC" = teste rápido com o Acre, destacando Rio Branco.
#   "BR" = Brasil inteiro (27 UFs), destacando Belém, Manaus, Juazeiro do
#          Norte, Recife, Vitória da Conquista, Rio de Janeiro, Itaperuna,
#          São Paulo, Sinop, Brasília (RGI "Distrito Federal") e Blumenau.
#
# COMO RODA EM DUAS ETAPAS
#   1. PAINEL: lê os CSVs da RAIS um por UF, EM PEDAÇOS (chunks), já jogando
#      fora as colunas e as linhas que não interessam, e grava o resultado
#      empilhado num único .parquet. Essa etapa é a caríssima (~18 GB de CSV)
#      e só roda uma vez — depois o parquet é reaproveitado.
#   2. FIGURA: lê o parquet, agrega por RGI e desenha. Roda em segundos.
#   Para forçar a reconstrução do painel: FORCAR_PAINEL = True, ou apague o
#   .parquet, ou rode com o argumento `--rebuild`.
#
# RESTRIÇÕES DA AMOSTRA (todas aplicadas em `aplica_filtros`)
#   (1) tempo integral : quantidade_horas_contratadas >= 36, != 99, não missing
#   (2) emprego formal : é o universo da RAIS (vínculos formais declarados)
#   (3) ensino superior: grau_instrucao_apos_2005 in (9, 10, 11)
#                        = superior completo, mestrado, doutorado
#       + natureza_juridica == 9999 ("Indefinido") é descartada
#       + (opcional, ligado por padrão) vinculo_ativo_3112 == 1, ou seja, o
#         vínculo estava ativo em 31/12 — isso mede o ESTOQUE de empregos no
#         fim do ano em vez de contar todo vínculo que existiu em algum momento
#         de 2022. Sem esse filtro, o setor privado (que tem muito mais
#         rotatividade) aparece inflado. Veja SOMENTE_VINCULOS_ATIVOS_3112.
#
# PÚBLICO vs PRIVADO (natureza_juridica, ver meudicionario.csv)
#   público : natureza_juridica <= 2038  (administração direta, autarquias,
#             fundações, empresas públicas, sociedades de economia mista)
#             + 2194, 2208, 2275 (entidades binacionais)
#   privado : todo o resto
#
# Caminhos são todos RELATIVOS a este arquivo (nada hard-coded). Num cluster,
# a variável de ambiente RAIS_PROJECT_ROOT pode apontar para outra raiz.
# ============================================================================

import os
import re
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch, Patch, Polygon as MplPolygon

# ----------------------------------------------------------------------------
# 0) CONFIGURAÇÃO DA RODADA
# ----------------------------------------------------------------------------
RODADA = "BR"  # "AC" (teste) ou "BR" (Brasil inteiro)

UFS_BRASIL = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
    "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO",
]

# Códigos cod_rgi (6 dígitos) conferidos contra a chave do IBGE.
RGIS_BRASIL = [
    150001,  # Belém
    130001,  # Manaus
    230011,  # Juazeiro do Norte
    260001,  # Recife
    290011,  # Vitória da Conquista
    330001,  # Rio de Janeiro
    330011,  # Itaperuna
    350001,  # São Paulo
    510007,  # Sinop
    530001,  # Distrito Federal (= Brasília)
    420019,  # Blumenau
]

# Nomes de exibição que sobrescrevem o nome_rgi do IBGE, por idioma.
# A RGI de Brasília leva o nome da UF ("Distrito Federal") na base do IBGE.
NOMES_EXIBICAO = {
    530001: {"pt": "Brasília (DF)", "en": "Brasília (DF)"},
}

CONFIGS = {
    "AC": {
        "ufs": ["AC"],
        "destaques": [120001],  # Rio Branco
        "nome_agregado": {"pt": "Acre", "en": "Acre"},
        "fig": (13.5, 7.6),
        # posição da pizza do agregado: None = entra na coluna com as demais
        "pos_total": None,
        "lado_rotulo_total": "dir",
        # retângulo da barra de cor (ou da legenda), em coordenadas de figura
        "pos_barra": (0.05, 0.10, 0.20, 0.018),
        # RGI usada como régua na variante "vs_referencia"
        "rgi_referencia": 120001,  # Rio Branco
        # empurra uma coluna para cima (+) ou para baixo (-), em fração de
        # figura, para desviar linha-guia que passe em cima de outra pizza
        "desloca_coluna": {"esq": 0.0, "dir": 0.0},
    },
    "BR": {
        "ufs": UFS_BRASIL,
        "destaques": RGIS_BRASIL,
        "nome_agregado": {"pt": "Brasil", "en": "Brazil"},
        "fig": (20.0, 15.0),
        # o Brasil deixa o canto sudoeste do mapa vazio: a pizza do total
        # (maior que as outras) mora ali. (x_centro, y_centro, lado)
        "pos_total": (0.325, 0.205, 0.155),
        # o rótulo aponta para o canto vazio, à esquerda, longe do mapa
        "lado_rotulo_total": "esq",
        "pos_barra": (0.255, 0.068, 0.165, 0.015),
        "rgi_referencia": 530001,  # Brasília (Distrito Federal)
        # a coluna da esquerda sobe: assim a linha-guia de Belém passa ACIMA
        # da pizza de Manaus em vez de cortá-la pelo meio
        "desloca_coluna": {"esq": 0.115, "dir": 0.0},
    },
}

CFG = CONFIGS[RODADA]
UFS = CFG["ufs"]
RGIS_DESTAQUE = CFG["destaques"]
NOME_AGREGADO = CFG["nome_agregado"]

# Idiomas a gerar. Cada um vira um arquivo com sufixo _pt / _en.
IDIOMAS = ["pt", "en"]

# Se True, mantém só vínculos ativos em 31/12 (estoque de fim de ano).
# Se False, conta todo vínculo formal observado em 2022 (fluxo).
SOMENTE_VINCULOS_ATIVOS_3112 = True

# RGIs com menos de N_MINIMO vínculos na amostra saem do coroplético (ficam
# cinza de "sem dado"). Sem isso, uma RGI com n = 1 pinta 0% ou 100% e domina
# visualmente a escala de cor — o mapa nacional tem muitas RGIs minúsculas.
# Elas continuam no CSV de saída. Ponha 0 para desligar.
N_MINIMO = 25

# A escala de cor é cortada nesses percentis para não ser comprimida por
# poucas RGIs extremas (prefeitura como único empregador formal => ~80%
# público). A barra de cor ganha setas nas pontas indicando o corte.
PERCENTIS_COR = (5, 95)

# Fatia com menos de LIMIAR_DENTRO % leva o rótulo FORA da pizza. Como as duas
# fatias somam 100, no máximo uma delas sai — e é sempre a minoritária, que é
# justamente a que não tem espaço interno para o texto.
LIMIAR_DENTRO = 45

# Em qual coluna cada RGI destacada entra ("esq" / "dir"). O que não estiver
# aqui é distribuído automaticamente (oeste à esquerda, leste à direita).
# Serve para ajustar à mão quando uma linha-guia fica cruzando o mapa.
LADO_DESTAQUE = {
    # esquerda: só as do oeste/norte, que é onde o mapa tem espaço vazio
    150001: "esq",  # Belém
    130001: "esq",  # Manaus
    510007: "esq",  # Sinop
    530001: "esq",  # Brasília (DF)
    # direita: todo o litoral/sudeste, de norte para sul
    230011: "dir",  # Juazeiro do Norte
    260001: "dir",  # Recife
    290011: "dir",  # Vitória da Conquista
    330011: "dir",  # Itaperuna
    330001: "dir",  # Rio de Janeiro
    350001: "dir",  # São Paulo
    420019: "dir",  # Blumenau
}

# Como o mapa por trás das pizzas é pintado. Uma figura é gerada para cada:
#   "gradiente"     : coroplético contínuo pela participação do emprego público
#   "nenhum"        : sem cor nenhuma — só o contorno das RGIs destacadas
#   "vs_referencia" : duas classes, acima ou abaixo da RGI de referência
VARIANTES_MAPA = ["gradiente", "nenhum", "vs_referencia"]

# Multiplicador aplicado a TODA fonte da figura. 1.0 = tamanho de relatório.
# Rodando com `--slides`, vira ESCALA_SLIDES e os arquivos ganham o sufixo
# _slides — o mesmo desenho, com texto grande o bastante para projeção.
ESCALA_FONTE = 1.0
ESCALA_SLIDES = 1.45


def fs(tamanho):
    """Tamanho de fonte já multiplicado pela escala da rodada."""
    return tamanho * ESCALA_FONTE


SUFIXO_VARIANTE = {
    "gradiente": "gradiente",
    "nenhum": "sem_cor",
    "vs_referencia": "vs_referencia",
}

# cores das duas classes da variante "vs_referencia": o azul continua
# significando "público", igual nas pizzas e no gradiente
COR_ACIMA = "#2a78d6"
COR_ABAIXO = "#dbe7f7"

# Linhas lidas por pedaço na construção do painel. Menor = menos memória.
TAMANHO_CHUNK = 2_000_000

# Reconstrói o painel mesmo que o .parquet já exista.
FORCAR_PAINEL = False

# ----------------------------------------------------------------------------
# 1) CAMINHOS — relativos a ESTE script
#    .../CONCURSOS/Data/PNADC/Code/11_rais_descriptives.py   (este arquivo)
#    .../CONCURSOS/Data/RAIS_Workers/Raw Data/                (insumos)
#    .../CONCURSOS/Data/PNADC/Cleaned Data/                   (painel .parquet)
#    .../CONCURSOS/Data/PNADC/Output/Figures/                 (figuras)
# ----------------------------------------------------------------------------
AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_PNADC = os.environ.get("RAIS_PROJECT_ROOT") or os.path.normpath(
    os.path.join(AQUI, "..")
)
DIR_DATA = os.path.normpath(os.path.join(DIR_PNADC, ".."))
# RAIS_RAW_DIR aponta a pasta dos CSVs para FORA do Dropbox. Isso importa:
# dentro do Dropbox, arquivos grandes viram "placeholder" de 0 byte quando o
# app resolve liberar espaço (atributo com.dropbox.placeholder), e a leitura
# falha com "No columns to parse from file" — como se o arquivo tivesse sido
# corrompido. Mantenha microdados fora de pasta sincronizada.
DIR_RAIS = os.environ.get("RAIS_RAW_DIR") or os.path.join(
    DIR_DATA, "RAIS_Workers", "Raw Data"
)
DIR_CLEANED = os.path.join(DIR_PNADC, "Cleaned Data")
DIR_OUTPUT = os.path.join(DIR_PNADC, "Output")
DIR_FIGURES = os.path.join(DIR_OUTPUT, "Figures")

ARQ_RAIS = os.path.join(DIR_RAIS, "microdados_vinculos_2022_{uf}.csv")
ARQ_CHAVE_RGI = os.path.join(
    DIR_RAIS, "regioes_geograficas_composicao_por_municipios_2017_20180911.csv"
)
ARQ_POLIGONOS = os.path.join(DIR_RAIS, "br_geobr_mapas_regiao_imediata.csv")

# o nome do painel carrega as UFs e o filtro de estoque, pra duas rodadas
# diferentes nunca se sobrescreverem
SUFIXO_PAINEL = f"{len(UFS)}ufs" if len(UFS) > 1 else UFS[0]
ARQ_PAINEL = os.path.join(
    DIR_CLEANED,
    f"rais2022_superior_fulltime_{SUFIXO_PAINEL}"
    f"{'_ativos3112' if SOMENTE_VINCULOS_ATIVOS_3112 else '_todos'}.parquet",
)

for _d in (DIR_CLEANED, DIR_OUTPUT, DIR_FIGURES):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------------
# 2) PALETA E ESTILO
#    Duas categorias => dois tons categóricos validados (azul e laranja).
#    O coroplético usa uma rampa SEQUENCIAL de um único tom (azul, claro->escuro),
#    porque ali a cor codifica magnitude, não identidade.
# ----------------------------------------------------------------------------
COR_PUBLICO = "#2a78d6"
COR_PRIVADO = "#eb6834"
# Branco puro: a figura entra em slide de fundo branco, e um off-white
# deixaria um retângulo visível em volta. SUPERFICIE também é a cor das
# bordas entre RGIs e do vão entre as fatias da pizza, então tudo acompanha.
SUPERFICIE = "#ffffff"
SEM_DADO = "#e6e5e0"
TINTA_FORTE = "#0b0b0b"
TINTA_MEDIA = "#52514e"
TINTA_FRACA = "#8a8983"

# rampa sequencial azul (claro -> escuro) derivada da mesma matiz do público
RAMPA_PUBLICO = LinearSegmentedColormap.from_list(
    "publico_seq", ["#eaf1fb", "#bdd5f2", "#7fb0e6", "#2a78d6", "#12457f"]
)

mpl.rcParams.update(
    {
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "font.family": "DejaVu Sans",
        "text.color": TINTA_FORTE,
        "pdf.fonttype": 42,
    }
)

# ----------------------------------------------------------------------------
# 2b) TEXTOS DA FIGURA, POR IDIOMA
#     Os nomes das RGIs vêm da chave do IBGE e ficam sempre em português —
#     são topônimos, não se traduzem. `sep_milhar` é o separador de milhar
#     usado no "n = ...".
# ----------------------------------------------------------------------------
TEXTOS = {
    "pt": {
        "titulo": "Emprego formal de nível superior: público ou privado",
        "subtitulo": (
            "Vínculos formais em tempo integral (≥36h) com ensino superior "
            "completo ou mais, por Região Geográfica Imediata"
        ),
        "publico": "Público",
        "privado": "Privado",
        "rotulo_barra": "% do emprego no setor público",
        "total": "{nome} (total)",
        "sep_milhar": ".",
        "fonte": "Fonte: RAIS 2022 (vínculos), IBGE (Regiões Geográficas Imediatas, 2017). ",
        "nota_estoque": "Vínculos ativos em 31/12/2022.",
        "nota_fluxo": "Todos os vínculos observados em 2022.",
        "nota_minimo": " RGIs com menos de {n} vínculos na amostra ficam em cinza.",
        "leg_acima": "Mais emprego público que {nome}",
    },
    "en": {
        "titulo": "Formal employment of college graduates: public or private",
        "subtitulo": (
            "Full-time formal jobs (≥36h/week) held by workers with a "
            "completed college degree or above, by Immediate Geographic Region"
        ),
        "publico": "Public",
        "privado": "Private",
        "rotulo_barra": "% of employment in the public sector",
        "total": "{nome} (total)",
        "sep_milhar": ",",
        "fonte": (
            "Source: RAIS 2022 (employment records), IBGE (Immediate Geographic "
            "Regions, 2017). "
        ),
        "nota_estoque": "Jobs active on Dec 31, 2022.",
        "nota_fluxo": "All jobs observed during 2022.",
        "nota_minimo": " RGIs with fewer than {n} jobs in the sample are shown in grey.",
        "leg_acima": "More public sector employment than {nome}",
    },
}


# ----------------------------------------------------------------------------
# 3) PAINEL — lê a RAIS em pedaços, filtra, empilha as UFs e grava .parquet
# ----------------------------------------------------------------------------
# `id_municipio` entra como RESERVA de `id_municipio_trabalho`: no extrato de
# 2024 a coluna de município de TRABALHO vem vazia em ~99% das linhas, e sem a
# reserva o merge descartaria quase a amostra inteira. Onde as duas existem,
# a de trabalho é a certa (é onde o emprego de fato acontece); `id_municipio`
# é o município de REGISTRO do estabelecimento, uma aproximação aceitável.
COLUNAS_RAIS = [
    "sigla_uf",
    "id_municipio_trabalho",
    "id_municipio",
    "quantidade_horas_contratadas",
    "grau_instrucao_apos_2005",
    "natureza_juridica",
    "vinculo_ativo_3112",
]

# colunas que o painel guarda (as outras só servem para filtrar)
COLUNAS_PAINEL = ["sigla_uf", "id_municipio_merge", "natureza_juridica"]

# códigos de natureza_juridica que são públicos apesar de > 2038 (binacionais)
NJ_PUBLICO_EXTRA = {2194, 2208, 2275}
NJ_CORTE_PUBLICO = 2038
NJ_INDEFINIDO = 9999
GRAU_SUPERIOR = {9, 10, 11}  # superior completo, mestrado, doutorado


# etapas do funil, na ordem em que `aplica_filtros` as aplica
ETAPAS = [
    ("lidos", "vínculos lidos"),
    ("superior", "com ensino superior completo ou mais"),
    ("integral", "em tempo integral (≥36h)"),
    ("nj_def", "com natureza jurídica definida"),
    ("ativos", "ativos em 31/12"),
    ("com_mun", "com município para o merge"),
    ("sem_trab_usou_reg", "  destes, via município de registro (reserva)"),
]


def aplica_filtros(d, funil):
    """Aplica as restrições da amostra a um pedaço da RAIS.

    `funil` é um dicionário que acumula a contagem de cada etapa ao longo dos
    pedaços, para o funil poder ser impresso por UF no fim da leitura.
    """
    funil["lidos"] += len(d)

    # (3) ensino superior
    d = d[d["grau_instrucao_apos_2005"].isin(GRAU_SUPERIOR)]
    funil["superior"] += len(d)

    # (1) tempo integral: >= 36h, 99 é código de inválido, e sem missing
    horas = d["quantidade_horas_contratadas"]
    d = d[horas.notna() & (horas != 99) & (horas >= 36)]
    funil["integral"] += len(d)

    # natureza jurídica indefinida (ou ausente) não pode ser classificada.
    # O notna() importa ao escalar: basta uma UF trazer o campo vazio para a
    # coluna virar float e o astype("int16") do painel quebrar no fim.
    nj = d["natureza_juridica"]
    d = d[nj.notna() & (nj != NJ_INDEFINIDO)]
    funil["nj_def"] += len(d)

    # estoque em 31/12 (opcional)
    if SOMENTE_VINCULOS_ATIVOS_3112:
        d = d[d["vinculo_ativo_3112"] == 1]
    funil["ativos"] += len(d)

    # chave do merge: município de TRABALHO quando existe, senão o município
    # de REGISTRO do estabelecimento. Sem nenhum dos dois não há como alocar.
    d = d.copy()
    trab = d["id_municipio_trabalho"] if "id_municipio_trabalho" in d else None
    reg = d["id_municipio"] if "id_municipio" in d else None
    if trab is not None and reg is not None:
        d["id_municipio_merge"] = trab.fillna(reg)
        funil["sem_trab_usou_reg"] += int(trab.isna().sum() - d["id_municipio_merge"].isna().sum())
    elif trab is not None:
        d["id_municipio_merge"] = trab
    elif reg is not None:
        d["id_municipio_merge"] = reg
    else:
        raise SystemExit(
            "Nem id_municipio_trabalho nem id_municipio existem no arquivo — "
            "não há chave para juntar com a RGI."
        )

    d = d[d["id_municipio_merge"].notna()]
    funil["com_mun"] += len(d)

    return d[COLUNAS_PAINEL]


def constroi_painel(ufs, caminho_saida):
    """Empilha as UFs já filtradas num único parquet enxuto.

    Lê em pedaços de TAMANHO_CHUNK linhas e descarta o que não interessa
    antes de acumular, então o pico de memória não depende do tamanho do CSV
    (SP sozinho tem 5,4 GB).
    """
    partes = []
    for uf in ufs:
        caminho = ARQ_RAIS.format(uf=uf)
        if not os.path.exists(caminho):
            print(f"  ATENÇÃO: {os.path.basename(caminho)} não existe — pulando")
            continue

        print(f"  {uf}: lendo {os.path.basename(caminho)} ...", flush=True)

        # arquivo de 0 byte = placeholder do Dropbox (conteúdo só na nuvem).
        # Sem esta checagem o erro aparece lá na frente como se fosse CSV
        # corrompido.
        if os.path.getsize(caminho) == 0:
            print(f"    ATENÇÃO: {os.path.basename(caminho)} tem 0 byte — "
                  "provável placeholder do Dropbox. Pulando.")
            continue

        # extratos diferentes trazem conjuntos de colunas diferentes: lê só o
        # cabeçalho e pede a interseção, senão usecols estoura com KeyError
        disponiveis = set(pd.read_csv(caminho, nrows=0).columns)
        faltando_essencial = {
            "quantidade_horas_contratadas",
            "grau_instrucao_apos_2005",
            "natureza_juridica",
        } - disponiveis
        if faltando_essencial:
            print(f"    ATENÇÃO: faltam colunas {sorted(faltando_essencial)} — pulando")
            continue
        usar = [c for c in COLUNAS_RAIS if c in disponiveis]

        funil = {chave: 0 for chave, _ in ETAPAS}
        pedacos = []
        for pedaco in pd.read_csv(
            caminho, usecols=usar, chunksize=TAMANHO_CHUNK
        ):
            filtrado = aplica_filtros(pedaco, funil)
            if len(filtrado):
                pedacos.append(filtrado)

        # funil de cada UF: dá pra ver em qual restrição a amostra encolhe
        for chave, rotulo in ETAPAS:
            print(f"      {funil[chave]:>12,}  {rotulo}", flush=True)

        if pedacos:
            partes.append(pd.concat(pedacos, ignore_index=True))

    if not partes:
        raise SystemExit("Nenhum arquivo da RAIS foi lido — confira DIR_RAIS.")

    painel = pd.concat(partes, ignore_index=True)

    # tipos enxutos: o painel inteiro fica pequeno o bastante pra caber na RAM
    painel["id_municipio_merge"] = painel["id_municipio_merge"].astype("int32")
    painel["natureza_juridica"] = painel["natureza_juridica"].astype("int16")
    painel["sigla_uf"] = painel["sigla_uf"].astype("category")

    # a checagem que faltou na leva de 2023: sem natureza_juridica < 2000 não
    # existe administração pública direta, e o emprego público zera na análise
    n_adm = int((painel["natureza_juridica"] < 2000).sum())
    if n_adm == 0:
        raise SystemExit(
            "\n" + "!" * 76 + "\n"
            "ABORTADO: o painel não tem NENHUM vínculo de administração\n"
            "pública direta (natureza_juridica < 2000). Foi exatamente esse o\n"
            "defeito dos CSVs de 2023 baixados em out/2024 — o emprego público\n"
            "sai perto de zero e o resultado não significa nada.\n"
            "Rebaixe os dados antes de seguir (11a_rais_import_parallel.py).\n"
            + "!" * 76
        )
    print(f"  administração pública direta: {n_adm:,} vínculos "
          f"({100 * n_adm / len(painel):.1f}%)")

    painel.to_parquet(caminho_saida, index=False)
    mb = os.path.getsize(caminho_saida) / 1e6
    print(f"  painel gravado: {len(painel):,} linhas, {mb:.1f} MB")
    print(f"  {caminho_saida}")
    return painel


def carrega_painel(ufs, refazer=False):
    """Devolve o painel, construindo-o se necessário, com `publico` marcado."""
    if refazer or FORCAR_PAINEL or not os.path.exists(ARQ_PAINEL):
        print("  construindo o painel a partir dos CSVs da RAIS ...")
        painel = constroi_painel(ufs, ARQ_PAINEL)
    else:
        print(f"  reaproveitando o painel já existente ({ARQ_PAINEL})")
        painel = pd.read_parquet(ARQ_PAINEL)
        print(f"  {len(painel):,} linhas")

        # Painel gravado ANTES da coluna de merge ganhar nome próprio: ali a
        # chave se chamava `id_municipio_trabalho`. Renomear aqui evita ter
        # de reconstruir o painel inteiro (horas de leitura de CSV) só por
        # causa do nome da coluna.
        if (
            "id_municipio_merge" not in painel.columns
            and "id_municipio_trabalho" in painel.columns
        ):
            print("  (painel antigo: id_municipio_trabalho -> id_municipio_merge)")
            painel = painel.rename(
                columns={"id_municipio_trabalho": "id_municipio_merge"}
            )

    if "id_municipio_merge" not in painel.columns:
        raise SystemExit(
            "O painel não tem a coluna de município para o merge.\n"
            f"  colunas presentes: {sorted(painel.columns)}\n"
            "  Reconstrua com --rebuild."
        )

    nj = painel["natureza_juridica"]
    painel["publico"] = (nj <= NJ_CORTE_PUBLICO) | nj.isin(NJ_PUBLICO_EXTRA)
    return painel


# ----------------------------------------------------------------------------
# 4) AGREGAÇÃO POR RGI
# ----------------------------------------------------------------------------
def agrega_por_rgi(painel, rgis_validas=None):
    """Cola a chave município->RGI e agrega a composição público/privado.

    `rgis_validas` restringe o resultado às RGIs mapeadas. Isso importa porque
    `sigla_uf` na RAIS segue o município de REGISTRO do estabelecimento, mas
    alocamos o vínculo pelo município de TRABALHO — então o arquivo de uma UF
    contém alguns vínculos que são trabalhados em outra (e vice-versa). Rodando
    o Brasil inteiro isso se resolve sozinho; rodando uma UF só, esses vínculos
    vazados têm de ser descartados para o total não misturar mercados.

    Devolve (tabela por RGI, dicionário do agregado).
    """
    chave = pd.read_csv(ARQ_CHAVE_RGI, encoding="utf-8-sig")
    chave = chave[["CD_GEOCODI", "cod_rgi", "nome_rgi"]].drop_duplicates("CD_GEOCODI")

    junto = painel.merge(
        chave,
        left_on="id_municipio_merge",
        right_on="CD_GEOCODI",
        how="left",
        validate="m:1",
    )

    sem_rgi = junto["cod_rgi"].isna().sum()
    if sem_rgi:
        print(f"  ATENÇÃO: {sem_rgi:,} vínculos sem RGI correspondente (descartados)")
        junto = junto[junto["cod_rgi"].notna()]

    if rgis_validas is not None:
        fora = ~junto["cod_rgi"].astype("int64").isin(rgis_validas)
        if fora.any():
            print(
                f"  {fora.sum():,} vínculos trabalhados fora das UFs mapeadas "
                f"(descartados; UF da RAIS = registro do estabelecimento)"
            )
            junto = junto[~fora]

    por_rgi = (
        junto.groupby(["cod_rgi", "nome_rgi"], as_index=False)
        .agg(n=("publico", "size"), n_publico=("publico", "sum"))
        .assign(
            cod_rgi=lambda t: t["cod_rgi"].astype("int64"),
            n_privado=lambda t: t["n"] - t["n_publico"],
            share_publico=lambda t: 100 * t["n_publico"] / t["n"],
        )
    )
    por_rgi["share_privado"] = 100 - por_rgi["share_publico"]

    # o rótulo do agregado é escolhido por idioma na hora de desenhar
    agregado = {
        "n": len(junto),
        "n_publico": int(junto["publico"].sum()),
        "share_publico": 100 * junto["publico"].mean(),
    }
    agregado["n_privado"] = agregado["n"] - agregado["n_publico"]
    agregado["share_privado"] = 100 - agregado["share_publico"]

    return por_rgi.sort_values("share_publico", ascending=False), agregado


# ----------------------------------------------------------------------------
# 5) POLÍGONOS — o CSV traz WKT (POLYGON / MULTIPOLYGON) na coluna `geometria`.
#    Não há geopandas nesta máquina, então extraímos os anéis com regex e
#    desenhamos cada um com matplotlib. (Buracos não são tratados; as RGIs
#    deste arquivo não têm — os anéis extras são ilhas.)
# ----------------------------------------------------------------------------
_ANEL = re.compile(r"\(([-0-9.,\s]+)\)")


def aneis_do_wkt(wkt):
    """Converte um WKT de (MULTI)POLYGON em lista de arrays (n, 2) lon/lat."""
    aneis = []
    for bloco in _ANEL.findall(wkt):
        pares = [p.split() for p in bloco.split(",") if p.strip()]
        pontos = np.array([[float(x), float(y)] for x, y in pares])
        if len(pontos) >= 3:
            aneis.append(pontos)
    return aneis


def centroide(aneis):
    """Centróide (shoelace) do maior anel — usado para ancorar a linha-guia."""
    maior = max(aneis, key=len)
    x, y = maior[:, 0], maior[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cruz = x * y1 - x1 * y
    area = cruz.sum() / 2
    if abs(area) < 1e-12:
        return x.mean(), y.mean()
    cx = ((x + x1) * cruz).sum() / (6 * area)
    cy = ((y + y1) * cruz).sum() / (6 * area)
    return cx, cy


def carrega_poligonos(ufs):
    """Lê os polígonos das RGIs das UFs pedidas."""
    geo = pd.read_csv(ARQ_POLIGONOS)
    geo = geo[geo["sigla_uf"].isin(ufs)].copy()
    geo["aneis"] = geo["geometria"].map(aneis_do_wkt)
    geo["centro"] = geo["aneis"].map(centroide)
    return geo[["id_uf", "sigla_uf", "id_regiao_imediata", "aneis", "centro"]]


# ----------------------------------------------------------------------------
# 6) DESENHO
# ----------------------------------------------------------------------------
def desenha_mapa(ax, geo, shares, destaques, n_rgis, variante, share_ref=None):
    """Desenha as RGIs. `variante` decide como o interior é pintado.

    "gradiente"     -> coroplético contínuo (devolve a norma, para a barra)
    "nenhum"        -> tudo num cinza neutro, sem codificar nada
    "vs_referencia" -> duas classes, acima/abaixo de `share_ref`

    Devolve a norma da escala contínua, ou None nas outras variantes.
    """
    norma = None
    if variante == "gradiente":
        valores = np.array([v for v in shares.values() if np.isfinite(v)])
        if len(valores):
            vmin, vmax = np.percentile(valores, PERCENTIS_COR)
        else:
            vmin, vmax = 0.0, 100.0
        if vmax - vmin < 1e-9:  # evita norma degenerada com uma única RGI
            vmin, vmax = vmin - 1, vmax + 1
        norma = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    # com 510 RGIs a borda branca tem de ser fina, senão o mapa vira renda
    espessura = 0.8 if n_rgis <= 30 else 0.22
    contorno = 2.0 if n_rgis <= 30 else 1.6

    def cor_da_rgi(cod):
        if variante == "nenhum":
            return SEM_DADO
        share = shares.get(cod, np.nan)
        if not np.isfinite(share):  # RGI sem dado suficiente fica cinza claro
            return SEM_DADO
        if variante == "gradiente":
            return RAMPA_PUBLICO(norma(share))
        # vs_referencia: só duas classes
        return COR_ACIMA if share >= share_ref else COR_ABAIXO

    todos_x, todos_y = [], []
    for _, linha in geo.iterrows():
        cod = int(linha["id_regiao_imediata"])
        cor = cor_da_rgi(cod)

        for anel in linha["aneis"]:
            ax.add_patch(
                MplPolygon(
                    anel,
                    closed=True,
                    facecolor=cor,
                    edgecolor=SUPERFICIE,
                    linewidth=espessura,
                    zorder=2,
                )
            )
            todos_x.append(anel[:, 0])
            todos_y.append(anel[:, 1])

        # contorno grosso só nas RGIs destacadas, por cima de todo o resto.
        # Vai em duas passadas: um halo claro embaixo e a linha escura em
        # cima, senão o contorno desaparece sobre as RGIs de azul escuro.
        if cod in destaques:
            for anel in linha["aneis"]:
                for cor_linha, esp, z in (
                    (SUPERFICIE, contorno * 2.2, 4),
                    (TINTA_FORTE, contorno, 5),
                ):
                    ax.add_patch(
                        MplPolygon(
                            anel,
                            closed=True,
                            facecolor="none",
                            edgecolor=cor_linha,
                            linewidth=esp,
                            zorder=z,
                        )
                    )

    x = np.concatenate(todos_x)
    y = np.concatenate(todos_y)
    folga_x = 0.03 * (x.max() - x.min())
    folga_y = 0.04 * (y.max() - y.min())
    ax.set_xlim(x.min() - folga_x, x.max() + folga_x)
    ax.set_ylim(y.min() - folga_y, y.max() + folga_y)

    # correção simples de latitude (equirretangular): 1 grau de lon encurta
    # por cos(lat), senão o mapa fica esticado na horizontal
    ax.set_aspect(1 / np.cos(np.radians(float(np.mean(y)))))
    ax.axis("off")
    return norma


def desenha_pizza(
    ax, share_publico, share_privado, titulo, n, txt, escala=1.0, lado_rotulo="dir"
):
    """Uma pizza de duas fatias: público vs privado, com rótulos diretos.

    `lado_rotulo` decide para que lado aponta a fatia pública (que é fina e
    leva rótulo externo): "dir" a põe às 3h subindo, "esq" às 9h subindo.
    Nas colunas laterais o rótulo sempre aponta para FORA do mapa, senão ele
    cai em cima da linha-guia ou do próprio mapa.
    """
    ax.set_facecolor(SUPERFICIE)
    fatias, _ = ax.pie(
        [share_publico, share_privado],
        colors=[COR_PUBLICO, COR_PRIVADO],
        startangle=0 if lado_rotulo == "dir" else 180,
        counterclock=(lado_rotulo == "dir"),
        # o vão de 2px na cor da superfície separa as fatias sem linha preta
        wedgeprops={"edgecolor": SUPERFICIE, "linewidth": 2.0},
    )

    fonte_rotulo = fs(10.5) * escala
    # rótulo direto: o número vai na fatia se ela couber, senão vai pra fora
    for fatia, valor, rotulo in zip(
        fatias,
        [share_publico, share_privado],
        [txt["publico"], txt["privado"]],
    ):
        meio = np.radians((fatia.theta1 + fatia.theta2) / 2)
        cos_m, sin_m = np.cos(meio), np.sin(meio)
        if valor >= LIMIAR_DENTRO:
            ax.text(
                0.55 * cos_m,
                0.55 * sin_m,
                f"{rotulo}\n{valor:.0f}%",
                ha="center",
                va="center",
                fontsize=fonte_rotulo,
                color="#ffffff",
                fontweight="bold",
                linespacing=1.25,
            )
        else:
            # fatia fina: rótulo fora, com uma linha-guia curta até a borda
            ax.annotate(
                f"{rotulo} {valor:.0f}%",
                xy=(1.01 * cos_m, 1.01 * sin_m),
                xytext=(1.30 * cos_m, 1.30 * sin_m),
                ha="left" if cos_m >= 0 else "right",
                va="center",
                fontsize=fonte_rotulo,
                color=TINTA_FORTE,
                fontweight="bold",
                arrowprops={
                    "arrowstyle": "-",
                    "color": TINTA_MEDIA,
                    "linewidth": 0.9,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )

    ax.set_title(
        titulo,
        fontsize=fs(12.5) * escala,
        fontweight="bold",
        color=TINTA_FORTE,
        pad=8 * escala,
    )
    ax.text(
        0,
        -1.30,
        f"n = {n:,}".replace(",", txt["sep_milhar"]),
        ha="center",
        va="top",
        fontsize=fs(9) * escala,
        color=TINTA_FRACA,
        transform=ax.transData,
    )
    ax.set_aspect("equal")


def distribui_colunas(destaques, centros):
    """Divide as RGIs destacadas em duas colunas que flanqueiam o mapa.

    As mais a oeste vão para a coluna da esquerda, as mais a leste para a
    direita, e dentro de cada coluna a ordem é de norte para sul. Assim as
    linhas-guia saem de cada lado do mapa sem se cruzarem.

    LADO_DESTAQUE sobrescreve a escolha automática, RGI por RGI.
    """
    com_centro = [(c, centros[c]) for c in destaques if c in centros]

    fixos_esq = [t for t in com_centro if LADO_DESTAQUE.get(t[0]) == "esq"]
    fixos_dir = [t for t in com_centro if LADO_DESTAQUE.get(t[0]) == "dir"]
    livres = sorted(
        [t for t in com_centro if t[0] not in LADO_DESTAQUE], key=lambda t: t[1][0]
    )

    # reparte os livres de forma a equilibrar as duas colunas
    n_esq = max(0, (len(com_centro) // 2) - len(fixos_esq))
    grupo_esq = fixos_esq + livres[:n_esq]
    grupo_dir = fixos_dir + livres[n_esq:]

    # dentro de cada coluna: norte no topo, sul embaixo
    ordena = lambda g: [c for c, _ in sorted(g, key=lambda t: -t[1][1])]
    return ordena(grupo_esq), ordena(grupo_dir)


def nome_exibicao(cod, nome_ibge, idioma):
    """Nome mostrado na pizza (permite sobrescrever o nome_rgi do IBGE)."""
    return NOMES_EXIBICAO.get(cod, {}).get(idioma, nome_ibge)


def monta_figura(
    geo, por_rgi, agregado, destaques, arquivo, idioma, variante="gradiente"
):
    """Mapa no centro, pizzas nas duas laterais, ligadas por linhas-guia."""
    txt = TEXTOS[idioma]
    info = por_rgi.set_index("cod_rgi")
    centros = geo.set_index("id_regiao_imediata")["centro"].to_dict()

    # só entram no coroplético as RGIs com amostra suficiente
    shares = {
        int(r.cod_rgi): r.share_publico
        for r in por_rgi.itertuples()
        if r.n >= N_MINIMO
    }

    # a régua da variante "vs_referencia"
    cod_ref = CFG["rgi_referencia"]
    share_ref = info["share_publico"].get(cod_ref, np.nan)
    if variante == "vs_referencia" and not np.isfinite(share_ref):
        raise SystemExit(
            f"RGI de referência {cod_ref} não tem dado — ajuste rgi_referencia."
        )

    presentes = [c for c in destaques if c in info.index]
    esquerda, direita = distribui_colunas(presentes, centros)
    if CFG["pos_total"] is None:  # sem lugar reservado: o total entra na coluna
        direita = direita + [None]

    fig_w, fig_h = CFG["fig"]
    fig = plt.figure(figsize=(fig_w, fig_h))

    # ---- geometria das colunas de pizza ----
    n_max = max(len(esquerda), len(direita), 1)
    # com muitas pizzas numa coluna, estica a faixa vertical disponível
    topo, base = (0.87, 0.10) if n_max <= 6 else (0.895, 0.065)
    faixa = (topo - base) / n_max
    lado = min(faixa * 0.74, 0.30)  # altura da caixa, em fração de figura
    largura = lado * fig_h / fig_w  # largura que deixa a caixa quadrada
    # fontes menores conforme a coluna fica mais cheia, senão o título de uma
    # pizza encosta no "n =" da pizza de cima
    escala = 1.0 if n_max <= 3 else (0.82 if n_max <= 6 else 0.72)

    # A fatia pública é fina e leva rótulo FORA da pizza, apontando para longe
    # do mapa. Esse rótulo mora além da caixa dos eixos, então cada coluna
    # precisa de uma folga externa, senão o texto sai cortado na borda da
    # figura. `folga` = o quanto o rótulo avança além da caixa + o texto.
    texto = 0.95 / fig_w  # largura reservada para "Público 8%" e a linha-guia
    folga = 0.15 * largura + texto

    margem = 0.008
    x_esq = margem + folga
    x_dir = 1 - margem - largura - folga

    # o mapa ocupa o que sobra entre as colunas (colunas vazias não reservam)
    vao = 0.02
    borda_esq = (x_esq + largura + vao) if esquerda else 0.03
    borda_dir = (1 - x_dir + vao) if direita else 0.03
    ax_mapa = fig.add_axes(
        [borda_esq, base - 0.03, 1 - borda_esq - borda_dir, topo - base + 0.03]
    )
    # ordem de empilhamento: mapa (0) < linhas-guia (2) < pizzas (3), então a
    # linha aparece por cima do mapa mas passa POR TRÁS da pizza
    ax_mapa.set_zorder(0)
    norma = desenha_mapa(
        ax_mapa, geo, shares, set(destaques), len(geo), variante, share_ref
    )

    def caixa(x0, i, n_col, desloca=0.0):
        """Caixa da i-ésima pizza da coluna, centralizada na sua faixa."""
        # colunas com menos pizzas ficam centradas verticalmente, e `desloca`
        # permite subir/descer a coluna inteira para desviar linhas-guia
        sobra = (n_max - n_col) * faixa / 2
        centro_y = topo - sobra + desloca - (i + 0.5) * faixa
        return [x0, centro_y - lado / 2, largura, lado]

    def pizza_de_rgi(ax, cod, lado_rotulo):
        linha = info.loc[cod]
        desenha_pizza(
            ax,
            linha["share_publico"],
            linha["share_privado"],
            nome_exibicao(cod, linha["nome_rgi"], idioma),
            int(linha["n"]),
            txt,
            escala,
            lado_rotulo,
        )

    def pizza_do_total(ax, esc, lado_rotulo):
        desenha_pizza(
            ax,
            agregado["share_publico"],
            agregado["share_privado"],
            txt["total"].format(nome=NOME_AGREGADO[idioma]),
            agregado["n"],
            txt,
            esc,
            lado_rotulo,
        )

    # ---- as duas colunas: rótulo para fora, linha-guia para dentro ----
    colunas = (
        (x_esq, esquerda, 1.06, "esq"),
        (x_dir, direita, -1.06, "dir"),
    )
    for x0, coluna, ancora, lado_rotulo in colunas:
        desloca = CFG["desloca_coluna"].get(lado_rotulo, 0.0)
        for i, cod in enumerate(coluna):
            ax_p = fig.add_axes(caixa(x0, i, len(coluna), desloca))
            ax_p.set_zorder(3)
            if cod is None:  # o total, quando não tem lugar reservado
                pizza_do_total(ax_p, escala, lado_rotulo)
                continue
            pizza_de_rgi(ax_p, cod, lado_rotulo)
            # linha-guia do centróide da RGI até a borda interna da pizza
            cx, cy = centros[cod]
            fig.add_artist(
                ConnectionPatch(
                    xyA=(cx, cy),
                    coordsA=ax_mapa.transData,
                    xyB=(ancora, 0.0),
                    coordsB=ax_p.transData,
                    color=TINTA_MEDIA,
                    linewidth=1.1,
                    linestyle=(0, (4, 3)),
                    zorder=2,
                )
            )

    # ---- pizza do agregado em lugar reservado (rodada BR) ----
    if CFG["pos_total"] is not None:
        xc, yc, lado_t = CFG["pos_total"]
        larg_t = lado_t * fig_h / fig_w
        ax_t = fig.add_axes([xc - larg_t / 2, yc - lado_t / 2, larg_t, lado_t])
        ax_t.set_zorder(3)
        pizza_do_total(ax_t, escala * 1.25, CFG["lado_rotulo_total"])

    # ---- títulos ----
    fig.text(
        0.015,
        0.965,
        txt["titulo"],
        fontsize=fs(19),
        fontweight="bold",
        color=TINTA_FORTE,
    )
    fig.text(
        0.015,
        0.934,
        txt["subtitulo"],
        fontsize=fs(11.5),
        color=TINTA_MEDIA,
    )

    # Sem caixa de legenda: cada fatia já leva o próprio nome e percentual
    # escritos nela, então a identidade das categorias não depende da cor.

    # ---- chave da cor do mapa: barra, legenda de duas classes, ou nada ----
    x_barra, y_barra = CFG["pos_barra"][0], CFG["pos_barra"][1]
    if variante == "gradiente":
        ax_cb = fig.add_axes(CFG["pos_barra"])
        barra = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norma, cmap=RAMPA_PUBLICO),
            cax=ax_cb,
            orientation="horizontal",
            extend="both",  # a escala é cortada nos percentis: sinaliza as pontas
        )
        barra.set_label(
            txt["rotulo_barra"], fontsize=fs(9.5), color=TINTA_MEDIA, labelpad=4
        )
        barra.ax.tick_params(labelsize=fs(8.5), colors=TINTA_MEDIA, length=2)
        barra.outline.set_visible(False)
    elif variante == "vs_referencia":
        # duas classes só se explicam com legenda: nada no mapa diz o corte
        nome_ref = nome_exibicao(
            cod_ref, info.loc[cod_ref, "nome_rgi"], idioma
        )
        # Só a classe escura é rotulada: ela é a afirmação da figura. O azul
        # claro é o "resto" e se entende por oposição, sem precisar de linha
        # própria. Fonte maior que o resto da legenda porque essa frase é o
        # que o público lê de longe.
        fig.legend(
            handles=[
                Patch(
                    facecolor=COR_ACIMA,
                    edgecolor=SUPERFICIE,
                    label=txt["leg_acima"].format(nome=nome_ref),
                ),
            ],
            # ancorada por BAIXO, no mesmo canto da barra de cor: crescer pra
            # cima a mantém longe da pizza do agregado e da nota de fonte
            loc="lower left",
            bbox_to_anchor=(x_barra, max(y_barra - 0.02, 0.035)),
            frameon=False,
            fontsize=fs(13.5),
            labelcolor=TINTA_FORTE,
            handlelength=1.3,
            handleheight=1.1,
            borderaxespad=0,
        )

    nota = txt["fonte"] + (
        txt["nota_estoque"] if SOMENTE_VINCULOS_ATIVOS_3112 else txt["nota_fluxo"]
    )
    # a ressalva do n mínimo só faz sentido onde a cor codifica algo
    if N_MINIMO > 0 and variante != "nenhum":
        nota += txt["nota_minimo"].format(n=N_MINIMO)
    fig.text(0.015, 0.018, nota, fontsize=fs(9), color=TINTA_FRACA)

    fig.savefig(arquivo + ".pdf", dpi=300)
    fig.savefig(arquivo + ".png", dpi=170)
    plt.close(fig)
    print(f"  {os.path.basename(arquivo)}.pdf / .png")


# ----------------------------------------------------------------------------
# 7) RODA
# ----------------------------------------------------------------------------
def main():
    global ESCALA_FONTE
    refazer = "--rebuild" in sys.argv
    slides = "--slides" in sys.argv
    if slides:
        ESCALA_FONTE = ESCALA_SLIDES
    print(f"RODADA = {RODADA}  ({len(UFS)} UF(s), {len(RGIS_DESTAQUE)} destaque(s))")
    if slides:
        print(f"MODO SLIDES: fontes x{ESCALA_SLIDES}, arquivos com sufixo _slides")
    print()

    # `--from-csv` reconstrói as figuras a partir da tabela por RGI já salva,
    # sem tocar no painel nem nos CSVs da RAIS. Serve para refazer a figura
    # noutra máquina levando só dois arquivos: essa tabela (poucos KB) e o
    # CSV de polígonos. É assim que se gera a versão de slides sem ter os
    # microdados por perto.
    do_csv = "--from-csv" in sys.argv

    print("1) polígonos das RGIs")
    geo = carrega_poligonos(UFS)
    rgis_validas = set(geo["id_regiao_imediata"].astype("int64"))
    print(f"  {len(rgis_validas)} RGIs no recorte")

    csv_saida = os.path.join(
        DIR_OUTPUT, f"rais2022_publico_privado_rgi_{SUFIXO_PAINEL}.csv"
    )

    if do_csv:
        print(f"\n2) lendo a tabela pronta: {csv_saida}")
        if not os.path.exists(csv_saida):
            raise SystemExit(
                f"--from-csv precisa de {csv_saida}, que não existe.\n"
                "Copie o CSV da máquina onde a rodada completa aconteceu."
            )
        por_rgi = pd.read_csv(csv_saida)
        # o agregado é a soma das RGIs da tabela — idêntico ao que a rodada
        # completa calcula, já que ela agrupa exatamente essas mesmas linhas
        n = int(por_rgi["n"].sum())
        n_pub = int(por_rgi["n_publico"].sum())
        agregado = {
            "n": n,
            "n_publico": n_pub,
            "n_privado": n - n_pub,
            "share_publico": 100 * n_pub / n,
            "share_privado": 100 - 100 * n_pub / n,
        }
        print(f"  {len(por_rgi)} RGIs")
    else:
        print("\n2) painel da RAIS")
        painel = carrega_painel(UFS, refazer=refazer)

        print("\n3) agregação por RGI")
        por_rgi, agregado = agrega_por_rgi(painel, rgis_validas)

    # tabela no terminal: os destaques, que são o que vai virar pizza
    print("\n   RGIs destacadas:")
    mostra = por_rgi[por_rgi["cod_rgi"].isin(RGIS_DESTAQUE)]
    mostra = mostra[["cod_rgi", "nome_rgi", "n", "n_publico", "share_publico"]]
    print(mostra.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(
        f"\n   {NOME_AGREGADO['pt']}: {agregado['share_publico']:.1f}% público / "
        f"{agregado['share_privado']:.1f}% privado (n = {agregado['n']:,})"
    )

    faltando = set(RGIS_DESTAQUE) - set(por_rgi["cod_rgi"])
    if faltando:
        print(f"\n   ATENÇÃO: destaques sem dado: {sorted(faltando)}")

    # salva a tabela completa pra reaproveitar sem reler o painel
    # (no modo --from-csv ela é a ENTRADA, não faz sentido reescrever)
    if not do_csv:
        por_rgi.to_csv(csv_saida, index=False)
        print(f"\n   {len(por_rgi)} RGIs na tabela -> {csv_saida}")

    print("\n4) figuras")
    # uma figura por (idioma x variante de cor do mapa)
    for variante in VARIANTES_MAPA:
        for idioma in IDIOMAS:
            monta_figura(
                geo,
                por_rgi,
                agregado,
                RGIS_DESTAQUE,
                os.path.join(
                    DIR_FIGURES,
                    f"rais2022_mapa_publico_privado_{SUFIXO_PAINEL}"
                    f"_{SUFIXO_VARIANTE[variante]}_{idioma}"
                    + ("_slides" if slides else ""),
                ),
                idioma,
                variante,
            )


if __name__ == "__main__":
    main()
