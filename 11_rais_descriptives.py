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
# ESTA VERSÃO É O TESTE COM O ACRE (destaque: Rio Branco), antes de escalar
#   para o Brasil inteiro com Belém, Manaus, Juazeiro do Norte, Recife,
#   Vitória da Conquista, Rio de Janeiro, Itaperuna, São Paulo, Sinop,
#   Brasília e Blumenau.
#
# RESTRIÇÕES DA AMOSTRA (todas aplicadas em `carrega_rais`)
#   (1) tempo integral : quantidade_horas_contratadas >= 36, != 99, não missing
#   (2) emprego formal : é o universo da RAIS (vínculos formais declarados)
#   (3) ensino superior: grau_instrucao_apos_2005 in (9, 10, 11)
#                        = superior completo, mestrado, doutorado
#       + natureza_juridica == 9999 ("Indefinido") é descartada
#       + (opcional, ligado por padrão) vinculo_ativo_3112 == 1, ou seja, o
#         vínculo estava ativo em 31/12 — isso mede o ESTOQUE de empregos no
#         fim do ano em vez de contar todo vínculo que existiu em algum momento
#         de 2023. Sem esse filtro, o setor privado (que tem muito mais
#         rotatividade) aparece inflado. Veja SOMENTE_VINCULOS_ATIVOS_3112.
#
# PÚBLICO vs PRIVADO (natureza_juridica, ver meudicionario.csv)
#   público : natureza_juridica <= 2038  (administração direta, autarquias,
#             fundações, empresas públicas, sociedades de economia mista)
#             + 2194, 2208, 2275 (entidades binacionais)
#   privado : todo o resto
#
# Caminhos são todos RELATIVOS a este arquivo (nada hard-coded).
# ============================================================================

import os
import re

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch, Polygon as MplPolygon

# ----------------------------------------------------------------------------
# 0) CONFIGURAÇÃO DA RODADA
# ----------------------------------------------------------------------------
# UFs a carregar da RAIS. Para escalar, troque por ["AC", "AM", "PA", ...] ou
# pela lista completa das 27 UFs.
UFS = ["AC"]

# RGIs a destacar com pizza própria (pelo código cod_rgi de 6 dígitos).
# 120001 = Rio Branco. Os nomes vêm da própria chave de municípios.
RGIS_DESTAQUE = [120001]

# Rótulo do agregado mostrado na última pizza, em cada idioma.
NOME_AGREGADO = {"pt": "Acre", "en": "Acre"}

# Idiomas a gerar. Cada um vira um arquivo com sufixo _pt / _en.
IDIOMAS = ["pt", "en"]

# --- para escalar para o Brasil (códigos já conferidos na chave do IBGE) ---
# UFS = ["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT","PA",
#        "PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"]
# NOME_AGREGADO = {"pt": "Brasil", "en": "Brazil"}
# RGIS_DESTAQUE = [
#     150001,  # Belém
#     130001,  # Manaus
#     230011,  # Juazeiro do Norte
#     260001,  # Recife
#     290011,  # Vitória da Conquista
#     330001,  # Rio de Janeiro
#     330011,  # Itaperuna
#     350001,  # São Paulo
#     510007,  # Sinop
#     530001,  # Distrito Federal (= Brasília; a RGI leva o nome da UF)
#     420019,  # Blumenau
# ]
# ATENÇÃO: com 11 destaques a coluna única de pizzas à direita não cabe —
# `monta_figura` precisa distribuir as pizzas em volta do mapa antes de rodar
# o Brasil. As demais etapas (leitura, filtros, merge, agregação) escalam
# direto; só o arranjo da figura muda.

# Se True, mantém só vínculos ativos em 31/12 (estoque de fim de ano).
# Se False, conta todo vínculo formal observado em 2023 (fluxo).
SOMENTE_VINCULOS_ATIVOS_3112 = True

# ----------------------------------------------------------------------------
# 1) CAMINHOS — relativos a ESTE script
#    .../CONCURSOS/Data/PNADC/Code/11_rais_descriptives.py   (este arquivo)
#    .../CONCURSOS/Data/RAIS_Workers/Raw Data/                (insumos)
#    .../CONCURSOS/Data/PNADC/Output/                         (saída)
# ----------------------------------------------------------------------------
AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_PNADC = os.path.normpath(os.path.join(AQUI, ".."))
DIR_DATA = os.path.normpath(os.path.join(DIR_PNADC, ".."))
DIR_RAIS = os.path.join(DIR_DATA, "RAIS_Workers", "Raw Data")
DIR_OUTPUT = os.path.join(DIR_PNADC, "Output")
DIR_FIGURES = os.path.join(DIR_OUTPUT, "Figures")

ARQ_RAIS = os.path.join(DIR_RAIS, "microdados_vinculos_2023_{uf}.csv")
ARQ_CHAVE_RGI = os.path.join(
    DIR_RAIS, "regioes_geograficas_composicao_por_municipios_2017_20180911.csv"
)
ARQ_POLIGONOS = os.path.join(DIR_RAIS, "br_geobr_mapas_regiao_imediata.csv")

os.makedirs(DIR_OUTPUT, exist_ok=True)
os.makedirs(DIR_FIGURES, exist_ok=True)

# ----------------------------------------------------------------------------
# 2) PALETA E ESTILO
#    Duas categorias => dois tons categóricos validados (azul e laranja).
#    O coroplético usa uma rampa SEQUENCIAL de um único tom (azul, claro->escuro),
#    porque ali a cor codifica magnitude, não identidade.
# ----------------------------------------------------------------------------
COR_PUBLICO = "#2a78d6"
COR_PRIVADO = "#eb6834"
SUPERFICIE = "#fcfcfb"
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
        "leg_publico": "Setor público",
        "leg_privado": "Setor privado",
        "rotulo_barra": "% do emprego no setor público",
        "total": "{nome} (total)",
        "sep_milhar": ".",
        "fonte": "Fonte: RAIS 2023 (vínculos), IBGE (Regiões Geográficas Imediatas, 2017). ",
        "nota_estoque": "Vínculos ativos em 31/12/2023.",
        "nota_fluxo": "Todos os vínculos observados em 2023.",
    },
    "en": {
        "titulo": "Formal employment of college graduates: public or private",
        "subtitulo": (
            "Full-time formal jobs (≥36h/week) held by workers with a "
            "completed college degree or above, by Immediate Geographic Region"
        ),
        "publico": "Public",
        "privado": "Private",
        "leg_publico": "Public sector",
        "leg_privado": "Private sector",
        "rotulo_barra": "% of employment in the public sector",
        "total": "{nome} (total)",
        "sep_milhar": ",",
        "fonte": (
            "Source: RAIS 2023 (employment records), IBGE (Immediate Geographic "
            "Regions, 2017). "
        ),
        "nota_estoque": "Jobs active on Dec 31, 2023.",
        "nota_fluxo": "All jobs observed during 2023.",
    },
}


# ----------------------------------------------------------------------------
# 3) LEITURA E FILTRAGEM DA RAIS
# ----------------------------------------------------------------------------
COLUNAS_RAIS = [
    "sigla_uf",
    "id_municipio_trabalho",
    "quantidade_horas_contratadas",
    "grau_instrucao_apos_2005",
    "natureza_juridica",
    "vinculo_ativo_3112",
]

# códigos de natureza_juridica que são públicos apesar de > 2038 (binacionais)
NJ_PUBLICO_EXTRA = {2194, 2208, 2275}
NJ_CORTE_PUBLICO = 2038
NJ_INDEFINIDO = 9999
GRAU_SUPERIOR = {9, 10, 11}  # superior completo, mestrado, doutorado


def carrega_rais(ufs):
    """Lê as UFs pedidas, aplica as restrições da amostra e classifica o setor.

    Devolve um DataFrame de vínculos com a coluna booleana `publico`.
    """
    partes = []
    for uf in ufs:
        caminho = ARQ_RAIS.format(uf=uf)
        print(f"  lendo {os.path.basename(caminho)} ...", flush=True)
        bruto = pd.read_csv(caminho, usecols=COLUNAS_RAIS)
        n0 = len(bruto)

        # (3) ensino superior
        d = bruto[bruto["grau_instrucao_apos_2005"].isin(GRAU_SUPERIOR)]

        # (1) tempo integral: >= 36h, 99 é código de inválido, e sem missing
        horas = d["quantidade_horas_contratadas"]
        d = d[horas.notna() & (horas != 99) & (horas >= 36)]

        # natureza jurídica indefinida não pode ser classificada
        d = d[d["natureza_juridica"] != NJ_INDEFINIDO]

        # estoque em 31/12 (opcional)
        if SOMENTE_VINCULOS_ATIVOS_3112:
            d = d[d["vinculo_ativo_3112"] == 1]

        # município de trabalho é a chave do merge; sem ele não há como alocar
        d = d[d["id_municipio_trabalho"].notna()]

        print(f"    {n0:,} vínculos -> {len(d):,} após as restrições")
        partes.append(d)

    rais = pd.concat(partes, ignore_index=True)

    nj = rais["natureza_juridica"]
    rais["publico"] = (nj <= NJ_CORTE_PUBLICO) | nj.isin(NJ_PUBLICO_EXTRA)
    rais["id_municipio_trabalho"] = rais["id_municipio_trabalho"].astype("int64")
    return rais


def agrega_por_rgi(rais, rgis_validas=None):
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

    junto = rais.merge(
        chave,
        left_on="id_municipio_trabalho",
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
# 4) POLÍGONOS — o CSV traz WKT (POLYGON / MULTIPOLYGON) na coluna `geometria`.
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
# 5) DESENHO
# ----------------------------------------------------------------------------
def desenha_mapa(ax, geo, por_rgi, destaques):
    """Coroplético das RGIs pela participação do emprego público."""
    shares = por_rgi.set_index("cod_rgi")["share_publico"].to_dict()

    valores = [v for v in shares.values() if np.isfinite(v)]
    vmin, vmax = (min(valores), max(valores)) if valores else (0, 100)
    if vmax - vmin < 1e-9:  # evita norma degenerada com uma única RGI
        vmin, vmax = vmin - 1, vmax + 1
    norma = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    todos_x, todos_y = [], []
    for _, linha in geo.iterrows():
        cod = int(linha["id_regiao_imediata"])
        share = shares.get(cod, np.nan)
        # RGI sem nenhum vínculo na amostra fica cinza claro (sem dado)
        cor = "#e6e5e0" if not np.isfinite(share) else RAMPA_PUBLICO(norma(share))
        eh_destaque = cod in destaques

        for anel in linha["aneis"]:
            ax.add_patch(
                MplPolygon(
                    anel,
                    closed=True,
                    facecolor=cor,
                    edgecolor=SUPERFICIE,
                    linewidth=0.8,
                    zorder=2,
                )
            )
            todos_x.append(anel[:, 0])
            todos_y.append(anel[:, 1])

        # contorno grosso só nas RGIs destacadas, por cima de todo o resto
        if eh_destaque:
            for anel in linha["aneis"]:
                ax.add_patch(
                    MplPolygon(
                        anel,
                        closed=True,
                        facecolor="none",
                        edgecolor=TINTA_FORTE,
                        linewidth=2.0,
                        zorder=4,
                    )
                )

    x = np.concatenate(todos_x)
    y = np.concatenate(todos_y)
    folga_x = 0.03 * (x.max() - x.min())
    folga_y = 0.06 * (y.max() - y.min())
    ax.set_xlim(x.min() - folga_x, x.max() + folga_x)
    ax.set_ylim(y.min() - folga_y, y.max() + folga_y)

    # correção simples de latitude (equirretangular): 1 grau de lon encurta
    # por cos(lat), senão o mapa fica esticado na horizontal
    ax.set_aspect(1 / np.cos(np.radians(float(np.mean(y)))))
    ax.axis("off")
    return norma


def desenha_pizza(ax, share_publico, share_privado, titulo, n, txt):
    """Uma pizza de duas fatias: público vs privado, com rótulos diretos."""
    ax.set_facecolor(SUPERFICIE)
    # começa às 3h e sobe: a fatia pública (quase sempre a menor) fica à
    # direita, num ângulo raso, onde o rótulo externo não bate no título
    fatias, _ = ax.pie(
        [share_publico, share_privado],
        colors=[COR_PUBLICO, COR_PRIVADO],
        startangle=0,
        counterclock=True,
        # o vão de 2px na cor da superfície separa as fatias sem linha preta
        wedgeprops={"edgecolor": SUPERFICIE, "linewidth": 2.0},
    )

    # rótulo direto: o número vai na fatia se ela couber, senão vai pra fora
    for fatia, valor, rotulo in zip(
        fatias,
        [share_publico, share_privado],
        [txt["publico"], txt["privado"]],
    ):
        meio = np.radians((fatia.theta1 + fatia.theta2) / 2)
        cos_m, sin_m = np.cos(meio), np.sin(meio)
        if valor >= 18:
            ax.text(
                0.60 * cos_m,
                0.60 * sin_m,
                f"{rotulo}\n{valor:.0f}%",
                ha="center",
                va="center",
                fontsize=10.5,
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
                fontsize=10.5,
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

    ax.set_title(titulo, fontsize=12.5, fontweight="bold", color=TINTA_FORTE, pad=8)
    ax.text(
        0,
        -1.42,
        f"n = {n:,}".replace(",", txt["sep_milhar"]),
        ha="center",
        va="top",
        fontsize=9,
        color=TINTA_FRACA,
        transform=ax.transData,
    )
    ax.set_aspect("equal")


def monta_figura(geo, por_rgi, agregado, destaques, arquivo, idioma):
    """Mapa à esquerda + coluna de pizzas à direita, ligadas por linhas-guia."""
    txt = TEXTOS[idioma]
    info = por_rgi.set_index("cod_rgi")
    pizzas = [c for c in destaques if c in info.index]

    fig_w, fig_h = 13.5, 7.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax_mapa = fig.add_axes([0.01, 0.10, 0.60, 0.76])
    norma = desenha_mapa(ax_mapa, geo, por_rgi, set(destaques))

    # a coluna da direita recebe as RGIs destacadas e, no fim, o agregado.
    # `lado` é a altura da caixa em fração de figura; a largura é corrigida
    # pela razão de aspecto para a caixa ficar quadrada em polegadas.
    n_pizzas = len(pizzas) + 1
    faixa = 0.76 / n_pizzas
    lado = min(faixa * 0.70, 0.30)
    largura = lado * fig_h / fig_w
    esquerda = 0.70
    centros = geo.set_index("id_regiao_imediata")["centro"].to_dict()

    def caixa(i):
        """Caixa da i-ésima pizza, centralizada na sua faixa vertical."""
        centro_y = 0.86 - (i + 0.5) * faixa
        return [esquerda, centro_y - lado / 2, largura, lado]

    for i, cod in enumerate(pizzas):
        linha = info.loc[cod]
        ax_p = fig.add_axes(caixa(i))
        desenha_pizza(
            ax_p,
            linha["share_publico"],
            linha["share_privado"],
            linha["nome_rgi"],
            int(linha["n"]),
            txt,
        )
        # linha-guia do centróide da RGI até a borda esquerda da pizza
        cx, cy = centros[cod]
        fig.add_artist(
            ConnectionPatch(
                xyA=(cx, cy),
                coordsA=ax_mapa.transData,
                xyB=(-1.06, 0.0),
                coordsB=ax_p.transData,
                color=TINTA_MEDIA,
                linewidth=1.2,
                linestyle=(0, (4, 3)),
                zorder=1,
            )
        )

    # pizza do agregado, sem linha-guia (não corresponde a uma RGI)
    ax_t = fig.add_axes(caixa(len(pizzas)))
    desenha_pizza(
        ax_t,
        agregado["share_publico"],
        agregado["share_privado"],
        txt["total"].format(nome=NOME_AGREGADO[idioma]),
        agregado["n"],
        txt,
    )

    # ---- títulos ----
    fig.text(
        0.02,
        0.955,
        txt["titulo"],
        fontsize=17,
        fontweight="bold",
        color=TINTA_FORTE,
    )
    fig.text(
        0.02,
        0.915,
        txt["subtitulo"],
        fontsize=10.5,
        color=TINTA_MEDIA,
    )

    # Sem caixa de legenda: cada fatia já leva o próprio nome e percentual
    # escritos nela, então a identidade das categorias não depende da cor.

    # ---- barra de cor do coroplético ----
    ax_cb = fig.add_axes([0.05, 0.10, 0.20, 0.018])
    barra = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norma, cmap=RAMPA_PUBLICO),
        cax=ax_cb,
        orientation="horizontal",
    )
    barra.set_label(txt["rotulo_barra"], fontsize=9, color=TINTA_MEDIA, labelpad=4)
    barra.ax.tick_params(labelsize=8, colors=TINTA_MEDIA, length=2)
    barra.outline.set_visible(False)

    fig.text(
        0.02,
        0.02,
        txt["fonte"]
        + (
            txt["nota_estoque"]
            if SOMENTE_VINCULOS_ATIVOS_3112
            else txt["nota_fluxo"]
        ),
        fontsize=8.5,
        color=TINTA_FRACA,
    )

    fig.savefig(arquivo + ".pdf", dpi=300)
    fig.savefig(arquivo + ".png", dpi=200)
    plt.close(fig)
    print(f"  figura salva em {arquivo}.pdf / .png")


# ----------------------------------------------------------------------------
# 6) RODA
# ----------------------------------------------------------------------------
def main():
    print("1) polígonos das RGIs")
    geo = carrega_poligonos(UFS)
    rgis_validas = set(geo["id_regiao_imediata"].astype("int64"))
    print(f"  {len(rgis_validas)} RGIs em {', '.join(UFS)}")

    print("2) RAIS")
    rais = carrega_rais(UFS)

    print("3) agregação por RGI")
    por_rgi, agregado = agrega_por_rgi(rais, rgis_validas)

    # tabela no terminal, pra conferir os números que entram na figura
    print("\n   Composição por RGI (vínculos formais, tempo integral, superior+):")
    mostra = por_rgi[["cod_rgi", "nome_rgi", "n", "n_publico", "share_publico"]]
    print(mostra.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(
        f"\n   {NOME_AGREGADO['pt']}: {agregado['share_publico']:.1f}% público / "
        f"{agregado['share_privado']:.1f}% privado (n = {agregado['n']:,})"
    )

    # salva a tabela pra reaproveitar depois sem reler a RAIS
    csv_saida = os.path.join(
        DIR_OUTPUT, f"rais2023_publico_privado_rgi_{'_'.join(UFS)}.csv"
    )
    por_rgi.to_csv(csv_saida, index=False)
    print(f"\n   tabela salva em {csv_saida}")

    print("\n4) figuras")
    sem_dado = rgis_validas - set(por_rgi["cod_rgi"])
    if sem_dado:
        print(f"  RGIs sem nenhum vínculo na amostra (cinza no mapa): {sorted(sem_dado)}")

    # mesma figura em cada idioma pedido: só os textos mudam
    for idioma in IDIOMAS:
        monta_figura(
            geo,
            por_rgi,
            agregado,
            RGIS_DESTAQUE,
            os.path.join(
                DIR_FIGURES,
                f"rais2023_mapa_publico_privado_{'_'.join(UFS)}_{idioma}",
            ),
            idioma,
        )


if __name__ == "__main__":
    main()
