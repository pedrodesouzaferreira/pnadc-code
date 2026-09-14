# -*- coding: utf-8 -*-
# ============================================================================
# 11a_rais_import_parallel.py
#
# Baixa a RAIS (vínculos) do BigQuery da Base dos Dados, uma UF por arquivo,
# EM PARALELO e sem precisar de navegador — feito para rodar no cluster.
#
# POR QUE NÃO USA `basedosdados` NEM `bd.read_sql`
#   1. Autenticação: `bd.read_sql` cai no fluxo OAuth de navegador, que não
#      existe num nó de cálculo. Aqui a credencial é SEMPRE uma chave de
#      service account (JSON), lida de GOOGLE_APPLICATION_CREDENTIALS.
#   2. Memória: `bd.read_sql` materializa a UF inteira num DataFrame. SP tem
#      ~20 milhões de vínculos — isso estoura a RAM. Aqui o resultado é
#      gravado em pedaços, direto para CSV.
#   3. `basedosdados` conflita com geopandas (ver nota do script original).
#      google-cloud-bigquery já vem como dependência dele, então não há
#      pacote novo a instalar.
#
# PRÉ-REQUISITOS (fazer UMA vez, num nó de login)
#   1. Criar uma service account no projeto de billing e baixar a chave JSON:
#        console.cloud.google.com -> IAM & Admin -> Service Accounts
#        -> Create -> conceder "BigQuery User" + "BigQuery Job User"
#        -> Keys -> Add key -> JSON
#   2. Copiar o JSON para o cluster, FORA do Dropbox e com permissão fechada:
#        mkdir -p ~/.config/gcp && chmod 700 ~/.config/gcp
#        # copie o arquivo para ~/.config/gcp/bq_key.json
#        chmod 600 ~/.config/gcp/bq_key.json
#   3. Exportar as variáveis (ou deixar no ~/.bashrc):
#        export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp/bq_key.json
#        export BQ_BILLING_PROJECT=educacaoideias
#        export RAIS_OUT_DIR=/n/netscratch/<lab>/pferreira/RAIS/Raw
#   4. Testar antes de submeter nada:
#        python 11a_rais_import_parallel.py --ufs AC --dry-run
#
# USO
#   python 11a_rais_import_parallel.py --year 2023                 # 27 UFs
#   python 11a_rais_import_parallel.py --year 2023 --workers 8
#   python 11a_rais_import_parallel.py --ufs AC,RO --force         # refaz
#   python 11a_rais_import_parallel.py --year 2023 --all-columns   # 67 colunas
#   python 11a_rais_import_parallel.py --year 2023 --dry-run       # só custo
#
# NOTAS DE CUSTO E TAMANHO
#   Por padrão só as colunas usadas na análise são selecionadas. O BigQuery é
#   colunar, então isso corta o scan (e a conta) e o tamanho do download em
#   cerca de 10x contra `SELECT *`: ~3 GB no lugar de ~30 GB no ano inteiro.
#   `--dry-run` informa quantos bytes cada UF escaneia e o custo estimado,
#   sem executar nada.
#
# CHECAGEM DE INTEGRIDADE
#   Os arquivos de 2023 baixados em out/2024 vieram SEM a administração
#   pública direta (nenhum natureza_juridica < 2000), o que zerava o emprego
#   público na análise. Este script conta esses vínculos em cada UF e grita
#   se vierem zero — o erro não passa mais calado.
# ============================================================================

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

try:
    import google.auth
    from google.cloud import bigquery
except ImportError:  # pragma: no cover
    sys.exit(
        "Falta google-cloud-bigquery. Instale com:\n"
        "    pip install 'google-cloud-bigquery[pandas]' google-cloud-bigquery-storage"
    )

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------------------------
TABELA = "basedosdados.br_me_rais.microdados_vinculos"

UFS_BRASIL = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
    "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO",
]

# Colunas que 11_rais_descriptives.py consome, mais algumas baratas que
# costumam ser pedidas depois (remuneração, ocupação, sexo, idade, raça).
COLUNAS_ANALISE = [
    "ano",
    "sigla_uf",
    "id_municipio",
    "id_municipio_trabalho",
    "quantidade_horas_contratadas",
    "grau_instrucao_apos_2005",
    "natureza_juridica",
    "vinculo_ativo_3112",
    "tipo_vinculo",
    "valor_remuneracao_media",
    "cbo_2002",
    "cnae_2",
    "sexo",
    "idade",
    "raca_cor",
]

# Colunas de código/identificador: gravadas como inteiro anulável para o CSV
# não sair com "1200013.0" quando a coluna tiver algum vazio.
COLUNAS_INTEIRAS = [
    "ano",
    "id_municipio",
    "id_municipio_trabalho",
    "grau_instrucao_apos_2005",
    "natureza_juridica",
    "vinculo_ativo_3112",
    "tipo_vinculo",
    "cbo_2002",
    "cnae_2",
    "sexo",
    "idade",
    "raca_cor",
    "quantidade_horas_contratadas",
]

# Pasta padrão quando RAIS_OUT_DIR não está definido (uso LOCAL, no Mac):
#   .../CONCURSOS/Data/PNADC/Code/este_arquivo.py
#   .../CONCURSOS/Data/RAIS_Workers/Raw Data/   <- destino
# É a mesma pasta que 11_rais_descriptives.py lê, então dá para baixar e
# rodar a análise em seguida sem mexer em caminho nenhum.
_AQUI = os.path.dirname(os.path.abspath(__file__))
DIR_PADRAO_LOCAL = os.path.normpath(
    os.path.join(_AQUI, "..", "..", "RAIS_Workers", "Raw Data")
)

# Credencial padrão, se GOOGLE_APPLICATION_CREDENTIALS não estiver exportada.
CRED_PADRAO = os.path.expanduser("~/.config/gcp/bq_key.json")

PRECO_POR_TIB = 6.25  # USD, on-demand; só para a estimativa do --dry-run
BYTES_POR_TIB = 1024**4

# teto de bytes cobrados por consulta — trava de segurança contra engano em
# WHERE que faça varrer a tabela toda
LIMITE_BYTES_POR_CONSULTA = 80 * 1024**3  # 80 GiB

_local = threading.local()
_trava_log = threading.Lock()


def log(msg):
    """print com timestamp, serializado entre as threads."""
    with _trava_log:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------
# CREDENCIAL — sempre offline (service account), nunca navegador
# ----------------------------------------------------------------------------
def carrega_credencial():
    """Lê o JSON de GOOGLE_APPLICATION_CREDENTIALS e devolve (cred, projeto).

    Aceita os DOIS formatos, porque nem toda organização deixa criar chave de
    service account (política `iam.disableServiceAccountKeyCreation`):

      "type": "service_account"  -> chave baixada do console / gcloud
      "type": "authorized_user"  -> credencial de usuário criada por
                                    `gcloud auth application-default login
                                     --no-launch-browser`

    Os dois funcionam sem navegador na hora de rodar. O que NÃO serve é
    depender do fluxo interativo durante o job.
    """
    caminho = os.path.expanduser(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "") or CRED_PADRAO
    )
    projeto = os.environ.get("BQ_BILLING_PROJECT", "")

    if not caminho or not os.path.exists(caminho):
        sys.exit(
            "ERRO: GOOGLE_APPLICATION_CREDENTIALS não aponta para um arquivo.\n"
            "  Este script NUNCA abre navegador. Use uma das duas opções:\n"
            "    a) chave JSON de service account (veja PRÉ-REQUISITOS); ou\n"
            "    b) gcloud auth application-default login --no-launch-browser\n"
            "       e aponte para ~/.config/gcloud/application_default_credentials.json\n"
            f"  valor atual: {caminho!r}"
        )
    if not projeto:
        sys.exit(
            "ERRO: defina BQ_BILLING_PROJECT com o projeto que paga as\n"
            "  consultas, ex.: export BQ_BILLING_PROJECT=educacaoideias"
        )

    try:
        # load_credentials_from_file entende os dois formatos de JSON
        cred, _ = google.auth.load_credentials_from_file(
            caminho, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    except Exception as e:
        sys.exit(
            f"ERRO ao ler a credencial em {caminho}:\n"
            f"  {type(e).__name__}: {e}\n"
            "  O arquivo precisa ser JSON com \"type\" igual a\n"
            "  \"service_account\" ou \"authorized_user\"."
        )

    # aviso de higiene: chave longa-vida em pasta sincronizada é vazamento
    if any(p in caminho for p in ("Dropbox", "Google Drive", "OneDrive")):
        log("AVISO: a credencial está numa pasta sincronizada na nuvem. "
            "Mova para ~/.config/gcp/ e use chmod 600.")

    return cred, projeto


def cliente(cred, projeto):
    """Um cliente BigQuery por thread (evita disputa no mesmo objeto)."""
    if not hasattr(_local, "cli"):
        _local.cli = bigquery.Client(project=projeto, credentials=cred)
    return _local.cli


# ----------------------------------------------------------------------------
# CONSULTA
# ----------------------------------------------------------------------------
def monta_query(ano, uf, todas_colunas):
    """SQL de uma UF. O filtro por `ano` aproveita o particionamento."""
    campos = "*" if todas_colunas else ",\n       ".join(COLUNAS_ANALISE)
    return f"""
        SELECT {campos}
        FROM `{TABELA}`
        WHERE ano = {ano} AND sigla_uf = '{uf}'
    """


def estima_bytes(cli, ano, uf, todas_colunas):
    """dry run: quantos bytes a consulta escanearia, sem executar."""
    cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    job = cli.query(monta_query(ano, uf, todas_colunas), job_config=cfg)
    return job.total_bytes_processed or 0


def baixa_uf(cred, projeto, ano, uf, dir_saida, todas_colunas, forcar):
    """Baixa uma UF para CSV, em pedaços. Devolve um dicionário de resultado."""
    destino = os.path.join(dir_saida, f"microdados_vinculos_{ano}_{uf}.csv")
    parcial = destino + ".parcial"

    # retomada: pula quem já existe e não está vazio
    if not forcar and os.path.exists(destino) and os.path.getsize(destino) > 0:
        mb = os.path.getsize(destino) / 1e6
        log(f"{uf}: já existe ({mb:,.1f} MB) — pulando (use --force para refazer)")
        return {"uf": uf, "status": "pulado", "linhas": None, "adm_direta": None}

    t0 = time.time()
    cli = cliente(cred, projeto)
    cfg = bigquery.QueryJobConfig(maximum_bytes_billed=LIMITE_BYTES_POR_CONSULTA)

    try:
        job = cli.query(monta_query(ano, uf, todas_colunas), job_config=cfg)
        resultado = job.result()
    except Exception as e:
        log(f"{uf}: FALHOU na consulta -> {type(e).__name__}: {e}")
        return {"uf": uf, "status": "erro", "erro": str(e)}

    escaneado = (job.total_bytes_processed or 0) / 1e9
    linhas = 0
    adm_direta = 0  # natureza_juridica < 2000 = administração pública direta
    primeiro = True

    try:
        # streaming: nunca carrega a UF inteira na memória
        try:
            pedacos = resultado.to_dataframe_iterable()
        except Exception:
            # versões antigas do cliente não têm o iterável
            pedacos = iter([resultado.to_dataframe()])

        if os.path.exists(parcial):
            os.remove(parcial)

        for pedaco in pedacos:
            if "natureza_juridica" in pedaco.columns:
                nj = pd.to_numeric(pedaco["natureza_juridica"], errors="coerce")
                adm_direta += int((nj < 2000).sum())

            # códigos como inteiro anulável: sem ".0" espalhado pelo CSV
            for col in COLUNAS_INTEIRAS:
                if col in pedaco.columns:
                    pedaco[col] = pd.to_numeric(
                        pedaco[col], errors="coerce"
                    ).astype("Int64")

            pedaco.to_csv(parcial, mode="w" if primeiro else "a",
                          header=primeiro, index=False)
            linhas += len(pedaco)
            primeiro = False

        if primeiro:  # a consulta não devolveu linha nenhuma
            log(f"{uf}: ATENÇÃO — consulta sem resultado. Ano {ano} existe?")
            return {"uf": uf, "status": "vazio", "linhas": 0, "adm_direta": 0}

        # rename atômico: um download interrompido nunca deixa CSV truncado
        # se passando por completo na lógica de retomada
        os.replace(parcial, destino)

    except Exception as e:
        log(f"{uf}: FALHOU ao gravar -> {type(e).__name__}: {e}")
        if os.path.exists(parcial):
            os.remove(parcial)
        return {"uf": uf, "status": "erro", "erro": str(e)}

    mb = os.path.getsize(destino) / 1e6
    dt = time.time() - t0
    aviso = "  <<< SEM ADM. PÚBLICA DIRETA!" if adm_direta == 0 else ""
    log(
        f"{uf}: {linhas:>10,} linhas | {mb:>8,.1f} MB | scan {escaneado:5.1f} GB "
        f"| {dt:5.0f}s | adm.direta {adm_direta:>9,}{aviso}"
    )
    return {
        "uf": uf,
        "status": "ok",
        "linhas": linhas,
        "adm_direta": adm_direta,
        "mb": mb,
        "segundos": dt,
    }


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Baixa a RAIS do BigQuery por UF, em paralelo, sem navegador."
    )
    ap.add_argument("--year", type=int, default=2023, help="ano da RAIS (padrão 2023)")
    ap.add_argument(
        "--ufs", default="", help="lista separada por vírgula (padrão: as 27)"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="downloads simultâneos (padrão 4, bom para internet de casa; "
             "no cluster o .sbatch passa 6)",
    )
    ap.add_argument(
        "--out",
        default=os.environ.get("RAIS_OUT_DIR", "") or DIR_PADRAO_LOCAL,
        help="pasta de saída (padrão: $RAIS_OUT_DIR, senão a pasta "
             "RAIS_Workers/Raw Data do próprio projeto)",
    )
    ap.add_argument(
        "--all-columns",
        action="store_true",
        help="baixa as 67 colunas em vez das da análise (~10x maior e mais caro)",
    )
    ap.add_argument(
        "--force", action="store_true", help="rebaixa UFs cujo arquivo já existe"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="só estima bytes escaneados e custo; não baixa nada",
    )
    args = ap.parse_args()

    ufs = [u.strip().upper() for u in args.ufs.split(",") if u.strip()] or UFS_BRASIL

    cred, projeto = carrega_credencial()

    # ---- dry run: custo antes de gastar ----
    if args.dry_run:
        log(f"dry run | ano {args.year} | {len(ufs)} UF(s) | projeto {projeto}")
        cli = cliente(cred, projeto)
        total = 0
        for uf in ufs:
            try:
                b = estima_bytes(cli, args.year, uf, args.all_columns)
            except Exception as e:
                log(f"{uf}: erro no dry run -> {type(e).__name__}: {e}")
                continue
            total += b
            log(f"{uf}: {b / 1e9:8.2f} GB escaneados")
        log(
            f"TOTAL: {total / 1e9:,.1f} GB  (~US$ "
            f"{total / BYTES_POR_TIB * PRECO_POR_TIB:,.2f} on-demand)"
        )
        colunas = "todas as 67" if args.all_columns else f"{len(COLUNAS_ANALISE)}"
        log(f"colunas selecionadas: {colunas}")
        return

    # ---- saída ----
    if not args.out:
        sys.exit(
            "ERRO: defina a pasta de saída com --out ou RAIS_OUT_DIR.\n"
            "  Use disco de scratch, NÃO o home (quota) e NÃO o Dropbox (sync)."
        )
    dir_saida = os.path.expanduser(args.out)
    os.makedirs(dir_saida, exist_ok=True)

    log(f"ano {args.year} | {len(ufs)} UF(s) | {args.workers} threads")
    log(f"projeto de billing: {projeto}")
    log(f"saída: {dir_saida}")
    log(f"colunas: {'todas' if args.all_columns else str(len(COLUNAS_ANALISE))}")
    log("-" * 78)

    t0 = time.time()
    resultados = []
    # I/O-bound (consulta + rede + disco): threads bastam e não duplicam RAM
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futuros = {
            pool.submit(
                baixa_uf,
                cred,
                projeto,
                args.year,
                uf,
                dir_saida,
                args.all_columns,
                args.force,
            ): uf
            for uf in ufs
        }
        for fut in as_completed(futuros):
            uf = futuros[fut]
            try:
                resultados.append(fut.result())
            except Exception as e:
                log(f"{uf}: exceção não tratada -> {type(e).__name__}: {e}")
                resultados.append({"uf": uf, "status": "erro", "erro": str(e)})

    # ---- resumo ----
    log("-" * 78)
    ok = [r for r in resultados if r["status"] == "ok"]
    pulados = [r for r in resultados if r["status"] == "pulado"]
    ruins = [r for r in resultados if r["status"] in ("erro", "vazio")]

    log(f"concluído em {(time.time() - t0) / 60:.1f} min")
    log(f"  baixadas : {len(ok)}")
    log(f"  puladas  : {len(pulados)}")
    log(f"  problemas: {len(ruins)}")

    if ok:
        log(f"  linhas   : {sum(r['linhas'] for r in ok):,}")
        log(f"  tamanho  : {sum(r['mb'] for r in ok) / 1000:,.1f} GB")

    # a checagem que faltou na leva de 2023
    sem_adm = [r["uf"] for r in ok if r["adm_direta"] == 0]
    if sem_adm:
        log("")
        log("!" * 78)
        log("ATENÇÃO: estas UFs vieram SEM administração pública direta")
        log(f"  (nenhum natureza_juridica < 2000): {', '.join(sorted(sem_adm))}")
        log("  Foi exatamente esse o defeito da leva de 2023: o emprego público")
        log("  zera na análise. NÃO use estes arquivos — confira a tabela de")
        log("  origem antes de seguir.")
        log("!" * 78)

    if ruins:
        log("")
        log(f"UFs com problema: {', '.join(sorted(r['uf'] for r in ruins))}")
        log("Rode o script de novo: as UFs já prontas são puladas.")
        sys.exit(1)


if __name__ == "__main__":
    main()
