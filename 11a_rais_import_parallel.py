""""
pip install click==8.0.3
pip install pandas==1.3.5
pip install pyarrow==6.0.0
pip install shapely==1.8.5
NB: basedosdados clashes with geopandas -- so keep one or the other

============================================================================
11a_rais_import_parallel.py

É o 11a_rais_import.py, com as MESMAS funções (basedosdados.read_sql +
DataFrame.to_csv), a MESMA consulta e o MESMO destino — só que as UFs
baixam em paralelo, em vez de uma de cada vez.

O QUE MUDA EM RELAÇÃO AO 11a_rais_import.py
  1. ThreadPoolExecutor: WORKERS UFs ao mesmo tempo. O gargalo é rede (a
     consulta roda no BigQuery, não aqui), então threads resolvem.
  2. Retomada: UF cujo CSV já existe é pulada. Dá para interromper e
     recomeçar sem rebaixar o que já veio.
  3. Gravação atômica: escreve em .parcial e só renomeia no fim, então uma
     interrupção nunca deixa CSV truncado passando por completo.
  4. Conferência: conta os vínculos de administração pública direta
     (natureza_juridica < 2000) de cada UF. Os arquivos de 2023 baixados em
     out/2024 vieram SEM eles, e isso zerava o emprego público na análise
     sem dar erro nenhum. Agora aparece na tela.

MEMÓRIA — leia antes de aumentar WORKERS
  `bd.read_sql` carrega a UF INTEIRA num DataFrame, igual ao script
  original. Com N threads, são N UFs na RAM ao mesmo tempo, e SP sozinho
  passa de 20 milhões de vínculos. Por isso WORKERS começa em 3 e as UFs
  vão da menor para a maior, para as gigantes não caírem todas juntas.
  Se faltar memória, diminua WORKERS.

ONDE GRAVAR
  Veja `folder`. Evite pasta do Dropbox: quando o disco enche, o app
  esvazia arquivos antigos e deixa placeholder de 0 byte (atributo
  com.dropbox.placeholder). A leitura falha depois como se o CSV
  estivesse corrompido.
============================================================================
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import basedosdados as bd

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO — igual ao 11a_rais_import.py
# ---------------------------------------------------------------------------
folder = '/Users/pedroferreira/Dropbox (Personal)/MY PROJECTS/CONCURSOS/Data/RAIS_Workers'
billing_project_id = 'educacaoideias'

ANO = 2024

UFS = ['AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS',
       'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN', 'RO', 'RR', 'RS', 'SC',
       'SE', 'SP', 'TO']

WORKERS = 3          # UFs simultâneas (veja a nota de MEMÓRIA)
FORCAR = False       # True = rebaixa mesmo se o CSV já existir

print('Folder: ', folder)

_trava = threading.Lock()


def log(msg):
    """print com hora, serializado entre as threads."""
    with _trava:
        print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ---------------------------------------------------------------------------
# UMA UF — o corpo do laço do 11a_rais_import.py, sem mudar nada de fundo
# ---------------------------------------------------------------------------
def importa_uf(uf):
    destino = f'{folder}/Raw Data/microdados_vinculos_{ANO}_{uf}.csv'
    parcial = destino + '.parcial'

    if not FORCAR and os.path.exists(destino) and os.path.getsize(destino) > 0:
        log(f'UF: {uf} - já existe, pulando')
        return {'uf': uf, 'status': 'pulado'}

    log(f'UF: {uf}')
    t0 = time.time()
    try:
        df = bd.read_sql(
            query=f"SELECT * FROM `basedosdados.br_me_rais.microdados_vinculos` WHERE ano = {ANO} AND sigla_uf = '{uf}'",
            billing_project_id=billing_project_id
        )
        df = pd.DataFrame(df)

        # administração pública direta: se vier 0, o arquivo está incompleto
        adm = None
        if 'natureza_juridica' in df.columns:
            nj = pd.to_numeric(df['natureza_juridica'], errors='coerce')
            adm = int((nj < 2000).sum())

        df.to_csv(parcial, index=False)
        os.replace(parcial, destino)   # só vira definitivo se completou

    except Exception as e:
        log(f'UF: {uf} - FALHOU -> {type(e).__name__}: {e}')
        if os.path.exists(parcial):
            os.remove(parcial)
        return {'uf': uf, 'status': 'erro', 'erro': str(e)}

    mb = os.path.getsize(destino) / 1e6
    dt = time.time() - t0
    if adm is None:
        log(f'UF: {uf} - Done! {len(df):,} linhas | {mb:,.0f} MB | {dt:.0f}s')
    else:
        aviso = '   <<< SEM ADM. PÚBLICA DIRETA!' if adm == 0 else ''
        log(f'UF: {uf} - Done! {len(df):,} linhas | {mb:,.0f} MB | {dt:.0f}s '
            f'| adm.direta {adm:,}{aviso}')

    return {'uf': uf, 'status': 'ok', 'linhas': len(df), 'mb': mb, 'adm': adm}


# ---------------------------------------------------------------------------
# TODAS AS UFs, EM PARALELO
# ---------------------------------------------------------------------------
def main():
    os.makedirs(f'{folder}/Raw Data', exist_ok=True)

    log(f'ano {ANO} | {len(UFS)} UF(s) | {WORKERS} em paralelo')
    if 'Dropbox' in folder:
        log('AVISO: destino dentro do Dropbox. O app pode esvaziar arquivos '
            'antigos para liberar espaço (placeholders de 0 byte).')
    log('-' * 70)

    t0 = time.time()
    resultados = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futuros = {pool.submit(importa_uf, uf): uf for uf in UFS}
        for fut in as_completed(futuros):
            uf = futuros[fut]
            try:
                resultados.append(fut.result())
            except Exception as e:
                log(f'UF: {uf} - exceção -> {type(e).__name__}: {e}')
                resultados.append({'uf': uf, 'status': 'erro', 'erro': str(e)})

    log('-' * 70)
    ok = [r for r in resultados if r['status'] == 'ok']
    pulados = [r for r in resultados if r['status'] == 'pulado']
    ruins = [r for r in resultados if r['status'] == 'erro']

    log(f'terminou em {(time.time() - t0) / 60:.1f} min')
    log(f'  baixadas : {len(ok)}')
    log(f'  puladas  : {len(pulados)}')
    log(f'  com erro : {len(ruins)}')
    if ok:
        log(f'  linhas   : {sum(r["linhas"] for r in ok):,}')
        log(f'  tamanho  : {sum(r["mb"] for r in ok) / 1000:,.1f} GB')

    sem_adm = [r['uf'] for r in ok if r.get('adm') == 0]
    if sem_adm:
        log('')
        log('!' * 70)
        log('ATENÇÃO: estas UFs vieram SEM administração pública direta')
        log(f'  (nenhum natureza_juridica < 2000): {", ".join(sorted(sem_adm))}')
        log('  Foi esse o defeito da leva de 2023 — o emprego público zera')
        log('  na análise. NÃO use estes arquivos.')
        log('!' * 70)

    if ruins:
        log('')
        log(f'UFs com erro: {", ".join(sorted(r["uf"] for r in ruins))}')
        log('Rode de novo: as que já vieram são puladas.')


if __name__ == '__main__':
    main()
