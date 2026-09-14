#!/bin/bash
# ============================================================================
# 11a_rais_import_parallel.sbatch
#
# Baixa a RAIS do BigQuery, 27 UFs em paralelo, sem navegador.
#
# ANTES DE SUBMETER — faça isto num nó de LOGIN, uma vez só:
#   1. A chave de service account tem de existir e estar fechada:
#        ls -l  $GOOGLE_APPLICATION_CREDENTIALS
#        chmod 600 $GOOGLE_APPLICATION_CREDENTIALS
#   2. Teste a credencial e veja o custo SEM baixar nada:
#        python 11a_rais_import_parallel.py --ufs AC --dry-run
#      Se isso falhar, o job também falha — resolva aqui primeiro.
#   3. Confirme que a origem já tem administração pública (o defeito de 2023):
#        python 11a_rais_import_parallel.py --ufs AC --out /tmp/testerais
#        # a linha do AC tem de mostrar "adm.direta" com ~90.000
#
# SUBMETER
#   cd .../PNADC/Code
#   mkdir -p logs                 # o SLURM abre os logs ANTES do script rodar
#   jid=$(sbatch --parsable 11a_rais_import_parallel.sbatch)
#   tail -f --retry logs/rais_import_${jid}.out
# ============================================================================

#SBATCH --job-name=rais_import
#SBATCH --output=logs/rais_import_%j.out
#SBATCH --error=logs/rais_import_%j.err
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # 1 por thread de download (--workers 6) + folga
#SBATCH --mem=32G                 # grava em pedaços; o pico é por thread
#SBATCH --time=0-08:00:00         # rede é imprevisível; UFs prontas são puladas
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro_ferreira@hks.harvard.edu

set -euo pipefail

mkdir -p logs

echo "=============================================="
echo "job    : ${SLURM_JOB_ID:-local}"
echo "nó     : $(hostname)"
echo "início : $(date)"
echo "=============================================="

# ---------------------------------------------------------------------------
# AMBIENTE — ajuste ao cluster (mesma seção do 11_rais_descriptives.sbatch)
# ---------------------------------------------------------------------------
module load python 2>/dev/null || true
# source "$HOME/envs/rais/bin/activate"

# ---------------------------------------------------------------------------
# CREDENCIAL E DESTINO — o script aborta com mensagem clara se faltar algo
# ---------------------------------------------------------------------------
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcp/bq_key.json}"
export BQ_BILLING_PROJECT="${BQ_BILLING_PROJECT:-educacaoideias}"

# scratch, NUNCA o home (quota) nem o Dropbox (sync). Ajuste o caminho do lab.
export RAIS_OUT_DIR="${RAIS_OUT_DIR:-/n/netscratch/kreindler_lab/Lab/pferreira/RAIS/Raw}"

ANO="${ANO:-2023}"

echo "credencial : $GOOGLE_APPLICATION_CREDENTIALS"
echo "billing    : $BQ_BILLING_PROJECT"
echo "saída      : $RAIS_OUT_DIR"
echo "ano        : $ANO"
echo

if [[ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
    echo "ERRO: chave de service account não encontrada."
    echo "      Veja PRÉ-REQUISITOS no cabeçalho de 11a_rais_import_parallel.py."
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ ! -f 11a_rais_import_parallel.py ]]; then
    echo "ERRO: 11a_rais_import_parallel.py não está em $(pwd)."
    echo "      Submeta de dentro da pasta Code/."
    exit 1
fi

# "$@" repassa argumentos (ex.: --force, --workers 8, --all-columns)
python -u 11a_rais_import_parallel.py --year "$ANO" --workers 6 "$@"

echo
echo "=============================================="
echo "fim    : $(date)"
echo "=============================================="
