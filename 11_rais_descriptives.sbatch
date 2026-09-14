#!/bin/bash
# ============================================================================
# 11_rais_descriptives.sbatch
#
# Roda 11_rais_descriptives.py com RODADA = "BR" (27 UFs).
#
# ETAPA CARA: a primeira execução lê ~18 GB de CSV da RAIS e grava o painel
# enxuto em Cleaned Data/*.parquet. Depois disso o parquet é reaproveitado e o
# script roda em segundos — então este job só precisa ser submetido de novo se
# as restrições da amostra mudarem (ou com --rebuild).
#
# COMO SUBMETER
#   cd ".../CONCURSOS/Data/PNADC/Code"
#   sbatch 11_rais_descriptives.sbatch                # usa o parquet se existir
#   sbatch 11_rais_descriptives.sbatch --rebuild      # força reconstruir
#
# ACOMPANHAR
#   squeue -u $USER
#   tail -f logs/rais_descr_<jobid>.out
#
# ANTES DE SUBMETER, CONFIRA
#   1. RODADA = "BR" em 11_rais_descriptives.py (linha ~62).
#   2. Os dados estão no cluster, na MESMA estrutura de pastas:
#        <raiz>/PNADC/Code/11_rais_descriptives.py
#        <raiz>/RAIS_Workers/Raw Data/microdados_vinculos_2023_*.csv
#        <raiz>/RAIS_Workers/Raw Data/regioes_geograficas_composicao_*.csv
#        <raiz>/RAIS_Workers/Raw Data/br_geobr_mapas_regiao_imediata.csv
#      Se a raiz no cluster for outra, descomente e ajuste RAIS_PROJECT_ROOT
#      mais abaixo (ele deve apontar para a pasta PNADC).
#   3. O módulo/ambiente de Python da seção "AMBIENTE" confere com o cluster.
# ============================================================================

#SBATCH --job-name=rais_descr
#SBATCH --output=logs/rais_descr_%j.out
#SBATCH --error=logs/rais_descr_%j.err
#SBATCH --partition=shared           # FASRC: shared | sapphire | serial_requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # pandas é 1 thread; as extras servem ao parquet
#SBATCH --mem=32GB                    # leitura em chunks: o pico não é o CSV inteiro
#SBATCH --time=0-01:00:00            # ~10-40 min esperados; folga para o CSV de SP
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pedro_ferreira@hks.harvard.edu

set -euo pipefail

mkdir -p logs

echo "=============================================="
echo "job      : ${SLURM_JOB_ID:-local}"
echo "nó       : $(hostname)"
echo "início   : $(date)"
echo "cpus     : ${SLURM_CPUS_PER_TASK:-?}   mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "=============================================="

# ---------------------------------------------------------------------------
# AMBIENTE — ajuste para o seu cluster. Deixe UMA das opções ativa.
# ---------------------------------------------------------------------------
# Opção A) módulo do sistema + venv (FASRC)
module load python
mamba activate "/n/home04/pferreira/mamba_env"
# source "$HOME/envs/rais/bin/activate"

# Opção B) conda / mamba
# module load Mambaforge 2>/dev/null || true
# source activate rais

# Precisa de: pandas, numpy, pyarrow, matplotlib
python -c "import pandas, numpy, pyarrow, matplotlib; \
print('pandas', pandas.__version__, '| numpy', numpy.__version__, \
'| pyarrow', pyarrow.__version__, '| matplotlib', matplotlib.__version__)"

# sem display no nó de cálculo: matplotlib tem de usar o backend de arquivo
export MPLBACKEND=Agg
# evita que numpy/BLAS abram mais threads do que as CPUs pedidas
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Se a árvore de pastas no cluster tiver outra raiz, aponte para a pasta PNADC:
# export RAIS_PROJECT_ROOT="/n/holyscratch01/$USER/CONCURSOS/Data/PNADC"

# ---------------------------------------------------------------------------
# RODA — submeta de dentro da pasta Code/, que é onde o .py mora
# ---------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [[ ! -f 11_rais_descriptives.py ]]; then
    echo "ERRO: 11_rais_descriptives.py não está em $(pwd)."
    echo "      Submeta de dentro da pasta Code/ (cd .../PNADC/Code)."
    exit 1
fi

echo "diretório: $(pwd)"
echo

# "$@" repassa argumentos do sbatch para o script (ex.: --rebuild)
python -u 11_rais_descriptives.py "$@"

echo
echo "=============================================="
echo "fim      : $(date)"
echo "=============================================="
