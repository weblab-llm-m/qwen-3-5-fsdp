#!/bin/bash
#SBATCH --job-name=merge-dpo
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

echo "=== Qwen3.5 DPO Merge ==="
echo "開始時刻: $(date)"
echo "ノード: $(hostname)"

mkdir -p outputs/logs

eval "$(conda shell.bash hook)"
set +u
conda activate qwen35-sft
set -u

# スクリプト自身のディレクトリ基準で参照 (cwd 非依存)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# sbatch はスクリプトをスプールへコピーして実行するため、BASH_SOURCE が
# リポジトリ外を指す場合は提出ディレクトリ基準で探索(リポジトリルート or merge/ から提出)
if [[ ! -f "${SCRIPT_DIR}/merge_full_model.py" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  for _cand in "${SLURM_SUBMIT_DIR}/merge" "${SLURM_SUBMIT_DIR}"; do
    if [[ -f "${_cand}/merge_full_model.py" ]]; then SCRIPT_DIR="${_cand}"; break; fi
  done
fi
if [[ ! -f "${SCRIPT_DIR}/merge_full_model.py" ]]; then
  echo "ERROR: merge_full_model.py が見つかりません (SCRIPT_DIR=${SCRIPT_DIR})。リポジトリルートから sbatch してください" >&2
  exit 1
fi

python "${SCRIPT_DIR}/merge_full_model.py" \
  --base_model /path/to/base_model \
  --checkpoint_dir ./output/qwen35-lastlayer-attn-dpo \
  --out_dir ./output/Qwen3.5-397B-A17B-lastlayer-attn-dpo-merged \
  --num_layers 60 \
  --output_dtype keep

echo "=== 完了: $(date) ==="
