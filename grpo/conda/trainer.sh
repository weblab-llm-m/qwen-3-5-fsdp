#!/bin/bash
# =============================================================================
# Qwen3.5-397B-A17B — トレーナー環境セットアップ
#
# FSDP + transformers >= 5.1.0 (Qwen3.5 対応) の学習環境。
# vLLM は NCCL weight sync 用ユーティリティとしてのみ使用。
# (モデルロードには使わないので transformers バージョン不一致は問題なし)
#
# 使い方:
#   bash setup_trainer_env.sh
#
# 前提:
#   - conda が使える
#   - CUDA 12.x ドライバがインストール済み
#   - fsdp_vllm_client.py, run_qwen35_expert_grpo.py が親ディレクトリ (grpo/) にあること
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# コピー元は grpo/ (このスクリプトは grpo/conda/ にある)
GRPO_SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="qwen35-trainer"
GRPO_DIR="${HOME}/qwen35_grpo"

echo "============================================================"
echo "  Qwen3.5 トレーナー環境セットアップ"
echo "  env:      ${ENV_NAME}"
echo "  grpo_dir: ${GRPO_DIR}"
echo "============================================================"
echo ""

# ファイル存在チェック
MISSING=false
for f in fsdp_vllm_client.py run_qwen35_expert_grpo.py run_qwen35_expert_grpo.sh; do
    if [ ! -f "${GRPO_SRC_DIR}/${f}" ]; then
        echo "  ⚠️  ${f} が ${GRPO_SRC_DIR} にありません"
        MISSING=true
    fi
done
if [ "${MISSING}" = true ]; then
    echo ""
    echo "  ファイル配置をスキップして環境構築のみ続行しますか? (y/N)"
    read -r REPLY
    if [[ ! "${REPLY}" =~ ^[Yy]$ ]]; then
        echo "中止します"
        exit 1
    fi
fi

# ============================================================
# 0) 既存環境の削除
# ============================================================
if conda env list 2>/dev/null | grep -q "${ENV_NAME}"; then
    echo "[0/7] 既存の conda 環境 '${ENV_NAME}' を削除中..."
    conda deactivate 2>/dev/null || true
    conda env remove -n "${ENV_NAME}" -y
    echo "  ✅ 削除完了"
else
    echo "[0/7] 既存環境なし — スキップ"
fi
echo ""

# ============================================================
# 1) conda 環境の作成
# ============================================================
echo "[1/7] conda 環境を作成: ${ENV_NAME} (Python 3.11)"
conda create -n "${ENV_NAME}" python=3.11 -y

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "  Python: $(python --version)"
echo "  Prefix: ${CONDA_PREFIX}"
echo ""

# ============================================================
# 2) PyTorch (CUDA 12.4)
# ============================================================
echo "[2/7] PyTorch インストール (CUDA 12.4)..."
pip install torch --index-url https://download.pytorch.org/whl/cu124

TORCH_VER=$(python -c "import torch; print(torch.__version__)")
echo "  ✅ PyTorch: ${TORCH_VER}"
echo ""

# ============================================================
# 3) 学習用パッケージ (TRL なし — トレーナーでは不要)
# ============================================================
echo "[3/7] 学習用パッケージをインストール..."
pip install \
    accelerate \
    sentencepiece \
    protobuf \
    wandb \
    requests \
    pillow \
    packaging

echo "  ✅ 学習パッケージ完了"
echo ""

# ============================================================
# 4) vLLM nightly (NCCL weight sync ユーティリティ用のみ)
#
#    PyNcclCommunicator / StatelessProcessGroup を使うために
#    vLLM パッケージが必要。モデルロードには使わないので
#    transformers バージョン不一致は問題なし。
#
#    ⚠️  vLLM が PyTorch を上書きする場合あり (cu128 等)
#       CUDA 12.x ドライバなら互換性あり
# ============================================================
echo "[4/7] vLLM nightly インストール (NCCL ユーティリティ用)..."
echo "  ℹ️  モデルロードには使用しません (transformers 不一致 OK)"

# --pre が必須: ないと PyPI の stable (0.15.1) にフォールバックする
pip install -U vllm --pre \
    --extra-index-url https://wheels.vllm.ai/nightly \
    2>&1

VLLM_CHECK=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "FAILED")
if [ "${VLLM_CHECK}" = "0.15.1" ] || [ "${VLLM_CHECK}" = "FAILED" ]; then
    echo ""
    echo "  ⚠️  nightly が入らなかった (${VLLM_CHECK}) — git から直接インストール"
    pip install -U "git+https://github.com/vllm-project/vllm.git" \
        2>&1 || {
            echo ""
            echo "  ⚠️  vLLM インストールに失敗"
            echo "  NCCL weight sync が使えません。"
        }
fi

VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "not installed")
echo "  vLLM: ${VLLM_VER}"
echo ""

# ============================================================
# 5) transformers >= 5.1.0 をソースからインストール
#
#    vLLM が古い transformers (4.x) をインストールしているが、
#    Qwen3.5 = qwen3_5_moe アーキテクチャには 5.1.0+ が必須。
#    --no-deps で上書きし、他のパッケージのダウングレードを防止。
#
#    vLLM の NCCL ユーティリティ (PyNcclCommunicator 等) は
#    transformers に依存しないため、上書きしても問題なし。
# ============================================================
echo "[5/7] transformers + huggingface_hub をソースインストール..."
echo "  Qwen3.5 = qwen3_5_moe (transformers >= 5.1.0 必須)"

OLD_TF=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "none")
echo "  vLLM インストール後の transformers: ${OLD_TF}"

# huggingface_hub + tokenizers を最新に更新 (transformers 5.x の依存)
pip install --no-cache-dir -U huggingface_hub tokenizers

# transformers をソースからインストール (--no-deps で PyTorch 等のダウングレード防止)
pip install --no-cache-dir --no-deps \
    git+https://github.com/huggingface/transformers.git

NEW_TF=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "FAILED")
NEW_HH=$(python -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>/dev/null || echo "FAILED")
echo "  ✅ transformers:   ${NEW_TF}"
echo "  ✅ huggingface_hub: ${NEW_HH}"

# バージョン検証
python -c "
from packaging.version import Version
import transformers
v = transformers.__version__.split('+')[0].split('.dev')[0]
if Version(v) < Version('5.1.0'):
    print(f'  ❌ transformers {v} < 5.1.0 — Qwen3.5 未対応!')
    exit(1)
else:
    print(f'  ✅ transformers {v} >= 5.1.0 (Qwen3.5 対応)')
" || {
    echo "  ❌ transformers のバージョン確認に失敗"
    echo "  手動修正: pip install -U huggingface_hub && pip install --no-deps git+https://github.com/huggingface/transformers.git"
}
echo ""

# ============================================================
# 6) ファイル配置
# ============================================================
echo "[6/7] GRPO スクリプトを配置: ${GRPO_DIR}"
mkdir -p "${GRPO_DIR}"

FILES_TO_COPY=(
    "fsdp_vllm_client.py"
    "run_qwen35_expert_grpo.py"
    "run_qwen35_expert_grpo.sh"
)

for f in "${FILES_TO_COPY[@]}"; do
    if [ -f "${GRPO_SRC_DIR}/${f}" ]; then
        cp "${GRPO_SRC_DIR}/${f}" "${GRPO_DIR}/${f}"
        echo "  ✅ ${f} → ${GRPO_DIR}/"
    else
        echo "  ⚠️  ${f} が見つかりません (スキップ)"
    fi
done
echo ""

# ============================================================
# 7) 総合検証
# ============================================================
echo "[7/7] 総合検証..."
echo ""

python -c "
import sys
print(f'Python: {sys.version.split()[0]}')
print()

# --- PyTorch ---
import torch
print(f'PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:   {torch.version.cuda}')
    print(f'  GPU count:      {torch.cuda.device_count()}')
    for i in range(min(torch.cuda.device_count(), 4)):
        props = torch.cuda.get_device_properties(i)
        print(f'    GPU {i}: {props.name} ({props.total_mem / 1e9:.1f} GB)')
print()

# --- Transformers ---
import transformers
ver = transformers.__version__
print(f'transformers: {ver}')
try:
    from packaging.version import Version
    v = ver.split('+')[0].split('.dev')[0]
    if Version(v) >= Version('5.1.0'):
        print(f'  ✅ Qwen3.5 対応')
    else:
        print(f'  ❌ < 5.1.0 — Qwen3.5 未対応!')
except Exception:
    pass
print()

# --- huggingface_hub ---
import huggingface_hub
print(f'huggingface_hub: {huggingface_hub.__version__}')
if hasattr(huggingface_hub, 'is_offline_mode'):
    print(f'  ✅ is_offline_mode: OK')
else:
    print(f'  ❌ is_offline_mode: NOT FOUND (古いバージョン)')
print()

# --- Accelerate ---
import accelerate
print(f'Accelerate: {accelerate.__version__}')
print()

# --- FSDP ---
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
    print(f'FSDP: ✅ OK')
except ImportError:
    print(f'FSDP: ❌ NOT AVAILABLE')
print()

# --- vLLM (NCCL utilities only) ---
try:
    import vllm
    print(f'vLLM: {vllm.__version__} (NCCL ユーティリティ用)')
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    print(f'  ✅ PyNcclCommunicator: OK')
    from vllm.distributed.utils import StatelessProcessGroup
    print(f'  ✅ StatelessProcessGroup: OK')
except ImportError as e:
    print(f'vLLM: ⚠️  {e}')
    print(f'  NCCL weight sync が使えません')
print()

# --- fsdp_vllm_client ---
import os
sys.path.insert(0, '${GRPO_DIR}')
try:
    from fsdp_vllm_client import FSDPVLLMClient
    print(f'fsdp_vllm_client: ✅ OK')
except ImportError as e:
    print(f'fsdp_vllm_client: ⚠️  {e}')
print()

# --- Qwen3.5 モデル ---
paths = [
    '/path/to/base_model',
    '/path/to/base_model',
]
for p in paths:
    if os.path.exists(p):
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(p, trust_remote_code=True)
            tc = getattr(cfg, 'text_config', cfg)
            mt = getattr(cfg, 'model_type', '?')
            nl = getattr(tc, 'num_hidden_layers', '?')
            ne = getattr(tc, 'num_experts', '?')
            print(f'Qwen3.5: ✅ {mt} (layers={nl}, experts={ne})')
            print(f'  path: {p}')
        except Exception as e:
            print(f'Qwen3.5: ❌ {e}')
        break
else:
    print(f'Qwen3.5: ⚠️  ローカルパスなし (スキップ)')
print()

# --- wandb ---
try:
    import wandb
    print(f'wandb: {wandb.__version__}')
except ImportError:
    print(f'wandb: NOT INSTALLED')
" 2>&1

echo ""
echo "============================================================"
echo "  トレーナー環境セットアップ完了"
echo "============================================================"
echo ""
echo "  環境名: ${ENV_NAME}"
echo "  用途:   FSDP トレーナー (node-XX〜node-XX)"
echo ""
echo "  ⚠️  注意事項:"
echo "    - transformers は 5.x (ソース) に上書き済み"
echo "    - vLLM は NCCL ユーティリティ用のみ (モデルロードには不使用)"
echo "    - TRL は不要 (fsdp_vllm_client.py で完全置換)"
echo ""
echo "  学習実行:"
echo "    conda activate ${ENV_NAME}"
echo "    cd ${GRPO_DIR}"
echo "    sbatch run_qwen35_expert_grpo.sh"
echo "============================================================"
