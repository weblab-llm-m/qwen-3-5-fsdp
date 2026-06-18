#!/bin/bash
# =============================================================================
# Qwen3.5-397B-A17B — vLLM サーバ環境セットアップ
#
# vLLM + TRL (WeightSyncWorkerExtension) のみ。
# transformers は vLLM が要求するバージョンをそのまま使う。
# トレーナー側とは別 conda 環境にすることで依存衝突を回避。
#
# 使い方:
#   bash setup_vllm_env.sh
#
# 前提:
#   - conda が使える
#   - CUDA 12.x ドライバがインストール済み
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="qwen35-vllm-serve"

echo "============================================================"
echo "  Qwen3.5 vLLM サーバ環境セットアップ"
echo "  env: ${ENV_NAME}"
echo "============================================================"
echo ""

# ============================================================
# 0) 既存環境の削除
# ============================================================
if conda env list 2>/dev/null | grep -q "${ENV_NAME}"; then
    echo "[0/5] 既存の conda 環境 '${ENV_NAME}' を削除中..."
    conda deactivate 2>/dev/null || true
    conda env remove -n "${ENV_NAME}" -y
    echo "  ✅ 削除完了"
else
    echo "[0/5] 既存環境なし — スキップ"
fi
echo ""

# ============================================================
# 1) conda 環境の作成
# ============================================================
echo "[1/5] conda 環境を作成: ${ENV_NAME} (Python 3.11)"
conda create -n "${ENV_NAME}" python=3.11 -y

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "  Python: $(python --version)"
echo "  Prefix: ${CONDA_PREFIX}"
echo ""

# ============================================================
# 2) vLLM nightly
#    Qwen3.5 (2026-02-16) 対応には nightly が必要
#    vLLM が自前の PyTorch + transformers をインストールする
# ============================================================
echo "[2/5] vLLM インストール (ソースから)..."
echo "  ⚠️  stable 0.15.1 は Qwen3.5 未対応 + v1 engine が multi-node TP に弱い"
echo "  nightly wheel は Python/CUDA 不一致で失敗しやすいため、git から直接インストール"
echo "  ※ ビルドに 10-20 分かかります"

pip install -U "git+https://github.com/vllm-project/vllm.git" \
    2>&1 || {
        echo ""
        echo "  ⚠️  git ビルド失敗 — nightly wheel にフォールバック"
        pip install -U vllm --pre \
            --extra-index-url https://wheels.vllm.ai/nightly \
            2>&1 || {
                echo "  ❌ vLLM のインストールに失敗"
                echo "  手動で: pip install git+https://github.com/vllm-project/vllm.git"
                exit 1
            }
    }

VLLM_CHECK=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "FAILED")
if [ "${VLLM_CHECK}" = "0.15.1" ]; then
    echo "  ❌ vLLM 0.15.1 (stable) がインストールされています"
    echo "     Qwen3.5 未対応。手動で git ソースからインストールしてください:"
    echo "     pip install -U git+https://github.com/vllm-project/vllm.git"
    exit 1
fi

VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "FAILED")
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "FAILED")
TF_VER=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "FAILED")
echo "  ✅ vLLM:         ${VLLM_VER}"
echo "  ✅ PyTorch:      ${TORCH_VER}"
echo "  ✅ transformers: ${TF_VER} (vLLM が管理)"
echo ""

# ============================================================
# 3) TRL (WeightSyncWorkerExtension 用)
#    vllm_trl_serve.py から trl.scripts.vllm_serve を呼ぶために必要
# ============================================================
echo "[3/5] TRL インストール (vLLM serve 用)..."
pip install "trl>=0.28.0" fastapi uvicorn pydantic requests

TRL_VER=$(python -c "import trl; print(trl.__version__)")
echo "  ✅ TRL: ${TRL_VER}"
echo ""

# ============================================================
# 4) vllm_trl_serve.py の配置確認
#    vllm serve には TRL の HTTP ルートがないため
#    カスタムサーバスクリプトが必要
# ============================================================
echo "[4/5] vllm_trl_serve.py の確認..."
SERVE_SCRIPT="${SCRIPT_DIR}/vllm_trl_serve.py"
if [ -f "${SERVE_SCRIPT}" ]; then
    echo "  ✅ ${SERVE_SCRIPT} (既存)"
else
    echo "  ❌ vllm_trl_serve.py が見つかりません"
    echo "     SLURM スクリプトと同じディレクトリに配置してください"
fi
echo ""

# ============================================================
# 5) TRL パッチ + 検証
# ============================================================
echo "[5/5] TRL/vLLM 互換パッチ + 検証..."

# vllm_ascend stub (TRL が Ascend NPU コードを import する問題の回避)
python3 << 'PATCH_VLLM_ASCEND'
import os, site
try:
    sp = site.getsitepackages()[0]
except Exception:
    import sys
    sp = os.path.join(os.path.dirname(sys.executable), "..", "lib",
                      f"python{sys.version_info.major}.{sys.version_info.minor}",
                      "site-packages")
pkg = os.path.join(sp, "vllm_ascend", "distributed", "device_communicators")
os.makedirs(pkg, exist_ok=True)
for d in [os.path.join(sp, "vllm_ascend"),
          os.path.join(sp, "vllm_ascend", "distributed"),
          pkg]:
    init = os.path.join(d, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()
pyhccl = os.path.join(pkg, "pyhccl.py")
if not os.path.exists(pyhccl):
    with open(pyhccl, "w") as f:
        f.write("class PyHcclCommunicator: pass\n")
    print("  ✅ vllm_ascend stub 作成")
else:
    print("  ✅ vllm_ascend stub 既存")
PATCH_VLLM_ASCEND

# WeightSyncWorkerExtension import 確認
python3 -c "
from trl.scripts.vllm_serve import WeightSyncWorkerExtension
print('  ✅ WeightSyncWorkerExtension: OK')
" 2>&1 || echo "  ❌ WeightSyncWorkerExtension import 失敗"

# 総合検証
python3 -c "
import sys
print()
print(f'Python:       {sys.version.split()[0]}')

import torch
print(f'PyTorch:      {torch.__version__}')
print(f'  CUDA:       {torch.version.cuda}')
print(f'  GPU count:  {torch.cuda.device_count()}')

import transformers
print(f'transformers: {transformers.__version__} (vLLM管理 — 上書き不要)')

import vllm
print(f'vLLM:         {vllm.__version__}')

import trl
print(f'TRL:          {trl.__version__}')

# NCCL utilities
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
print(f'NCCL utils:   ✅ PyNcclCommunicator + StatelessProcessGroup')
print()
" 2>&1

echo ""
echo "============================================================"
echo "  vLLM サーバ環境セットアップ完了"
echo "============================================================"
echo ""
echo "  環境名: ${ENV_NAME}"
echo "  用途:   vLLM サーバ (node-XX, node-XX)"
echo ""
echo "  ⚠️  transformers はvLLMが管理するバージョンのまま。"
echo "     上書きしないでください。"
echo ""
echo "  起動方法 (SLURM スクリプトが自動実行):"
echo "    conda activate ${ENV_NAME}"
echo "    python vllm_trl_serve.py \\"
echo "      --model /path/to/Qwen3.5-397B-A17B \\"
echo "      --tensor-parallel-size 16 \\"
echo "      --distributed-executor-backend ray \\"
echo "      --language-model-only \\"
echo "      --port 8000"
echo "============================================================"
