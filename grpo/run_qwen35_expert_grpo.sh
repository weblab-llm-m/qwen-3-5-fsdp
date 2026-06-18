#!/bin/bash
#SBATCH --job-name=qwen35-expert-grpo
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodelist=node-XX,node-XX,node-XX,node-XX,node-XX,node-XX,node-XX,node-XX
#SBATCH --nodes=8
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=240
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err
#SBATCH --exclusive

# ============================================================
# Qwen3.5-397B-A17B Expert-only GRPO (分離環境版)
#
# Node layout (8 nodes, 64 GPUs):
#   node-XX,node-XX : vLLM server (TP=16, ray cluster, 16 GPU)
#                  conda env: qwen35-vllm-serve
#   node-XX : Trainer    (FSDP FULL_SHARD, 48 GPU)
#                  conda env: qwen35-trainer
#
# 2つの conda 環境を使い分けることで依存衝突を回避:
#   vLLM側: vLLM nightly + vLLM が要求する transformers (4.x)
#   Trainer側: transformers >= 5.1.0 (Qwen3.5) + vLLM (NCCL only)
#
# IMPORTANT: vLLM server runs a warmup generate on startup to trigger
# FlashInfer GDN JIT compilation BEFORE accepting connections. This
# takes 40-60+ minutes on first run. The health check loop waits for
# this to complete. Subsequent runs use cached kernels and are fast.
# ============================================================

set -euo pipefail

echo "================================================================"
echo "  Qwen3.5-397B-A17B Expert GRPO (分離環境版)"
echo "  vLLM 2 nodes (TP=16) + Trainer 6 nodes (48 GPU)"
echo "  開始: $(date)"
echo "================================================================"

mkdir -p outputs/logs

# ============================================================
# ネットワーク設定
# ============================================================
export TARGET_IFNAME="eth0"

module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load cuda/12.4

export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="${TARGET_IFNAME}"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export GLOO_SOCKET_IFNAME="${TARGET_IFNAME}"
export TP_SOCKET_IFNAME="${TARGET_IFNAME}"

export OMP_NUM_THREADS=2
ulimit -n 262144 || true

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200

# FlashInfer GDN カーネル JIT キャッシュ永続化
# 初回 40-60分 → 2回目以降キャッシュヒットでスキップ
export FLASHINFER_CACHE_DIR="${HOME}/.cache/flashinfer"
mkdir -p "${FLASHINFER_CACHE_DIR}"

# Ray Compiled DAG タイムアウト延長 (GDN JIT 初回コンパイル対策)
# 2時間に設定: JIT + ウォームアップで40-60分かかる可能性がある
export RAY_CGRAPH_get_timeout=7200
export RAY_CGRAPH_teardown_timeout=120

# ============================================================
# conda 環境 (分離)
# ============================================================
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
CONDA_ENV_VLLM="${CONDA_BASE}/envs/qwen35-vllm-serve"
CONDA_ENV_TRAINER="${CONDA_BASE}/envs/qwen35-trainer"

echo "vLLM env   : ${CONDA_ENV_VLLM}"
echo "Trainer env: ${CONDA_ENV_TRAINER}"

# ============================================================
# ノード割り当て
# ============================================================
ALL_NODES=($(scontrol show hostnames "${SLURM_NODELIST}"))
VLLM_NODE_COUNT=2
TRAINER_NODE_COUNT=6

VLLM_NODES=("${ALL_NODES[@]:0:${VLLM_NODE_COUNT}}")
TRAINER_NODES=("${ALL_NODES[@]:${VLLM_NODE_COUNT}}")
TRAINER_NNODES=${#TRAINER_NODES[@]}

VLLM_HEAD="${VLLM_NODES[0]}"

echo "vLLM nodes   : ${VLLM_NODES[*]}"
echo "Trainer nodes: ${TRAINER_NODES[*]} (${TRAINER_NNODES} nodes, $((TRAINER_NNODES * 8)) GPUs)"

# ============================================================
# IP アドレス取得
# ============================================================
get_node_ip() {
    ssh -o StrictHostKeyChecking=no "$1" \
        "ip -4 addr show ${TARGET_IFNAME} 2>/dev/null \
         | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1"
}

VLLM_HEAD_IP=$(get_node_ip "${VLLM_HEAD}")
TRAINER_MASTER_IP=$(get_node_ip "${TRAINER_NODES[0]}")

# vLLM 全ノードの IP を取得 (Ray --node-ip-address 用)
declare -a VLLM_NODE_IPS
for NODE in "${VLLM_NODES[@]}"; do
    NODE_IP=$(get_node_ip "${NODE}")
    VLLM_NODE_IPS+=("${NODE_IP}")
    echo "  ${NODE} -> ${NODE_IP}"
done

echo "vLLM head IP     : ${VLLM_HEAD_IP}"
echo "Trainer master IP: ${TRAINER_MASTER_IP}"

if [ -z "${VLLM_HEAD_IP}" ] || [ -z "${TRAINER_MASTER_IP}" ]; then
    echo "ERROR: IP 取得失敗"
    exit 1
fi

# ============================================================
# 設定値
# ============================================================
BASE_MODEL="/path/to/base_model"
DATASET="/path/to/train.jsonl"
OUTPUT_DIR="./output/qwen35-expert-grpo-$(date +%Y%m%d-%H%M)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# sbatch はスクリプトをスプールへコピーして実行するため、BASH_SOURCE が
# リポジトリ外を指す場合は提出ディレクトリ基準で探索(リポジトリルート or grpo/ から提出)
if [[ ! -f "${SCRIPT_DIR}/run_qwen35_expert_grpo.py" && -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  for _cand in "${SLURM_SUBMIT_DIR}/grpo" "${SLURM_SUBMIT_DIR}"; do
    if [[ -f "${_cand}/run_qwen35_expert_grpo.py" ]]; then SCRIPT_DIR="${_cand}"; break; fi
  done
fi
if [[ ! -f "${SCRIPT_DIR}/run_qwen35_expert_grpo.py" ]]; then
  echo "ERROR: run_qwen35_expert_grpo.py が見つかりません (SCRIPT_DIR=${SCRIPT_DIR})。リポジトリルートから sbatch してください" >&2
  exit 1
fi

SCRIPT="${SCRIPT_DIR}/run_qwen35_expert_grpo.py"

# Expert mask JSON
MASK_JSON=""

# vLLM 設定
VLLM_PORT=8000
VLLM_TP=16
RAY_PORT=6379
VLLM_GPU_UTIL=0.90
VLLM_MAX_NUM_SEQS=128

# GRPO 設定
GRPO_GROUP_SIZE=4
BATCH_SIZE=2
GEN_MAX_TOKENS=512
GEN_TEMPERATURE=0.7

# Attention training mode
TRAIN_ATTENTION="indices"
ATTN_LAYER_INDICES="59"
TRAIN_LAYERNORM=""

# Judge 設定
JUDGE_BACKEND="openrouter"
JUDGE_MODEL="qwen/qwen3-235b-a22b"
JUDGE_API_BASE="https://openrouter.ai/api"
JUDGE_CONCURRENCY=100

# W&B 設定
WANDB_PROJECT="qwen35-expert-grpo"
WANDB_RUN_NAME="expert-grpo-8node-$(date +%Y%m%d-%H%M)"

# ============================================================
# 環境セットアップスクリプト (vLLM用 / Trainer用)
# ============================================================

# --- vLLM ノード用 ---
ENV_SETUP_VLLM="${SCRIPT_DIR}/.env_setup_vllm_${SLURM_JOB_ID}.sh"
cat > "${ENV_SETUP_VLLM}" << ENVEOF
#!/bin/bash
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV_VLLM}

for init in /etc/profile.d/modules.sh /usr/share/modules/init/bash; do
    [ -f "\$init" ] && source "\$init" 2>/dev/null && break
done
module load nccl/2.22.3 2>/dev/null || true
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true

export TARGET_IFNAME="${TARGET_IFNAME}"
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="\${TARGET_IFNAME}"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME="\${TARGET_IFNAME}"
export TP_SOCKET_IFNAME="\${TARGET_IFNAME}"
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# FlashInfer JIT キャッシュ永続化 (GDN カーネル初回 40-60分 → 2回目以降スキップ)
export FLASHINFER_CACHE_DIR="${HOME}/.cache/flashinfer"
mkdir -p "\${FLASHINFER_CACHE_DIR}" 2>/dev/null || true

# Ray Compiled DAG タイムアウト延長 (GDN JIT 初回コンパイル対策 — 2時間)
export RAY_CGRAPH_get_timeout=7200
export RAY_CGRAPH_teardown_timeout=120
ENVEOF
chmod +x "${ENV_SETUP_VLLM}"

# --- Trainer ノード用 ---
ENV_SETUP_TRAINER="${SCRIPT_DIR}/.env_setup_trainer_${SLURM_JOB_ID}.sh"
cat > "${ENV_SETUP_TRAINER}" << ENVEOF
#!/bin/bash
eval "\$(conda shell.bash hook)"
conda activate ${CONDA_ENV_TRAINER}

for init in /etc/profile.d/modules.sh /usr/share/modules/init/bash; do
    [ -f "\$init" ] && source "\$init" 2>/dev/null && break
done
module load nccl/2.22.3 2>/dev/null || true
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt 2>/dev/null || true
module load cuda/12.4 2>/dev/null || true

export TARGET_IFNAME="${TARGET_IFNAME}"
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="\${TARGET_IFNAME}"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_SOCKET_IFNAME="\${TARGET_IFNAME}"
export TP_SOCKET_IFNAME="\${TARGET_IFNAME}"
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${SCRIPT_DIR}:\${PYTHONPATH:-}"
ENVEOF
chmod +x "${ENV_SETUP_TRAINER}"

echo "vLLM env script   : ${ENV_SETUP_VLLM}"
echo "Trainer env script: ${ENV_SETUP_TRAINER}"

# ============================================================
# クリーンアップ関数 + trap
# set -euo pipefail のため途中で異常終了しても vLLM/Ray が
# 残留しないよう、EXIT trap で必ずクリーンアップを実行する
# ============================================================
CLEANUP_DONE=0
cleanup() {
    if [ "${CLEANUP_DONE}" -eq 1 ]; then
        return 0
    fi
    CLEANUP_DONE=1
    echo ""
    echo "================================================================"
    echo "  Cleanup (vLLM / Ray / 一時ファイル)"
    echo "================================================================"

    if [ -n "${VLLM_PID:-}" ]; then
        echo "[Cleanup] Stopping vLLM server (PID=${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        sleep 3
    fi

    for NODE in "${VLLM_NODES[@]}"; do
        echo "[Cleanup] Stopping ray on ${NODE}..."
        ssh -o StrictHostKeyChecking=no "${NODE}" \
            "source ${ENV_SETUP_VLLM} 2>/dev/null; ray stop --force 2>/dev/null; true" &
    done
    wait

    # 環境スクリプト削除
    rm -f "${ENV_SETUP_VLLM}" "${ENV_SETUP_TRAINER}"
}
trap cleanup EXIT

# ============================================================
# TRL compatibility patches (vLLM env で実行)
# ============================================================
source "${ENV_SETUP_VLLM}"

echo "[PATCH] vllm_ascend stub (vLLM env)..."
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
    print("[PATCH] Created vllm_ascend stub")
else:
    print("[PATCH] vllm_ascend stub already exists")
PATCH_VLLM_ASCEND

python3 -c "
from trl.scripts.vllm_serve import WeightSyncWorkerExtension
print('[PATCH] ✅ WeightSyncWorkerExtension import OK')
" 2>&1 || {
    echo "[PATCH] ⚠️  WeightSyncWorkerExtension import failed"
}

# ============================================================
# Phase 1: Ray クラスタ起動 (vLLM 2ノード)
# ============================================================
echo ""
echo "================================================================"
echo "  Phase 1: Ray cluster on vLLM nodes (env: qwen35-vllm-serve)"
echo "================================================================"

# 既存 ray を停止
for NODE in "${VLLM_NODES[@]}"; do
    ssh -o StrictHostKeyChecking=no "${NODE}" \
        "source ${ENV_SETUP_VLLM} 2>/dev/null && ray stop --force 2>/dev/null; true" &
done
wait
sleep 2

# Ray head 起動 (--node-ip-address + VLLM_HOST_IP で正しいインターフェースを強制)
echo "[Ray] Starting head on ${VLLM_HEAD} (IP: ${VLLM_HEAD_IP})..."
ssh -o StrictHostKeyChecking=no "${VLLM_HEAD}" "bash -c '
    source ${ENV_SETUP_VLLM}
    export VLLM_HOST_IP=${VLLM_HEAD_IP}
    ray start --head \
        --port=${RAY_PORT} \
        --num-gpus=8 \
        --node-ip-address=${VLLM_HEAD_IP} \
        --dashboard-host=0.0.0.0 \
        --temp-dir=/tmp/ray_vllm_\${USER}
    echo \"[Ray] Head started on \$(hostname) (${VLLM_HEAD_IP}), VLLM_HOST_IP=\${VLLM_HOST_IP}\"
'" 2>&1 | sed 's/^/  [ray-head] /' &
sleep 10

# Ray worker 起動 (--node-ip-address + VLLM_HOST_IP を各ノードのIPで設定)
for idx in $(seq 1 $((${#VLLM_NODES[@]} - 1))); do
    NODE="${VLLM_NODES[$idx]}"
    WORKER_IP="${VLLM_NODE_IPS[$idx]}"
    echo "[Ray] Starting worker on ${NODE} (IP: ${WORKER_IP})..."
    ssh -o StrictHostKeyChecking=no "${NODE}" "bash -c '
        source ${ENV_SETUP_VLLM}
        export VLLM_HOST_IP=${WORKER_IP}
        ray start \
            --address=${VLLM_HEAD_IP}:${RAY_PORT} \
            --num-gpus=8 \
            --node-ip-address=${WORKER_IP} \
            --temp-dir=/tmp/ray_vllm_\${USER}
        echo \"[Ray] Worker joined from \$(hostname) (${WORKER_IP}), VLLM_HOST_IP=\${VLLM_HOST_IP}\"
    '" 2>&1 | sed "s/^/  [ray-${NODE}] /" &
done
sleep 10

# Ray クラスタ確認 (GPU 数とノード IP を表示)
echo "[Ray] Checking cluster status..."
ssh -o StrictHostKeyChecking=no "${VLLM_HEAD}" "bash -c '
    source ${ENV_SETUP_VLLM}
    python3 -c \"
import ray
ray.init(address=\\\"auto\\\")
nodes = ray.nodes()
print(f\\\"Ray nodes: {len(nodes)}\\\")
total_gpus = 0
for n in nodes:
    alive = n.get(\\\"Alive\\\", False)
    ip = n.get(\\\"NodeManagerAddress\\\", \\\"?\\\")
    gpus = n.get(\\\"Resources\\\", {}).get(\\\"GPU\\\", 0)
    total_gpus += gpus
    print(f\\\"  {ip}: {int(gpus)} GPUs, alive={alive}\\\")
print(f\\\"Total GPUs: {int(total_gpus)}\\\")
if total_gpus < 16:
    print(f\\\"⚠️  Expected 16 GPUs but only {int(total_gpus)} available!\\\")
ray.shutdown()
\"
'" 2>&1 || echo "[Ray] Status check failed (non-fatal)"

# ============================================================
# Phase 2: vLLM サーバ起動 (vLLM env)
# ============================================================
echo ""
echo "================================================================"
echo "  Phase 2: vLLM server (env: qwen35-vllm-serve)"
echo "  TP=${VLLM_TP}, model=$(basename ${BASE_MODEL})"
echo "  NOTE: First run includes warmup (~40-60 min for GDN JIT)"
echo "  Subsequent runs will be fast (cached kernels)"
echo "================================================================"

VLLM_LOG="outputs/logs/vllm-${SLURM_JOB_ID}.log"

# TRL の HTTP ルート (/init_communicator, /update_named_param 等) を含む
# カスタムサーバスクリプトを使用
# vllm serve にはこれらのルートが存在しないため
ssh -o StrictHostKeyChecking=no "${VLLM_HEAD}" "bash -c '
    source ${ENV_SETUP_VLLM}
    export VLLM_HOST_IP=${VLLM_HEAD_IP}
    export RAY_ADDRESS=${VLLM_HEAD_IP}:${RAY_PORT}
    echo \"[vLLM] Launching vllm_trl_serve.py on \$(hostname)\"
    echo \"[vLLM] VLLM_HOST_IP=\${VLLM_HOST_IP}\"
    echo \"[vLLM] RAY_ADDRESS=\${RAY_ADDRESS}\"
    echo \"[vLLM] RAY_CGRAPH_get_timeout=\${RAY_CGRAPH_get_timeout}\"
    echo \"[vLLM] FLASHINFER_CACHE_DIR=\${FLASHINFER_CACHE_DIR}\"
    echo \"[vLLM] Python: \$(python --version)\"
    echo \"[vLLM] vLLM: \$(python -c \"import vllm; print(vllm.__version__)\" 2>/dev/null)\"
    echo \"[vLLM] TP=${VLLM_TP}, port=${VLLM_PORT}, executor=ray\"

    python3 ${SCRIPT_DIR}/vllm_trl_serve.py \
        --model ${BASE_MODEL} \
        --language-model-only \
        --tensor-parallel-size ${VLLM_TP} \
        --distributed-executor-backend ray \
        --port ${VLLM_PORT} \
        --dtype bfloat16 \
        --trust-remote-code \
        --gpu-memory-utilization ${VLLM_GPU_UTIL} \
        --enable-prefix-caching \
        --max-model-len 4096 \
        --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
        --enforce-eager \
        2>&1
'" > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

echo "[vLLM] Background PID=${VLLM_PID}, log=${VLLM_LOG}"

# vLLM 起動待ち (最大90分)
# warmup generate (GDN JIT) が完了するまで /health は返らない
# 初回: ~5分(モデル読込) + ~40-60分(JIT warmup) = ~45-65分
# 2回目以降: ~5分(モデル読込) + ~数秒(キャッシュヒット) = ~5-6分
echo "[vLLM] Waiting for server at http://${VLLM_HEAD_IP}:${VLLM_PORT}/health ..."
echo "[vLLM] (includes warmup — first run may take 40-60+ min for GDN JIT)"
VLLM_READY=0
WAIT_START=$(date +%s)
for i in $(seq 1 540); do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        "http://${VLLM_HEAD_IP}:${VLLM_PORT}/health" 2>/dev/null || echo "000")

    if [ "${HTTP_CODE}" = "200" ]; then
        WAIT_END=$(date +%s)
        WAIT_SECS=$((WAIT_END - WAIT_START))
        echo "[vLLM] ✅ Server ready! (waited ${WAIT_SECS}s = $((WAIT_SECS/60))m)"
        VLLM_READY=1
        break
    fi

    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM process died unexpectedly"
        echo "=== Last 50 lines of vLLM log ==="
        tail -50 "${VLLM_LOG}" 2>/dev/null
        exit 1
    fi

    # Progress every 60s
    if [ $((i % 6)) -eq 0 ]; then
        ELAPSED=$(($(date +%s) - WAIT_START))
        echo "[vLLM] Still loading/warming up... (${ELAPSED}s = $((ELAPSED/60))m elapsed)"
        tail -2 "${VLLM_LOG}" 2>/dev/null | sed 's/^/  /' || true
    fi
    sleep 10
done

if [ "${VLLM_READY}" -ne 1 ]; then
    WAIT_END=$(date +%s)
    WAIT_SECS=$((WAIT_END - WAIT_START))
    echo "ERROR: vLLM server did not start within ${WAIT_SECS}s"
    echo "=== Last 100 lines of vLLM log ==="
    tail -100 "${VLLM_LOG}" 2>/dev/null
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

echo "[vLLM] Endpoint check:"
curl -s "http://${VLLM_HEAD_IP}:${VLLM_PORT}/get_world_size" 2>/dev/null \
    | python3 -m json.tool 2>/dev/null || true

# ============================================================
# Phase 3: Trainer 起動 (6ノード, 48 GPU, Trainer env)
# ============================================================
echo ""
echo "================================================================"
echo "  Phase 3: Trainer launch (env: qwen35-trainer)"
echo "  ${TRAINER_NNODES} nodes × 8 GPUs = $((TRAINER_NNODES * 8)) processes"
echo "================================================================"

TRAINER_MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
TRAINER_NPROC=$((TRAINER_NNODES * 8))
VLLM_URL="http://${VLLM_HEAD_IP}:${VLLM_PORT}"

echo "Trainer MASTER : ${TRAINER_MASTER_IP}:${TRAINER_MASTER_PORT}"
echo "Trainer procs  : ${TRAINER_NPROC}"
echo "vLLM URL       : ${VLLM_URL}"
echo "Attention mode : ${TRAIN_ATTENTION}"

# EXTRA_ARGS 構築
EXTRA_ARGS=""
if [ -n "${MASK_JSON}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --mask_json ${MASK_JSON}"
fi
if [ "${TRAIN_ATTENTION}" != "none" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --train_attention ${TRAIN_ATTENTION}"
    if [ -n "${ATTN_LAYER_INDICES}" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --attn_layer_indices ${ATTN_LAYER_INDICES}"
    fi
fi
if [ -n "${TRAIN_LAYERNORM}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} ${TRAIN_LAYERNORM}"
fi

TRAINER_NODELIST_CSV=$(IFS=,; echo "${TRAINER_NODES[*]}")
TRAINER_NODES_STR="${TRAINER_NODES[*]}"

echo "Trainer nodelist: ${TRAINER_NODELIST_CSV}"
echo ""

# set -e で srun 非0終了時に Phase 4 が飛ばされないよう
# `|| TRAINER_EXIT=$?` で終了コードを捕捉する
TRAINER_EXIT=0
srun --nodes="${TRAINER_NNODES}" \
     --ntasks="${TRAINER_NNODES}" \
     --ntasks-per-node=1 \
     -w "${TRAINER_NODELIST_CSV}" \
     bash -c '
set -euo pipefail

# ★ Trainer 環境を activate (vLLM env ではない)
source '"${ENV_SETUP_TRAINER}"'

echo "[Trainer] Python: $(python --version)"
echo "[Trainer] transformers: $(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)"
echo "[Trainer] vLLM (NCCL): $(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "N/A")"

# ---- ノードランク算出 ----
TRAINER_NODES_ARR=('"${TRAINER_NODES_STR}"')
MY_HOST=$(hostname -s)
NODE_RANK=-1
for i in "${!TRAINER_NODES_ARR[@]}"; do
    TGT=$(echo "${TRAINER_NODES_ARR[$i]}" | cut -d. -f1)
    if [ "${TGT}" = "${MY_HOST}" ]; then
        NODE_RANK=$i
        break
    fi
done

if [ ${NODE_RANK} -lt 0 ]; then
    echo "ERROR: hostname ${MY_HOST} not in trainer nodes: ${TRAINER_NODES_ARR[*]}"
    exit 1
fi

NNODES='"${TRAINER_NNODES}"'
NPROC=8

echo "[Trainer] node_rank=${NODE_RANK}/${NNODES} host=${MY_HOST}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

accelerate launch \
  --multi_gpu \
  --num_machines ${NNODES} \
  --num_processes $((NNODES * NPROC)) \
  --machine_rank ${NODE_RANK} \
  --main_process_ip '"${TRAINER_MASTER_IP}"' \
  --main_process_port '"${TRAINER_MASTER_PORT}"' \
  --mixed_precision no \
  '"${SCRIPT}"' \
    --model '"${BASE_MODEL}"' \
    --layers 60 \
    --experts 512 \
    --dataset '"${DATASET}"' \
    --seq_len 4096 \
    --batch_size '"${BATCH_SIZE}"' \
    --grpo_group_size '"${GRPO_GROUP_SIZE}"' \
    --steps 100 \
    --lr 5e-5 \
    --grad_clip 1.0 \
    --gen_max_new_tokens '"${GEN_MAX_TOKENS}"' \
    --gen_temperature '"${GEN_TEMPERATURE}"' \
    --gen_top_p 0.9 \
    --gen_repetition_penalty 1.05 \
    --include_shared_expert \
    --use_vllm \
    --vllm_base_url '"${VLLM_URL}"' \
    --vllm_update_every 1 \
    --vllm_connection_timeout 600 \
    --judge_backend '"${JUDGE_BACKEND}"' \
    --judge_model_name '"${JUDGE_MODEL}"' \
    --judge_api_base '"${JUDGE_API_BASE}"' \
    --judge_max_new_tokens 2048 \
    --judge_concurrency '"${JUDGE_CONCURRENCY}"' \
    --reward_scale 1.0 \
    --reward_clip 1.0 \
    --wandb \
    --wandb_project '"${WANDB_PROJECT}"' \
    --wandb_run_name '"${WANDB_RUN_NAME}"' \
    --log_every 1 \
    --save_updated_only \
    --save_dir '"${OUTPUT_DIR}"' \
    '"${EXTRA_ARGS}"'
' || TRAINER_EXIT=$?

echo "[Trainer] 終了 exit_code=${TRAINER_EXIT}"

# ============================================================
# Phase 4: クリーンアップ (trap でも実行されるが明示的に呼ぶ)
# ============================================================
echo ""
echo "================================================================"
echo "  Phase 4: Cleanup"
echo "================================================================"

cleanup

echo ""
echo "================================================================"
echo "  ジョブ完了: $(date)"
echo "  Trainer exit: ${TRAINER_EXIT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  vLLM log: ${VLLM_LOG}"
echo "================================================================"

exit ${TRAINER_EXIT}
