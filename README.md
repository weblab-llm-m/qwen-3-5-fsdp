# Qwen3.5-397B-A17B-FSDP

A multi-node distributed training framework for **Qwen3.5-397B-A17B** built on FSDP (Fully Sharded Data Parallel). Provides three training pipelines — partial-layer SFT / partial-layer DPO / expert-only GRPO — plus a checkpoint merge tool that produces vLLM-ready models. Designed for SLURM + H100 multi-node clusters.

## Model Overview

| Item | Value |
|---|---|
| Model | Qwen3.5-397B-A17B |
| Total Parameters | 397B (17B active) |
| Layers | 60 |
| Attention | Hybrid: 45 Gated DeltaNet + 15 Gated Attention (3:1) |
| MoE | 512 experts (10 routed + 1 shared / layer) |
| Precision | BF16 native |
| Vision | ViT encoder (frozen during training) |

## Features

- **Partial-layer fine-tuning** — Train only the last N layers, restricted to attention / MLP / full modules (SFT, DPO)
- **Expert-only GRPO** — Train selected MoE expert parameters via reinforcement learning with LLM-as-judge rewards
- **Attention-mode selection in GRPO** — Train DeltaNet-only / full-attention-only / all / specific layer indices
- **LoRA support** — Custom hook-based LoRA injection in SFT / DPO (no PEFT dependency)
- **FSDP-vLLM weight sync** — Real-time NCCL weight broadcast from trainer to a Ray-backed vLLM server during GRPO
- **Centralized generation & judging** — Rank 0 gathers all prompts, generates and judges once, then broadcasts results
- **MoE-aware DPO** — Concatenated chosen+rejected forward pass for routing consistency, checkpointed cross-entropy for OOM prevention
- **Detach hooks** — Cut the autograd graph at the frozen→trainable boundary, drastically reducing VRAM
- **Sequence packing** — Pack short samples to maximize GPU utilization (SFT)
- **VLM-aware merge** — Auto-supplement visual encoder weights, config, and tokenizer files during merge
- **W&B integration** — Optional Weights & Biases experiment tracking in all pipelines

## Directory Structure

```
.
├── grpo/                                # GRPO (Reinforcement Learning)
│   ├── run_qwen35_expert_grpo.py        # Expert-only GRPO trainer
│   ├── run_qwen35_expert_grpo.sh        # SLURM job script (8 nodes, 4-phase orchestration)
│   ├── fsdp_vllm_client.py              # FSDP-aware vLLM client (NCCL weight sync)
│   ├── vllm_trl_serve.py                # vLLM + TRL weight-sync HTTP server
│   └── conda/
│       ├── trainer.sh                   # Trainer environment setup (qwen35-trainer)
│       └── vllm.sh                      # vLLM server environment setup (qwen35-vllm-serve)
├── sft/                                 # SFT (Supervised Fine-Tuning)
│   ├── run_qwen35_lastlayer_sft.py      # Partial-layer SFT trainer
│   ├── run_qwen35_lastlayer_sft.sh      # SLURM job script (10 nodes)
│   └── conda/
│       └── env.sh                       # SFT/DPO/Merge environment setup (qwen35-sft)
├── dpo/                                 # DPO (Direct Preference Optimization)
│   ├── run_qwen35_lastlayer_dpo.py      # Partial-layer DPO trainer
│   └── run_qwen35_lastlayer_dpo.sh      # SLURM job script (8 nodes)
└── merge/                               # Checkpoint Merge
    ├── merge_full_model.py              # FSDP sharded checkpoint → vLLM-ready model
    └── slurm_full_merge.sh              # SLURM job script (1 node)
```

## Prerequisites

- **GPU**: NVIDIA H100 multi-node cluster (10 nodes for SFT, 8 nodes for DPO / GRPO)
- **CUDA**: 12.x driver (SLURM scripts load `cuda/12.4`, `nccl/2.22.3`, `hpcx/2.18.1`)
- **Conda**: with Python 3.11
- **SLURM**: Job scheduler
- **transformers**: >= 5.1.0 (Qwen3.5 / `qwen3_5_moe` support; the SFT env installs >= 5.2.0)
- **Network**: High-speed interconnect (InfiniBand / RoCE) for NCCL
- **CPU RAM**: Rank 0 must hold the full BF16 base model (~794 GB) during loading

## Setup

### 1. Create the Conda Environments

For SFT / DPO / Merge (single environment, env name `qwen35-sft`):

```bash
bash sft/conda/env.sh
conda activate qwen35-sft
```

The script installs PyTorch (CUDA 12.4), `transformers>=5.2.0`, `accelerate`, `sentencepiece`, `protobuf`, and `wandb`, then verifies CUDA and the Qwen3.5 config (`num_hidden_layers`, `layer_types`).

For GRPO, **two separate environments** are required due to a transformers version conflict:

```bash
# vLLM server environment (env name: qwen35-vllm-serve)
#   vLLM built from source (nightly; stable 0.15.1 does not support Qwen3.5)
#   + TRL >= 0.28.0 (WeightSyncWorkerExtension) + fastapi/uvicorn/pydantic
#   + vllm_ascend stub patch
bash grpo/conda/vllm.sh

# Trainer environment (env name: qwen35-trainer)
#   PyTorch (CUDA 12.4) + accelerate + transformers >= 5.1.0 from source
#   + vLLM nightly (used ONLY for its NCCL utilities:
#     PyNcclCommunicator / StatelessProcessGroup — not for model loading)
#   No TRL needed — fsdp_vllm_client.py fully replaces it.
bash grpo/conda/trainer.sh
```

### 2. Prepare the Data

**SFT / GRPO** — ChatML JSONL (one JSON object per line):

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

- SFT also accepts a `text` field containing raw ChatML markup (`<|im_start|>role\n...<|im_end|>`), which is parsed into messages. Labels are masked so that only assistant turns are trained (manual masking — Qwen3.5's chat template does not support `return_assistant_tokens_mask`).
- GRPO uses the first `user` message as the prompt and the first `assistant` message (if present) as the **reference answer shown to the judge**.

**DPO** — prompt / chosen / rejected JSONL:

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

**Expert mask JSON (GRPO, optional)** — selects which routed experts to train:

```json
{"selected_layer_experts": [{"layer_idx": 0, "expert_id": 5}, {"layer_idx": 1, "expert_id": 12}]}
```

or a common expert-ID list applied to all layers:

```json
{"selected_experts_unique": [5, 12, 42, 87]}
```

Keys `selected_experts` and `picked_experts` are also accepted. Any layer whose mask ends up empty falls back to *all experts enabled* as a safety measure.

### 3. Configure Paths

Edit each `.sh` file to match your environment:

```bash
BASE_MODEL="/path/to/base_model"
DATASET="/path/to/dataset.jsonl"
```

Also adjust the SLURM settings (`--partition`, `--nodelist`) and the network interface (`TARGET_IFNAME`, default `eth0`).

**GRPO additional setup:**

```bash
export OPENROUTER_API_KEY="your-api-key"   # Required for the OpenRouter LLM judge
```

---

## 1. Supervised Fine-Tuning (SFT)

Fine-tunes the attention / MLP / full modules of the last N layers on ChatML JSONL data. Supports LoRA, sequence packing, assistant-turn masking, and detach hooks (frozen layers run without autograd; MLP outputs are detached when training attention-only).

- **Node config**: 10 nodes × 8 GPUs = 80 GPUs (FSDP FULL_SHARD)
- **Loading**: rank 0 loads BF16 weights to CPU; all other ranks build a meta model (zero CPU RAM); FSDP broadcasts via `sync_module_states=True`
- **Checkpoint output**: `params.rank{rank}.pt` (SHARDED_STATE_DICT, trainable params only) + `params_meta.json`

### Launch

```bash
sbatch sft/run_qwen35_lastlayer_sft.sh
```

### Basic Usage

```bash
accelerate launch \
    --multi_gpu \
    --num_machines 10 \
    --num_processes 80 \
    --mixed_precision no \
    sft/run_qwen35_lastlayer_sft.py \
    --model /path/to/Qwen3.5-397B-A17B \
    --num_layers 60 \
    --train_layers 1 \
    --train_modules attn \
    --dataset /path/to/train.jsonl \
    --seq_len 4096 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --packing \
    --epochs 1 \
    --lr 2e-5 \
    --save_every 50 \
    --wandb_project qwen35-sft \
    --save_dir ./output/qwen35-lastlayer-attn-sft
```

Note: `--mixed_precision no` is intentional — BF16 is handled by the FSDP `MixedPrecision` policy inside the script.

### Full Argument Reference

#### Model & Layer Selection

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | **required** | Path to base model |
| `--num_layers` | int | 60 | Number of decoder layers (Qwen3.5 = 60) |
| `--train_layers` | int | 1 | Number of layers to train, counted from the last layer |
| `--train_modules` | str | attn | Modules to train: `attn` (attention + input_layernorm) / `mlp` (MoE + post_attention_layernorm) / `full` (all modules in the layer) |

#### Data & Batching

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | **required** | Path to ChatML JSONL training data |
| `--seq_len` | int | 4096 | Maximum sequence length (truncation) |
| `--batch_size` | int | 2 | Micro-batch size per GPU |
| `--gradient_accumulation_steps` | int | 4 | Gradient accumulation steps |
| `--packing` | flag | off | Pack multiple samples into fixed-length `seq_len` sequences |
| `--bucket_buffer` | int | 256 | Reserved for length bucketing (accepted by the CLI; not used by the current training loop) |

#### Training Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--epochs` | int | 1 | Number of epochs (`<=0` switches to step-only mode) |
| `--steps` | int | 0 | Maximum optimizer steps (0 = unlimited; acts as a cap in epoch mode) |
| `--lr` | float | 2e-5 | Learning rate (AdamW, betas=(0.9, 0.98)) |
| `--weight_decay` | float | 0.01 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping threshold (0 to disable) |
| `--optim` | str | adamw | Optimizer (only `adamw` is available) |
| `--seed` | int | 42 | Random seed |

#### LoRA

Custom hook-based LoRA: `lora_A` / `lora_B` parameters are registered on every 2-D-weight linear module in the target layers (layernorms are skipped) and applied via forward hooks. The base weights stay frozen.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_lora` | flag | off | Enable LoRA (only `lora_A` / `lora_B` are trainable) |
| `--lora_rank` | int | 16 | LoRA rank r |
| `--lora_alpha` | int | 32 | LoRA alpha (scaling = alpha / r) |
| `--lora_dropout` | float | 0.0 | LoRA dropout rate |

#### Save & Logging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--save_dir` | str | ./output/qwen35-lastlayer-sft | Checkpoint save directory |
| `--save_every` | int | 0 | Save a `step-{N}` checkpoint every N optimizer steps (0 = only at end) |
| `--log_every` | int | 1 | Log output interval (optimizer steps) |
| `--wandb_project` | str | "" | W&B project name (empty string disables W&B) |
| `--wandb_run_name` | str | "" | W&B run name |

---

## 2. Direct Preference Optimization (DPO)

Optimizes the attention / MLP / full modules of the last N layers via DPO on {prompt, chosen, rejected} pairs. Model loading, layer selection, LoRA, detach hooks, and checkpoint format are identical to SFT.

- **Node config**: 8 nodes × 8 GPUs = 64 GPUs (FSDP FULL_SHARD)
- **Reference log-probs**: precomputed once before training, distributed across all ranks and merged via `all_gather_object` — no reference model is kept in memory during training
- **Checkpoint output**: `params.rank{rank}.pt` + `params_meta.json` (includes `beta`, `loss_type`)

### Processing Flow

1. Tokenize each pair with assistant-turn label masking (chosen and rejected padded to a common length)
2. Precompute reference log-probs with a **concatenated chosen+rejected forward pass** (batch size forced to 1 for MoE routing consistency)
3. Training: single concatenated forward per micro-batch → per-sequence log-probs via **checkpointed cross-entropy** (avoids materializing the full log-softmax) → DPO loss
4. Logs loss, reward margin, and preference accuracy; periodic sharded saves

### Launch

```bash
sbatch dpo/run_qwen35_lastlayer_dpo.sh
```

### Basic Usage

```bash
accelerate launch \
    --multi_gpu \
    --num_machines 8 \
    --num_processes 64 \
    --mixed_precision no \
    dpo/run_qwen35_lastlayer_dpo.py \
    --model /path/to/Qwen3.5-397B-A17B \
    --num_layers 60 \
    --train_layers 1 \
    --train_modules attn \
    --dataset /path/to/train_dpo.jsonl \
    --seq_len 4096 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --epochs 3 \
    --lr 5e-6 \
    --beta 0.1 \
    --loss_type sigmoid \
    --save_every 50 \
    --wandb_project qwen35-dpo \
    --save_dir ./output/qwen35-lastlayer-attn-dpo
```

### Full Argument Reference

#### Model & Layer Selection

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | **required** | Path to base model |
| `--num_layers` | int | 60 | Number of decoder layers |
| `--train_layers` | int | 1 | Number of layers to train, counted from the last layer |
| `--train_modules` | str | attn | Modules to train (`attn` / `mlp` / `full`) |

#### Data & Batching

| Argument | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | **required** | Path to DPO JSONL (prompt / chosen / rejected) |
| `--seq_len` | int | 4096 | Maximum sequence length |
| `--batch_size` | int | 1 | Micro-batch size per rank (in pairs) |
| `--gradient_accumulation_steps` | int | 4 | Gradient accumulation steps (`sync_each_batch=True`) |
| `--ref_batch_size` | int | 1 | Ignored — forced to 1 for MoE routing consistency during reference log-prob precompute |

#### DPO Parameters

| Argument | Type | Default | Description |
|---|---|---|---|
| `--beta` | float | 0.1 | DPO temperature β (higher = more conservative) |
| `--loss_type` | str | sigmoid | Loss variant: `sigmoid` / `hinge` / `ipo` |
| `--label_smoothing` | float | 0.0 | Label smoothing factor (sigmoid loss only) |
| `--length_normalise` | flag | off | Normalize log-probs by response token length |

#### Training Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--epochs` | int | 1 | Number of epochs (`<=0` switches to step-only mode) |
| `--steps` | int | 0 | Maximum optimizer steps (0 = unlimited) |
| `--lr` | float | 5e-6 | Learning rate (AdamW, betas=(0.9, 0.98)) |
| `--weight_decay` | float | 0.01 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping threshold |
| `--seed` | int | 42 | Random seed (also used by DistributedSampler shuffling) |

#### LoRA

Same mechanism and arguments as SFT: `--use_lora` (flag, off), `--lora_rank` (16), `--lora_alpha` (32), `--lora_dropout` (0.0).

#### Save & Logging

| Argument | Type | Default | Description |
|---|---|---|---|
| `--save_dir` | str | ./output/qwen35-lastlayer-dpo | Checkpoint save directory |
| `--save_every` | int | 0 | Save every N optimizer steps (0 = only at end) |
| `--log_every` | int | 1 | Log output interval (also logs reward margin / accuracy) |
| `--wandb_project` | str | "" | W&B project name (empty disables) |
| `--wandb_run_name` | str | "" | W&B run name |

---

## 3. Expert-Only GRPO

Trains selected MoE expert parameters (optionally plus shared experts, routers, attention layers, and layernorms) via a GRPO-style objective with an LLM-as-judge reward. Candidate generation runs on a separate vLLM server with real-time NCCL weight synchronization.

- **Node config**: 8 nodes total = 2 vLLM nodes (Ray cluster, TP=16) + 6 trainer nodes (FSDP FULL_SHARD, 48 GPUs)
- **Reward**: judge scores each candidate `{0, 0.5, 1}`; group-normalized advantage `(score − group_mean) / (group_std + 1e-6)`; judge failures return NaN and are masked out of the advantage (groups with < 2 valid scores contribute no gradient)
- **Centralized actor-judge**: every step, all ranks' prompts are gathered to rank 0, which generates via vLLM and judges once, then broadcasts results back — this is built in, not optional
- **Checkpoint output**: `model_sharded.rank{rank}.pt` (full sharded model) or `updated_params_sharded.rank{rank}.pt` with `--save_updated_only`, plus a meta JSON

### Launch

```bash
sbatch grpo/run_qwen35_expert_grpo.sh
```

The SLURM script orchestrates 4 phases automatically:

1. **Phase 1**: Start a Ray cluster on the 2 vLLM nodes (head + worker, `--num-gpus=8` each, IPs pinned via `--node-ip-address` / `VLLM_HOST_IP`)
2. **Phase 2**: Launch `vllm_trl_serve.py` (TP=16, `max-model-len 4096`, `max-num-seqs 128`, `gpu-memory-utilization 0.90`, `enforce-eager`, prefix caching). The server runs a warmup generation to trigger FlashInfer GDN JIT compilation (**40–60 min on first run**) before accepting connections; the health-check loop polls `/health` every 10 s for up to 90 minutes
3. **Phase 3**: Launch the FSDP trainer on the remaining 6 nodes (48 processes) in the `qwen35-trainer` env
4. **Phase 4**: Cleanup — stop the vLLM server and Ray on all vLLM nodes

### Basic Usage

```bash
accelerate launch \
    --multi_gpu \
    --num_machines 6 \
    --num_processes 48 \
    --mixed_precision no \
    grpo/run_qwen35_expert_grpo.py \
    --model /path/to/Qwen3.5-397B-A17B \
    --layers 60 \
    --experts 512 \
    --dataset /path/to/train.jsonl \
    --seq_len 4096 \
    --batch_size 2 \
    --grpo_group_size 4 \
    --steps 100 \
    --lr 5e-5 \
    --gen_max_new_tokens 512 \
    --include_shared_expert \
    --train_attention indices \
    --attn_layer_indices 59 \
    --use_vllm \
    --vllm_base_url "http://vllm-head:8000" \
    --vllm_update_every 1 \
    --judge_backend openrouter \
    --judge_model_name qwen/qwen3-235b-a22b \
    --judge_api_base https://openrouter.ai/api \
    --judge_concurrency 100 \
    --wandb \
    --save_updated_only \
    --save_dir ./output/qwen35-expert-grpo
```

### Full Argument Reference

#### Model & Data

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | **required** | Path to base model |
| `--layers` | int | 60 | Number of decoder layers (Qwen3.5 = 60) |
| `--experts` | int | 512 | Number of routed experts per layer (Qwen3.5 = 512) |
| `--dataset` | str | **required** | Path to ChatML JSONL (user prompt + optional assistant reference) |
| `--seq_len` | int | 4096 | Maximum sequence length for the policy batch |

#### Expert Selection

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mask_json` | str | None | Expert selection mask JSON (omit to train no routed experts) |
| `--include_shared_expert` | flag | off | Also train the shared expert in layers where any routed expert is selected |
| `--train_router` | flag | off | Also train router/gate weights (`mlp.gate.weight`) |

#### Attention Training

Qwen3.5 exposes `layer_types` in its config: 45 `linear_attention` (Gated DeltaNet) + 15 `full_attention` (Gated Attention) layers.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--train_attention` | str | none | Which layers' attention params to train: `none` / `full` (full-attention layers only) / `linear` (DeltaNet layers only) / `all` / `indices` |
| `--attn_layer_indices` | str | None | Comma-separated layer indices, required for `--train_attention=indices` (e.g. `"0,15,30,45,59"`) |
| `--train_layernorm` | flag | off | Also train layernorm params in the attention-trained layers |

#### Training Control

| Argument | Type | Default | Description |
|---|---|---|---|
| `--grpo_group_size` | int | 4 | Number of candidate responses per prompt |
| `--steps` | int | 1500 | Maximum training steps |
| `--epochs` | int | 0 | Number of epochs (>0 overrides `--steps` with `epochs × steps_per_epoch`) |
| `--batch_size` | int | 2 | Prompts per rank per step |
| `--lr` | float | 5e-5 | Learning rate (AdamW, betas=(0.9, 0.98)) |
| `--weight_decay` | float | 0.0 | Weight decay |
| `--grad_clip` | float | 1.0 | Gradient clipping threshold |

#### Generation

| Argument | Type | Default | Description |
|---|---|---|---|
| `--gen_max_new_tokens` | int | 192 | Maximum tokens per candidate response |
| `--gen_temperature` | float | 0.7 | Sampling temperature |
| `--gen_top_p` | float | 0.9 | Top-p (nucleus) sampling |
| `--gen_repetition_penalty` | float | 1.05 | Repetition penalty |

#### vLLM Integration

| Argument | Type | Default | Description |
|---|---|---|---|
| `--use_vllm` | flag | off | Generate via vLLM server(s) instead of local `model.generate()` (local generation in centralized mode is only safe without FSDP) |
| `--vllm_base_url` | str | None | vLLM server URL (single server) |
| `--vllm_base_urls` | str | None | Comma-separated URLs for multiple servers (prompts are round-robined and generated in parallel) |
| `--vllm_update_every` | int | 1 | Sync weights to vLLM every N optimizer steps |
| `--vllm_force_n1` | flag | off | Emulate n>1 sampling as repeated n=1 calls (debugging) |
| `--vllm_connection_timeout` | int | 240 | Server health-check timeout at client startup (seconds) |

#### Judge

The judge grades each candidate for correctness, completeness, safety, and clarity, and must output `[1]`, `[0.5]`, or `[0]` on its final line. Scores are cached per rank (`reward_cache` is rewritten to `<name>.rank{rank}.jsonl`). API keys are read from `--judge_api_key`, else `OPENROUTER_API_KEY` (openrouter backend), else `JUDGE_API_KEY`. Retries handle 429 rate limits (honoring `Retry-After` with jitter), 5xx, and connection errors.

| Argument | Type | Default | Description |
|---|---|---|---|
| `--judge_backend` | str | local | `local` (the training model judges itself) / `openrouter` / `openai` |
| `--judge_model_name` | str | None | Judge model name (required for external backends) |
| `--judge_api_base` | str | None | API base URL (defaults: `https://openrouter.ai/api` for openrouter, `http://localhost:8000` for openai) |
| `--judge_api_key` | str | None | API key (falls back to env vars, see above) |
| `--judge_max_new_tokens` | int | 2048 | Maximum generation tokens for the judge |
| `--judge_temperature` | float | 0.0 | Judge sampling temperature |
| `--judge_max_retries` | int | 1000 | Maximum retry count per judge call |
| `--judge_concurrency` | int | 1 | Parallel judge API calls (external backends only; e.g. 100 for OpenRouter) |

#### Reward

| Argument | Type | Default | Description |
|---|---|---|---|
| `--reward_scale` | float | 1.0 | Advantage scaling factor |
| `--reward_clip` | float | 1.0 | Advantage clipping range (0 to disable) |
| `--ema_alpha` | float | 0.05 | EMA smoothing factor for the reward baseline (logged for monitoring) |
| `--reward_cache` | str | judge_score_cache.jsonl | Judge score cache file (per-rank suffix added automatically) |

#### Logging & W&B

| Argument | Type | Default | Description |
|---|---|---|---|
| `--wandb` | flag | off | Enable W&B logging (metrics + per-step sample table with prompts, candidates, judge outputs, scores, advantages) |
| `--wandb_project` | str | qwen35_expert_grpo | W&B project name |
| `--wandb_run_name` | str | None | W&B run name |
| `--wandb_entity` | str | None | W&B entity |
| `--log_every` | int | 20 | Log output interval (steps) |
| `--debug_print` | flag | off | Print sample prompts/candidates each step |

#### Save Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--save_dir` | str | expert_ckpts_qwen35_grpo | Checkpoint save directory |
| `--save_updated_only` | flag | off | Save only trainable parameters (sharded, `updated_params_sharded.rank{rank}.pt`) |
| `--save_legacy_only` | flag | off | Save only via `accelerator.save_state()` |
| `--save_legacy_after_new` | flag | off | Also run `accelerator.save_state()` after the new-format save |
| `--legacy_save_subdir` | str | legacy_accelerate_state | Subdirectory name for the legacy save |

### GRPO Step Flow

1. All ranks gather their prompts/references to rank 0 (`all_gather_object`)
2. Rank 0 generates `group_size` candidates per prompt on the vLLM server(s)
3. Rank 0 judges all candidates (parallel external API calls with score caching)
4. Results are broadcast to all ranks; each rank slices out its own portion
5. Group-normalized advantages are computed (NaN-masked for judge failures)
6. Policy gradient: `loss = −mean(advantage × mean_token_logp)` over valid samples, backward through FSDP, clip, AdamW step
7. Trainable weights are streamed to the vLLM server(s) via NCCL (`--vllm_update_every`), followed by a prefix-cache reset

---

## 4. vLLM Inference Server

`vllm serve` does not expose the TRL weight-sync HTTP routes, so GRPO uses the custom server `grpo/vllm_trl_serve.py`. It wraps vLLM's offline `LLM()` API (multi-node TP via Ray) with FastAPI endpoints and loads TRL's `WeightSyncWorkerExtension` as a `worker_extension_cls`.

**Fire-and-forget design**: `init_communicator` / `update_named_param` / `close_communicator` submit `collective_rpc` to a single-thread executor and return immediately. The NCCL operations inside `collective_rpc` block until *all* ranks (workers + trainer client) participate — if the HTTP handler blocked on them, the client could never reach its own NCCL call, causing deadlock.

**GDN warmup**: on startup (before uvicorn binds the port), the server runs a 1-token warmup generation to trigger FlashInfer Gated DeltaNet JIT compilation (40–60+ min on first run; cached afterwards). `/health` therefore does not return 200 until JIT is complete. `RAY_CGRAPH_get_timeout` defaults to 7200 s inside the script.

### Launch

Launched automatically by `grpo/run_qwen35_expert_grpo.sh` (Phase 2). Manual launch:

```bash
conda activate qwen35-vllm-serve
python grpo/vllm_trl_serve.py \
    --model /path/to/Qwen3.5-397B-A17B \
    --language-model-only \
    --tensor-parallel-size 16 \
    --distributed-executor-backend ray \
    --port 8000 \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --max-model-len 4096 \
    --max-num-seqs 128 \
    --enforce-eager
```

### Server Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | **required** | Path to model |
| `--revision` | str | None | Model revision |
| `--tensor-parallel-size` | int | 1 | Tensor parallel size |
| `--pipeline-parallel-size` | int | 1 | Pipeline parallel size |
| `--host` | str | 0.0.0.0 | HTTP bind address |
| `--port` | int | 8000 | HTTP port |
| `--gpu-memory-utilization` | float | 0.90 | vLLM GPU memory fraction |
| `--dtype` | str | auto | Model dtype |
| `--max-model-len` | int | None | Maximum model context length |
| `--enable-prefix-caching` | flag | off | Enable prefix caching |
| `--enforce-eager` | flag | off | Disable CUDA graphs |
| `--trust-remote-code` | flag | off | Trust remote code |
| `--language-model-only` | flag | off | Skip the vision encoder (`limit_mm_per_prompt={"image": 0, "video": 0}`) |
| `--distributed-executor-backend` | str | ray | Executor backend: `ray` / `mp` / `uni` |
| `--kv-cache-dtype` | str | auto | KV cache dtype |
| `--max-num-seqs` | int | None | Maximum sequences per scheduler iteration |
| `--log-level` | str | info | uvicorn log level |
| `--skip-warmup` | flag | off | Skip the warmup generation (not recommended for GDN models) |

### HTTP Endpoints

| Endpoint | Method | Blocking | Description |
|---|---|---|---|
| `/health` | GET | — | Health check (`{"status": "ok"}`) |
| `/get_world_size` | GET | — | Returns `{"world_size": TP}` |
| `/get_tensor_parallel_size` | GET | — | Alias returning `{"tensor_parallel_size": TP}` |
| `/init_communicator` | POST | fire-and-forget | Picks an open port, starts worker-side NCCL group setup, returns `{host, port, world_size}` (world_size = TP + 1 client) |
| `/update_named_param` | POST | fire-and-forget | Receives `{name, dtype, shape}` metadata; workers then receive the tensor via NCCL broadcast |
| `/close_communicator` | POST | fire-and-forget | Tears down the NCCL group |
| `/generate` | POST | synchronous | Text generation; accepts `prompts, n, max_tokens, temperature, top_p, top_k, min_p, repetition_penalty, guided_decoding_regex`; returns `{"completion_ids": [[int, ...], ...]}` |
| `/reset_prefix_cache` | POST | synchronous | Clears the prefix cache (mandatory after weight updates — stale KV blocks otherwise cause hangs or wrong outputs) |

Trailing-slash variants (`/health/`, etc.) are also registered for every route.

### Client: `FSDPVLLMClient` (grpo/fsdp_vllm_client.py)

Standalone FSDP-aware client — no TRL dependency on the trainer side. Only trainer rank 0 joins the NCCL group; all ranks may use HTTP generation. Exported alias: `VLLMClient = FSDPVLLMClient`.

| Method | Description |
|---|---|
| `__init__(base_url, connection_timeout=240, max_retries=10)` | Waits for `/health` to return 200 (up to `connection_timeout` s) |
| `init_communicator(device)` | POSTs `/init_communicator`, creates a `StatelessProcessGroup` at the returned host:port as the last rank (trainer), builds a `PyNcclCommunicator` (rank 0 only) |
| `update_named_param(name, tensor)` | POSTs param metadata, then NCCL-broadcasts the tensor with `src = world_size − 1`; the broadcast returning guarantees the workers received it |
| `update_model_params(model, trainable_only=False)` | Sends all (or trainable-only) params of a non-FSDP model, then resets the prefix cache |
| `update_model_params_streaming_fsdp(fsdp_model, trainable_only=False)` | Streams FSDP-sharded weights unit-by-unit via `FSDP.summon_full_params(rank0_only=True)`; all ranks must call it (collective), only rank 0 broadcasts; skips units with no trainable params; resets the prefix cache and barriers at the end |
| `reset_prefix_cache()` | POSTs `/reset_prefix_cache` |
| `generate(prompts, n=1, max_tokens=192, temperature=0.7, top_p=0.9, repetition_penalty=1.05, min_p=0.0, **kwargs)` | POSTs `/generate` with retries; the first call uses an extended 7200 s timeout (JIT safety net), later calls 600 s |
| `close_communicator()` | POSTs `/close_communicator` and drops the local communicator |

**Weight sync protocol** (per parameter):

1. `init_communicator` — establish the NCCL group (StatelessProcessGroup + PyNcclCommunicator, world_size = TP + 1)
2. `update_named_param` — `summon_full_params` per FSDP unit → rank 0 NCCL broadcast (src = trainer rank); NCCL's blocking semantics serialize updates without extra locking
3. `reset_prefix_cache` — invalidate stale KV blocks
4. `close_communicator` — cleanup at the end of training

---

## 5. Model Merging

`merge/merge_full_model.py` consolidates FSDP sharded checkpoints into a `save_pretrained`-compatible model directory that vLLM can serve directly. Supports both full fine-tune and LoRA checkpoints from the SFT/DPO pipelines (`params.rank*.pt`), plus `attn_params.rank*.pt` and a generic `*rank*.pt` fallback. Runs on a single node on CPU.

- **Auto-detection**: rank-file pattern, meta file (`params_meta.json` / `attn_params_meta.json`), and LoRA mode (from meta `use_lora` or the presence of `lora_A`/`lora_B` keys)
- **VLM completion (Phase 5)**: fixes `save_pretrained` saving only the `language_model` submodel — copies config / tokenizer / processor / chat template files from the base, copies missing `visual.*` weight shards, and updates `model.safetensors.index.json`
- **FP8-aware**: the same script handles FP8 bases (e.g. GLM-5-FP8) with block-wise FP8→BF16 dequantization; Qwen3.5 is BF16 native so no dequant occurs

### Processing Flow

1. **Phase 1**: Two-pass streaming merge of `rank*.pt` ShardedTensor files (metadata scan → allocate → stream-write), with strict dim-0 contiguity validation
2. **Phase 2**: Load the base model to CPU (`low_cpu_mem_usage=True`, finalize-loading skipped)
3. **Phase 3**: Full mode — overwrite base weights with checkpoint tensors (VLM-aware key mapping `model.layers.*` ↔ `model.language_model.layers.*`); LoRA mode — `W += B @ A × (α / r)`
4. **Phase 4**: Save — `keep` mode uses `save_pretrained` with original dtypes; `bf16` mode streams layer-by-layer safetensors with layer eviction to the meta device
5. **Phase 5**: Copy non-weight files from base (protecting the merged index.json), supplement visual encoder shards (VLM only), verify output (config model_type, tokenizer files, shard completeness, visual weights)

### Launch

```bash
sbatch merge/slurm_full_merge.sh
```

### Basic Usage

```bash
python merge/merge_full_model.py \
    --base_model /path/to/Qwen3.5-397B-A17B \
    --checkpoint_dir ./output/qwen35-lastlayer-attn-dpo \
    --out_dir ./output/Qwen3.5-397B-A17B-merged \
    --num_layers 60 \
    --output_dtype keep

# Serve the result:
vllm serve ./output/Qwen3.5-397B-A17B-merged --trust-remote-code
```

### Full Argument Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--base_model` | str | **required** | Path to base model (HF local directory) |
| `--checkpoint_dir` | str | **required** | Directory containing `params.rank*.pt` (+ `params_meta.json`) |
| `--out_dir` | str | **required** | Output directory |
| `--num_layers` | int | 78 | Number of decoder layers — **set to 60 for Qwen3.5** (default targets GLM-5) |
| `--world_size` | int | 0 | Number of rank files (0 = read from meta / auto-glob) |
| `--output_dtype` | str | keep | `keep` = `save_pretrained` with original dtypes / `bf16` = streaming layer-by-layer BF16 safetensors |
| `--max_shard_size` | str | 5GB | `save_pretrained` shard size (keep mode only) |
| `--strict` | flag | on | Strict shard validation: abort on dim-0 gaps or incomplete tensors |
| `--force_lora_rank` | int | 0 | Override `lora_rank` from meta (0 = use meta, meta default 16) |
| `--force_lora_alpha` | float | 0 | Override `lora_alpha` from meta (0 = use meta, meta default 32) |
| `--skip_vlm_fix` | flag | off | Skip VLM post-processing (visual weight copy) |

---

## 6. SLURM Configuration

### Environment Variables

Common settings used by all training job scripts:

```bash
# Modules
module load nccl/2.22.3
module load hpcx/2.18.1-gcc-cuda12/hpcx-mt
module load cuda/12.4

# Network interface (adjust to your cluster)
export TARGET_IFNAME="eth0"

# NCCL
export NCCL_DEBUG=WARN
export NCCL_P2P_LEVEL=NVL                     # Use NVLink
export NCCL_IB_DISABLE=0                      # Enable InfiniBand
export NCCL_SOCKET_IFNAME="${TARGET_IFNAME}"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_NCHANNELS=4
export NCCL_NCHANNELS_PER_NET_PEER=1
export NCCL_CUMEM_ENABLE=0
export NCCL_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200

# Gloo / TensorPipe
export GLOO_SOCKET_IFNAME="${TARGET_IFNAME}"
export TP_SOCKET_IFNAME="${TARGET_IFNAME}"

# PyTorch
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=7200
export OMP_NUM_THREADS=2
ulimit -n 262144
```

GRPO-specific additions:

```bash
# FlashInfer GDN JIT cache persistence (first run 40-60 min → cache hit afterwards)
export FLASHINFER_CACHE_DIR="${HOME}/.cache/flashinfer"

# Ray Compiled DAG timeouts (must cover GDN JIT compilation)
export RAY_CGRAPH_get_timeout=7200
export RAY_CGRAPH_teardown_timeout=120

# Judge API key
export OPENROUTER_API_KEY="your-api-key"
```

### Node Configuration

| Pipeline | Nodes | GPUs/Node | Total GPUs | Notes |
|---|---|---|---|---|
| SFT | 10 | 8 | 80 | accelerate launch, FSDP FULL_SHARD |
| DPO | 8 | 8 | 64 | accelerate launch, FSDP FULL_SHARD |
| GRPO — vLLM | 2 | 8 | 16 | Ray cluster, TP=16, env `qwen35-vllm-serve` |
| GRPO — Trainer | 6 | 8 | 48 | accelerate launch, env `qwen35-trainer` |
| Merge | 1 | 1 | 1 | Merge itself runs on CPU (`--mem=0` = full node RAM) |

### Accelerate Launch Template

```bash
accelerate launch \
    --multi_gpu \
    --num_machines ${SLURM_NNODES} \
    --num_processes $((SLURM_NNODES * 8)) \
    --machine_rank ${SLURM_NODEID} \
    --main_process_ip ${MASTER_ADDR} \
    --main_process_port ${MASTER_PORT} \
    --mixed_precision no \
    script.py [args...]
```

`--mixed_precision no` is deliberate in every pipeline: BF16 is applied via the FSDP `MixedPrecision` policy (`param_dtype` / `reduce_dtype` / `buffer_dtype` all `torch.bfloat16`) inside each trainer.

---

## GRPO Architecture

```
┌──────────────────────────────┐   ┌──────────────────────────────┐
│  vLLM Server (2 nodes)       │   │  FSDP Trainer (6 nodes)      │
│  conda: qwen35-vllm-serve    │   │  conda: qwen35-trainer       │
│  TP=16, Ray cluster          │   │  48 GPUs, FULL_SHARD         │
│  - /generate (inference)     │◄──│  - GRPO objective            │
│  - /update_named_param       │   │  - NCCL weight sync          │
│  - /init_communicator        │   │  - LLM-as-Judge reward       │
│  - /reset_prefix_cache       │   │  - Rank-0 gather/broadcast   │
└──────────────────────────────┘   └──────────────────────────────┘
        HTTP + NCCL                     Accelerate + FSDP
```

## Memory Efficiency Techniques

Training a 397B-parameter model requires aggressive memory optimization:

- **Rank 0-only model loading** — Only rank 0 loads BF16 weights to CPU; other ranks build a meta-device model with zero CPU RAM (the HF loading internals are temporarily no-op'ed). FSDP broadcasts real weights via `sync_module_states=True`
- **FSDP FULL_SHARD** — Shards parameters, gradients, and optimizer states (ZeRO Stage 3 equivalent) with per-decoder-layer auto-wrapping (the layer class is auto-discovered from transformers or the instantiated model)
- **Detach hooks (SFT/DPO)** — A forward pre-hook on the first trainable layer detaches its inputs, so frozen layers 0..N−1 never build an autograd graph; when training attention-only, MLP (MoE) outputs are also detached
- **Activation checkpointing** — Enabled in GRPO (`gradient_checkpointing_enable`); disabled in SFT/DPO (partial-layer training already bounds activation memory)
- **Selective parameter training** — Only target modules get `requires_grad=True`; GRPO checkpoints can save trainable params only
- **BF16 throughout** — FSDP `MixedPrecision` with `param_dtype` / `reduce_dtype` / `buffer_dtype` all `torch.bfloat16`
- **Concatenated forward pass (DPO)** — Single forward for chosen + rejected, keeping MoE routing consistent between the two
- **Checkpointed cross-entropy (DPO)** — Per-token log-probs computed inside `torch.utils.checkpoint` to avoid materializing the full-vocabulary log-softmax
- **Reference log-prob precompute (DPO)** — No reference model held during training
- **Streaming merge** — Two-pass shard merge plus layer-by-layer BF16 save with meta-device eviction and explicit `gc.collect()`

## System Requirements

- CUDA 12.x driver
- conda (Python 3.11)
- SLURM job scheduler
- 8+ node H100 cluster (DPO / GRPO) / 10 nodes (SFT)
- transformers >= 5.1.0 (Qwen3.5 support)
- InfiniBand / RoCE network (for NCCL)

## Notes

- **GDN JIT**: Qwen3.5's Gated DeltaNet kernels (FlashInfer) are JIT-compiled on first run, taking 40–60 minutes. The vLLM server warms up before accepting connections, and `FLASHINFER_CACHE_DIR` persists the compiled kernels for subsequent runs. `RAY_CGRAPH_get_timeout=7200` must cover the compilation window.
- **Separate GRPO environments**: the vLLM server needs vLLM-nightly's transformers (4.x), while the trainer needs transformers >= 5.1.0 — hence two conda environments. The trainer installs vLLM only for its NCCL utilities (`PyNcclCommunicator`, `StatelessProcessGroup`), never for model loading.
- **vllm_ascend stub**: TRL's `vllm_serve` imports an Ascend NPU module; both the env setup and the SLURM script create a stub package so `WeightSyncWorkerExtension` imports cleanly.
- **Prefix cache invalidation**: after every weight sync, the client resets the vLLM prefix cache — stale KV blocks otherwise cause hangs or incorrect outputs (see vllm-project/vllm#9744, huggingface/trl#3024).
- **CPU memory**: rank 0 needs enough RAM to hold the full BF16 base model (~794 GB) during loading; the merge node likewise loads the full model on CPU.
- **Assistant-turn masking**: Qwen3.5's chat template does not support `return_assistant_tokens_mask`, so SFT/DPO locate assistant spans in the formatted string and map character offsets to token offsets via prefix re-tokenization.

## License

[MIT](LICENSE)

## Acknowledgements

This program is based on results obtained from a project, JPNP25006, commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
