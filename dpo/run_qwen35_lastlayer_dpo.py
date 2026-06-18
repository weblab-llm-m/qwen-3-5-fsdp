#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partial-layer DPO for Qwen3.5-397B-A17B (FSDP, BF16).

Based on run_qwen35_lastlayer_sft.py — same model loading / FSDP / layer
selection logic, with DPO-specific changes:
  - Reads JSONL with {prompt, chosen, rejected} fields
  - Precomputes reference log-probs (distributed) before training
  - DPO loss: -log σ(β · (Δ_chosen - Δ_rejected))
  - Concatenates chosen+rejected into one forward pass per step
  - Optional label smoothing, length normalisation

Supports:
  - Full fine-tuning of selected modules (attn, mlp, or full layer)
  - LoRA adaptation (--use_lora)
  - Configurable number of trainable layers (--train_layers N)
  - Gradient accumulation (--gradient_accumulation_steps)
  - W&B logging (--wandb_project)

Model: Qwen3.5-397B-A17B
  - 60 layers, hybrid attention (45 Gated DeltaNet + 15 Gated Attention)
  - 512 MoE experts (10 routed + 1 shared)
  - BF16 native

Loading strategy: same as SFT (rank 0 loads, others meta, FSDP syncs).
"""

import os
import json
import math
import random
import argparse
import datetime
import time
from typing import List, Dict, Optional, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs, GradientAccumulationPlugin
from torch.distributed.fsdp import (
    ShardingStrategy,
    StateDictType,
    ShardedStateDictConfig,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    _FSDP_AVAILABLE = True
except Exception:
    _FSDP_AVAILABLE = False


# ============================================================
# Utilities (shared with SFT)
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_sdp_backends():
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


# ============================================================
# Trainable parameter selection (identical to SFT)
# ============================================================

TRAIN_MODULE_PATTERNS = {
    "attn": {
        "include": ("self_attn.", "attn.", "attention.", "gated_deltanet.",
                    "linear_attn.", "input_layernorm."),
        "description": "attention (DeltaNet or GatedAttn) + input_layernorm",
    },
    "mlp": {
        "include": ("mlp.", "moe.", "feed_forward.", "ffn.",
                    "block_sparse_moe.", "experts.",
                    "shared_expert.", "shared_experts.",
                    "post_attention_layernorm.", "post_attn_layernorm."),
        "description": "mlp/moe + post_attention_layernorm",
    },
    "full": {
        "include": None,
        "description": "all modules in layer",
    },
}


def inject_lora_adapters(model, train_layer_indices, train_modules,
                         lora_rank, lora_alpha, lora_dropout, rank=0):
    """Inject LoRA A/B parameters into target linear layers."""
    pattern_info = TRAIN_MODULE_PATTERNS.get(train_modules)
    if pattern_info is None:
        raise ValueError(f"Unknown train_modules: {train_modules}")

    include_patterns = pattern_info["include"]
    train_set = set(train_layer_indices)

    if include_patterns is not None:
        target_prefixes = []
        for i in train_set:
            for pat in include_patterns:
                target_prefixes.append(f"layers.{i}.{pat}")
    else:
        target_prefixes = [f"layers.{i}." for i in train_set]

    skip_lower = ("layernorm", "layer_norm", "rmsnorm", "norm")

    scaling = lora_alpha / lora_rank
    dropout_mod = nn.Dropout(lora_dropout) if lora_dropout > 0 else None

    injected = 0
    total_lora_params = 0

    for name, module in model.named_modules():
        if not any(pref in name for pref in target_prefixes):
            continue
        name_lower = name.lower().split(".")[-1]
        if any(s in name_lower for s in skip_lower):
            continue
        w = getattr(module, "weight", None)
        if w is None or not isinstance(w, (nn.Parameter, torch.Tensor)):
            continue
        if w.dim() != 2:
            continue

        out_features, in_features = w.shape[0], w.shape[1]
        lora_A = nn.Parameter(
            torch.empty(lora_rank, in_features, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        lora_B = nn.Parameter(
            torch.zeros(out_features, lora_rank, dtype=torch.bfloat16))

        module.register_parameter("lora_A", lora_A)
        module.register_parameter("lora_B", lora_B)

        def _make_lora_hook(s, drop):
            def _hook(mod, inp, out):
                x = inp[0] if isinstance(inp, tuple) else inp
                h = x.to(mod.lora_A.dtype) @ mod.lora_A.T
                if drop is not None:
                    h = drop(h)
                h = h @ mod.lora_B.T * s
                if isinstance(out, tuple):
                    return (out[0] + h,) + out[1:]
                return out + h
            return _hook

        module.register_forward_hook(_make_lora_hook(scaling, dropout_mod))
        injected += 1
        total_lora_params += lora_rank * (in_features + out_features)

        if rank == 0 and injected <= 5:
            print(f"  [LoRA] {name}: ({out_features}, {in_features}) -> r={lora_rank}",
                  flush=True)

    if rank == 0:
        if injected > 5:
            print(f"  [LoRA] ... and {injected - 5} more modules", flush=True)
        print(f"[LoRA] Injected {injected} adapters | r={lora_rank} a={lora_alpha} "
              f"dropout={lora_dropout} | {total_lora_params:,} LoRA params", flush=True)

    return injected


def mark_trainable(model, train_layer_indices, train_modules="attn",
                   use_lora=False):
    """Freeze all except trainable layers' selected modules."""
    if train_modules not in TRAIN_MODULE_PATTERNS:
        raise ValueError(
            f"Unknown --train_modules '{train_modules}'. "
            f"Choose from: {list(TRAIN_MODULE_PATTERNS.keys())}"
        )

    description = TRAIN_MODULE_PATTERNS[train_modules]["description"]
    train_set = set(train_layer_indices)
    total = 0
    trainable = 0

    if use_lora:
        for name, p in model.named_parameters():
            total += p.numel()
            if "lora_A" in name or "lora_B" in name:
                p.requires_grad = True
                trainable += p.numel()
            else:
                p.requires_grad = False
    else:
        include_patterns = TRAIN_MODULE_PATTERNS[train_modules]["include"]
        if include_patterns is not None:
            all_prefixes = []
            for i in train_set:
                for pat in include_patterns:
                    all_prefixes.append(f"layers.{i}.{pat}")
        else:
            all_prefixes = [f"layers.{i}." for i in train_set]

        for name, p in model.named_parameters():
            total += p.numel()
            match = any(pref in name for pref in all_prefixes)
            if match:
                p.requires_grad = True
                trainable += p.numel()
            else:
                p.requires_grad = False

    mode_str = f"lora({train_modules})" if use_lora else train_modules
    layers_str = ",".join(str(i) for i in sorted(train_set))
    print(f"[INFO] trainable({mode_str})={trainable:,} / total={total:,} "
          f"({trainable/total*100:.4f}%) [layers {layers_str}, "
          f"modules={description}]", flush=True)

    return trainable


# ============================================================
# DPO Dataset
# ============================================================

class DPOMapDataset(Dataset):
    """Map-style dataset for DPO training."""

    def __init__(self, path: str, tok, seq_len: int):
        self.tok = tok
        self.seq_len = seq_len
        self.samples: List[Dict] = []

        self.ref_chosen_logps: Optional[List[float]] = None
        self.ref_rejected_logps: Optional[List[float]] = None

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                result = self._tokenize_pair(obj)
                if result is not None:
                    self.samples.append(result)

        print(f"[DPO-DS] Loaded {len(self.samples)} pairs from {path}", flush=True)

    def _build_messages(self, prompt: str, response: str):
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]

    def _tokenize_with_labels(self, messages) -> Optional[Dict]:
        """Tokenise message list; mask non-assistant tokens in labels."""
        try:
            formatted = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            try:
                formatted = self.tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                return None
        except Exception:
            return None

        if not formatted:
            return None

        enc = self.tok(formatted, add_special_tokens=False)
        input_ids = enc["input_ids"]
        if not input_ids:
            return None

        labels = [-100] * len(input_ids)

        assistant_header = "<|im_start|>assistant\n"
        im_end_str = "<|im_end|>"
        pos = 0
        while True:
            start = formatted.find(assistant_header, pos)
            if start == -1:
                break
            content_start = start + len(assistant_header)
            end = formatted.find(im_end_str, content_start)
            if end == -1:
                end = len(formatted)
            content_end = end + len(im_end_str)

            prefix_before = formatted[:content_start]
            prefix_after = formatted[:content_end]
            tok_start = len(self.tok(prefix_before, add_special_tokens=False)["input_ids"])
            tok_end = len(self.tok(prefix_after, add_special_tokens=False)["input_ids"])

            for j in range(tok_start, min(tok_end, len(labels))):
                labels[j] = input_ids[j]
            pos = content_end

        if all(l == -100 for l in labels):
            labels = list(input_ids)

        input_ids = input_ids[:self.seq_len]
        labels = labels[:self.seq_len]

        return {"input_ids": input_ids, "labels": labels}

    def _tokenize_pair(self, obj: dict) -> Optional[Dict]:
        prompt = obj.get("prompt")
        chosen = obj.get("chosen")
        rejected = obj.get("rejected")

        if not prompt or not chosen or not rejected:
            return None

        chosen_msgs = self._build_messages(prompt, chosen)
        rejected_msgs = self._build_messages(prompt, rejected)

        chosen_tok = self._tokenize_with_labels(chosen_msgs)
        rejected_tok = self._tokenize_with_labels(rejected_msgs)

        if chosen_tok is None or rejected_tok is None:
            return None

        n_chosen_train = sum(1 for l in chosen_tok["labels"] if l != -100)
        n_rejected_train = sum(1 for l in rejected_tok["labels"] if l != -100)
        if n_chosen_train == 0 or n_rejected_train == 0:
            return None

        return {
            "chosen_input_ids": chosen_tok["input_ids"],
            "chosen_labels": chosen_tok["labels"],
            "rejected_input_ids": rejected_tok["input_ids"],
            "rejected_labels": rejected_tok["labels"],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        result = {
            "chosen_input_ids": s["chosen_input_ids"],
            "chosen_labels": s["chosen_labels"],
            "rejected_input_ids": s["rejected_input_ids"],
            "rejected_labels": s["rejected_labels"],
            "index": idx,
        }
        if self.ref_chosen_logps is not None:
            result["ref_chosen_logps"] = self.ref_chosen_logps[idx]
            result["ref_rejected_logps"] = self.ref_rejected_logps[idx]
        return result


def dpo_collate_fn(batch, pad_id, seq_len):
    """Collate DPO pairs — chosen and rejected padded to same length."""
    all_c_ids = [b["chosen_input_ids"] for b in batch]
    all_c_labels = [b["chosen_labels"] for b in batch]
    all_r_ids = [b["rejected_input_ids"] for b in batch]
    all_r_labels = [b["rejected_labels"] for b in batch]

    # Single max_len across BOTH chosen AND rejected
    c_max = max(len(ids) for ids in all_c_ids)
    r_max = max(len(ids) for ids in all_r_ids)
    max_len = min(max(c_max, r_max), seq_len)

    def _pad_to(all_ids, all_labels):
        ids_t = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        mask_t = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels_t = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, (ids, labs) in enumerate(zip(all_ids, all_labels)):
            slen = min(len(ids), max_len)
            ids_t[i, :slen] = torch.tensor(ids[:slen], dtype=torch.long)
            mask_t[i, :slen] = 1
            labels_t[i, :slen] = torch.tensor(labs[:slen], dtype=torch.long)

        return ids_t, mask_t, labels_t

    c_ids, c_mask, c_labels = _pad_to(all_c_ids, all_c_labels)
    r_ids, r_mask, r_labels = _pad_to(all_r_ids, all_r_labels)

    result = {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "chosen_labels": c_labels,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels": r_labels,
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }

    if "ref_chosen_logps" in batch[0]:
        result["ref_chosen_logps"] = torch.tensor(
            [b["ref_chosen_logps"] for b in batch], dtype=torch.float32)
        result["ref_rejected_logps"] = torch.tensor(
            [b["ref_rejected_logps"] for b in batch], dtype=torch.float32)

    return result


def _cross_entropy_no_save(logits_flat: torch.Tensor, labels_flat: torch.Tensor) -> torch.Tensor:
    """F.cross_entropy with reduction='none', wrapped for torch.utils.checkpoint."""
    return F.cross_entropy(logits_flat, labels_flat, reduction='none')


def compute_per_sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute sum of log-probs over non-masked positions per sequence."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    mask = (shift_labels != -100)
    B, T, V = shift_logits.shape

    per_token_ce = torch.utils.checkpoint.checkpoint(
        _cross_entropy_no_save,
        shift_logits.view(-1, V),
        shift_labels.clamp(min=0).view(-1),
        use_reentrant=False,
    ).view(B, T)

    per_token_logps = -per_token_ce * mask.float()
    return per_token_logps.sum(dim=-1)


def compute_per_sequence_logps_with_length(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Like compute_per_sequence_logps but also returns sequence lengths."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    mask = (shift_labels != -100)
    B, T, V = shift_logits.shape

    per_token_ce = torch.utils.checkpoint.checkpoint(
        _cross_entropy_no_save,
        shift_logits.view(-1, V),
        shift_labels.clamp(min=0).view(-1),
        use_reentrant=False,
    ).view(B, T)

    per_token_logps = -per_token_ce * mask.float()
    seq_logps = per_token_logps.sum(dim=-1)
    seq_lengths = mask.float().sum(dim=-1)
    return seq_logps, seq_lengths


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    length_normalise: bool = False,
    chosen_lengths: Optional[torch.Tensor] = None,
    rejected_lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss.

    Returns: (loss, chosen_rewards, rejected_rewards)
    """
    pc = policy_chosen_logps
    pr = policy_rejected_logps
    rc = ref_chosen_logps.to(pc.device)
    rr = ref_rejected_logps.to(pr.device)

    if length_normalise and chosen_lengths is not None and rejected_lengths is not None:
        pc = pc / chosen_lengths.clamp(min=1).to(pc.device)
        pr = pr / rejected_lengths.clamp(min=1).to(pr.device)
        rc = rc / chosen_lengths.clamp(min=1).to(rc.device)
        rr = rr / rejected_lengths.clamp(min=1).to(rr.device)

    chosen_rewards = beta * (pc - rc)
    rejected_rewards = beta * (pr - rr)

    if loss_type == "sigmoid":
        logits = chosen_rewards - rejected_rewards
        if label_smoothing > 0:
            loss = (
                (1 - label_smoothing) * F.logsigmoid(-logits)
                + label_smoothing * F.logsigmoid(logits)
            )
            loss = -loss.mean()
        else:
            loss = -F.logsigmoid(logits).mean()
    elif loss_type == "hinge":
        loss = torch.relu(1.0 - (chosen_rewards - rejected_rewards)).mean()
    elif loss_type == "ipo":
        logits = (pc - pr) - (rc - rr)
        loss = ((logits - 1.0 / (2.0 * beta)) ** 2).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, chosen_rewards.detach(), rejected_rewards.detach()


# ============================================================
# DPO Trainer
# ============================================================

class DPOTrainer:
    def __init__(self, args):
        set_seed(args.seed)
        configure_sdp_backends()
        self.args = args
        self.global_step = 0
        self.global_epoch = 0
        self.use_wandb = False

        # --------------------------------------------------
        # 1) Discover layer wrapping class
        # --------------------------------------------------
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)

        wrap_class = None
        for cls_name in [
            "Qwen3_5MoeDecoderLayer",
            "Qwen3_5MoeTextDecoderLayer",
            "Qwen3NextMoeDecoderLayer",
        ]:
            try:
                import transformers.models
                for attr_name in dir(transformers.models):
                    mod = getattr(transformers.models, attr_name, None)
                    if mod and hasattr(mod, cls_name):
                        wrap_class = getattr(mod, cls_name)
                        break
                if wrap_class:
                    break
            except Exception:
                continue

        if wrap_class is None:
            try:
                for mod_name in ["qwen3_5_moe", "qwen3_5", "qwen3_next_moe", "qwen3_next"]:
                    try:
                        mod = __import__(f"transformers.models.{mod_name}.modeling_{mod_name}",
                                         fromlist=["_"])
                        for name in dir(mod):
                            if "DecoderLayer" in name:
                                wrap_class = getattr(mod, name)
                                break
                        if wrap_class:
                            break
                    except ImportError:
                        continue
            except Exception:
                pass

        if wrap_class is not None:
            print(f"[FSDP] Auto-wrap class: {wrap_class.__name__}", flush=True)
        else:
            print("[WARN] Could not find decoder layer class. "
                  "Will discover from loaded model.", flush=True)

        # --------------------------------------------------
        # 2) FSDP / Accelerator setup
        # --------------------------------------------------
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=True,
            use_orig_params=True,
            activation_checkpointing=False,
            limit_all_gathers=True,
            cpu_offload=False,
        )

        if wrap_class is not None:
            fsdp_plugin.auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={wrap_class},
            )

        fsdp_plugin.mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        def _param_init_fn(module):
            has_meta = any(
                p.device.type == "meta"
                for p in module.parameters(recurse=False)
            ) or any(
                b is not None and b.device.type == "meta"
                for b in module.buffers(recurse=False)
            )
            if has_meta:
                module.to_empty(device=torch.device("cpu"), recurse=False)
            return module

        fsdp_plugin.param_init_fn = _param_init_fn

        pg_kwargs = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=7200))

        ga_plugin = GradientAccumulationPlugin(
            num_steps=args.gradient_accumulation_steps,
            sync_each_batch=True,
        )

        self.accelerator = Accelerator(
            mixed_precision="no",
            fsdp_plugin=fsdp_plugin,
            gradient_accumulation_plugin=ga_plugin,
            kwargs_handlers=[pg_kwargs],
        )

        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.device = self.accelerator.device

        if self.accelerator.is_main_process:
            print(f"[DIST] rank={self.rank} / world_size={self.world_size}", flush=True)

        # --------------------------------------------------
        # 3) Model loading (identical to SFT)
        # --------------------------------------------------
        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        if self.accelerator.is_main_process:
            print(f"[LOAD] model: {args.model}", flush=True)

        if self.rank == 0:
            print("[LOAD] rank 0: loading BF16 weights to CPU...", flush=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model, config=config, device_map=None,
                trust_remote_code=True, low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
            )
        else:
            print(f"[LOAD] rank {self.rank}: creating meta model (zero CPU RAM)...", flush=True)
            import transformers.modeling_utils as _tm_utils

            _orig_load = _tm_utils.PreTrainedModel._load_pretrained_model
            _orig_finalize = _tm_utils.PreTrainedModel._finalize_model_loading
            _orig_check = getattr(_tm_utils, "check_and_set_device_map", None)

            @classmethod
            def _noop_load(cls, model, *a, **kw):
                return {"missing_keys": [], "unexpected_keys": [],
                        "mismatched_keys": [], "error_msgs": []}, None

            @classmethod
            def _noop_finalize(cls, model, *a, **kw):
                return {"missing_keys": [], "unexpected_keys": [],
                        "mismatched_keys": [], "error_msgs": []}

            _tm_utils.PreTrainedModel._load_pretrained_model = _noop_load
            _tm_utils.PreTrainedModel._finalize_model_loading = _noop_finalize
            if _orig_check is not None:
                _tm_utils.check_and_set_device_map = lambda device_map, **kw: device_map

            try:
                with torch.device("meta"):
                    base_model = AutoModelForCausalLM.from_pretrained(
                        args.model, config=config, device_map=None,
                        trust_remote_code=True, low_cpu_mem_usage=True,
                        torch_dtype=torch.bfloat16,
                    )
            finally:
                _tm_utils.PreTrainedModel._load_pretrained_model = _orig_load
                _tm_utils.PreTrainedModel._finalize_model_loading = _orig_finalize
                if _orig_check is not None:
                    _tm_utils.check_and_set_device_map = _orig_check
            print(f"[LOAD] rank {self.rank}: meta model created.", flush=True)

        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

        # Discover wrap class from loaded model if needed
        if wrap_class is None:
            layers_list = None
            for mod in base_model.modules():
                if isinstance(mod, nn.ModuleList) and len(mod) == args.num_layers:
                    layers_list = mod
                    break
            if layers_list is not None and len(layers_list) > 0:
                wrap_class = type(layers_list[0])
                print(f"[FSDP] Discovered wrap class from model: {wrap_class.__name__}",
                      flush=True)
                fsdp_plugin.auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={wrap_class},
                )

        # Model structure diagnostic (rank 0)
        if self.rank == 0:
            print("\n[MODEL] Architecture diagnostic:", flush=True)
            for name, mod in base_model.named_modules():
                depth = name.count(".")
                if depth <= 2:
                    print(f"  {'  ' * depth}{name}: {type(mod).__name__}", flush=True)
                elif "layers.0." in name and depth <= 4:
                    short = name.replace("model.layers.0.", "L0.")
                    print(f"    {short}: {type(mod).__name__}", flush=True)
            dtype_counts = {}
            for _, p in base_model.named_parameters():
                dt = str(p.dtype)
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
            print(f"\n[MODEL] Parameter dtypes: {dtype_counts}", flush=True)

        base_model.train()

        # --------------------------------------------------
        # 4) Prepare trainable layers
        # --------------------------------------------------
        last_layer_idx = args.num_layers - 1
        train_layer_indices = list(range(
            last_layer_idx - args.train_layers + 1, last_layer_idx + 1))
        first_train_idx = train_layer_indices[0]
        self.train_layer_indices = train_layer_indices
        layers_str = ",".join(str(i) for i in train_layer_indices)

        if args.use_lora:
            if self.accelerator.is_main_process:
                print(f"[LoRA] Injecting adapters into layers [{layers_str}]...", flush=True)
            n_lora = inject_lora_adapters(
                base_model, train_layer_indices, args.train_modules,
                lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout, rank=self.rank,
            )
            if n_lora == 0:
                raise RuntimeError(
                    f"No LoRA adapters injected for layers {train_layer_indices}")

        trainable_count = mark_trainable(
            base_model, train_layer_indices, args.train_modules,
            use_lora=args.use_lora,
        )
        if trainable_count == 0:
            raise RuntimeError(f"No trainable params for layers {train_layer_indices}")

        # --------------------------------------------------
        # 5) FSDP wrap
        # --------------------------------------------------
        if self.rank != 0:
            for module in base_model.modules():
                for bname in list(module._buffers.keys()):
                    buf = module._buffers[bname]
                    if buf is not None and buf.device.type == "meta":
                        module._buffers[bname] = torch.empty(
                            buf.shape, dtype=buf.dtype, device="cpu")

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            print("[FSDP] Wrapping model (sync_module_states=True)...", flush=True)

        self.model = self.accelerator.prepare(base_model)
        self.is_fsdp = isinstance(self.model, FSDP)

        if self.accelerator.is_main_process:
            print("[FSDP] Model wrapped successfully.", flush=True)

        # --------------------------------------------------
        # 6) Post-FSDP detach hooks (identical to SFT)
        # --------------------------------------------------
        def _find_layers_list(model):
            for mod in model.modules():
                if isinstance(mod, nn.ModuleList) and len(mod) == args.num_layers:
                    return mod
            for attr_path in ["model.layers", "layers",
                              "_fsdp_wrapped_module.model.layers"]:
                obj = model
                try:
                    for part in attr_path.split("."):
                        obj = getattr(obj, part)
                    if isinstance(obj, nn.ModuleList):
                        return obj
                except AttributeError:
                    continue
            return None

        fsdp_layers = _find_layers_list(self.model)
        fsdp_train_layer = None

        if fsdp_layers is not None and first_train_idx < len(fsdp_layers):
            fsdp_train_layer = fsdp_layers[first_train_idx]
            if self.accelerator.is_main_process:
                print(f"[DETACH] Found layers ({len(fsdp_layers)} layers), "
                      f"target: index {first_train_idx} -> {type(fsdp_train_layer).__name__}",
                      flush=True)

        if fsdp_train_layer is not None:
            def _detach_and_enable_grad(module, args_tuple):
                new_args = []
                for a in args_tuple:
                    if isinstance(a, torch.Tensor):
                        new_args.append(a.detach().requires_grad_(True))
                    else:
                        new_args.append(a)
                return tuple(new_args)

            fsdp_train_layer.register_forward_pre_hook(_detach_and_enable_grad)
            if self.accelerator.is_main_process:
                print(f"[DETACH] Installed detach+grad hook on layer {first_train_idx} "
                      f"(frozen layers 0-{first_train_idx-1} run without grad)", flush=True)

        if self.accelerator.is_main_process:
            print(f"[DETACH] Summary: layer-input detach="
                  f"{'OK' if fsdp_train_layer else 'FAILED'}", flush=True)

        # MLP output detach hooks (prevent backward through MoE).
        # Only for attn-only training: in "mlp" mode this hook would cut
        # gradients to the very modules being trained.
        mlp_detach_count = 0
        if args.train_modules == "attn":
            for tidx in train_layer_indices:
                layer_mod = fsdp_layers[tidx] if fsdp_layers is not None else None
                if layer_mod is None:
                    continue
                inner = getattr(layer_mod, '_fsdp_wrapped_module', layer_mod)
                inner = getattr(inner, '_checkpoint_wrapped_module', inner)
                mlp_module = None
                for attr_name in ("mlp", "moe", "feed_forward", "ffn",
                                  "block_sparse_moe", "shared_experts"):
                    mlp_module = getattr(inner, attr_name, None)
                    if mlp_module is not None:
                        break
                if mlp_module is not None:
                    def _make_mlp_detach(idx):
                        def _hook(module, input, output):
                            if isinstance(output, torch.Tensor) and output.requires_grad:
                                return output.detach().requires_grad_(True)
                            elif isinstance(output, tuple):
                                new = []
                                for o in output:
                                    if isinstance(o, torch.Tensor) and o.requires_grad:
                                        new.append(o.detach().requires_grad_(True))
                                    else:
                                        new.append(o)
                                return tuple(new)
                            return output
                        return _hook
                    mlp_module.register_forward_hook(_make_mlp_detach(tidx))
                    mlp_detach_count += 1
                    if self.accelerator.is_main_process:
                        print(f"[DETACH] MLP output detach on layer {tidx} "
                              f"({type(mlp_module).__name__})", flush=True)

        if self.accelerator.is_main_process and mlp_detach_count > 0:
            print(f"[DETACH] MLP detach: {mlp_detach_count} layers", flush=True)

        # --------------------------------------------------
        # 7) Optimizer
        # --------------------------------------------------
        trainable_params = [p for _, p in self.model.named_parameters()
                           if p.requires_grad and p.numel() > 0]

        if self.accelerator.is_main_process:
            print(f"[OPT] Trainable params for optimizer: {len(trainable_params)}", flush=True)

        has_params = len(trainable_params) > 0
        if self.accelerator.is_main_process and not has_params:
            raise RuntimeError("No trainable parameters on rank 0 after FSDP wrapping.")

        if not trainable_params:
            _dummy = nn.Parameter(torch.zeros(1, device=self.device, dtype=torch.bfloat16))
            _dummy.requires_grad = True
            trainable_params = [_dummy]
            self._dummy_param = _dummy

        self.opt = torch.optim.AdamW(
            trainable_params, lr=args.lr, betas=(0.9, 0.98),
            weight_decay=args.weight_decay, foreach=False,
        )
        self.opt = self.accelerator.prepare(self.opt)

        # Pad token
        if self.tok.pad_token_id is not None:
            self.pad_token_id = int(self.tok.pad_token_id)
        elif self.tok.eos_token_id is not None:
            self.pad_token_id = int(self.tok.eos_token_id)
        else:
            raise RuntimeError("Tokenizer has no pad/eos token id.")

        # --------------------------------------------------
        # 8) Dataset
        # --------------------------------------------------
        if self.accelerator.is_main_process:
            print(f"[DATA] dataset: {args.dataset}", flush=True)

        self.dataset = DPOMapDataset(args.dataset, tok=self.tok, seq_len=args.seq_len)

        if self.accelerator.is_main_process and len(self.dataset) > 0:
            s0 = self.dataset.samples[0]
            for side in ("chosen", "rejected"):
                ids = s0[f"{side}_input_ids"]
                labs = s0[f"{side}_labels"]
                n_train = sum(1 for l in labs if l != -100)
                n_mask = sum(1 for l in labs if l == -100)
                print(f"[DATA] sample 0 {side}: {len(ids)} tokens, "
                      f"{n_train} trainable, {n_mask} masked", flush=True)

        # --------------------------------------------------
        # 9) Precompute reference log-probs
        # --------------------------------------------------
        if self.accelerator.is_main_process:
            print(f"\n[REF] Precomputing reference log-probs "
                  f"({len(self.dataset)} pairs)...", flush=True)

        self._precompute_ref_logprobs()

        if self.accelerator.is_main_process:
            rc = self.dataset.ref_chosen_logps
            rr = self.dataset.ref_rejected_logps
            avg_c = sum(rc) / max(len(rc), 1)
            avg_r = sum(rr) / max(len(rr), 1)
            print(f"[REF] Done. avg ref_chosen_logp={avg_c:.2f}, "
                  f"avg ref_rejected_logp={avg_r:.2f}", flush=True)

        # --------------------------------------------------
        # 10) Training DataLoader
        # --------------------------------------------------
        self.train_sampler = DistributedSampler(
            self.dataset, num_replicas=self.world_size, rank=self.rank,
            shuffle=True, seed=args.seed, drop_last=True,
        )
        _collate = partial(dpo_collate_fn, pad_id=self.pad_token_id, seq_len=args.seq_len)
        self.loader = DataLoader(
            self.dataset, batch_size=args.batch_size,
            sampler=self.train_sampler, collate_fn=_collate,
            num_workers=2, pin_memory=True, drop_last=True,
        )

        gas = args.gradient_accumulation_steps
        eff_batch = args.batch_size * gas * self.world_size
        samples_per_rank = len(self.dataset) // self.world_size
        micro_per_epoch = max(1, samples_per_rank // args.batch_size)
        optim_per_epoch = micro_per_epoch // gas
        total_steps = optim_per_epoch * (args.epochs or 1)
        self.estimated_total_steps = max(1, total_steps)

        if self.accelerator.is_main_process:
            print(f"[DATA] total_pairs={len(self.dataset)}, "
                  f"micro_batches/epoch~={micro_per_epoch}, "
                  f"optim_steps/epoch~={optim_per_epoch}, "
                  f"estimated_total_steps~={total_steps}, "
                  f"effective_batch_size={eff_batch}", flush=True)

        # Summary
        if self.accelerator.is_main_process:
            layer_types = getattr(text_config, "layer_types", [])
            train_layer_types = [
                layer_types[i] if i < len(layer_types) else "unknown"
                for i in train_layer_indices
            ]

            print("=" * 56)
            print(f"  Qwen3.5 Partial-Layer DPO "
                  f"({'LoRA' if args.use_lora else 'Full'})")
            print(f"  model        : {args.model}")
            print(f"  num_layers   : {args.num_layers}")
            print(f"  train_layers : {layers_str} ({len(train_layer_indices)} layers)")
            print(f"  layer_types  : {train_layer_types}")
            print(f"  train_modules: {args.train_modules} "
                  f"({TRAIN_MODULE_PATTERNS[args.train_modules]['description']})")
            print(f"  dataset      : {args.dataset}")
            print(f"  batch_size   : {args.batch_size} (micro, per rank)")
            print(f"  grad_accum   : {gas}")
            print(f"  eff_batch    : {eff_batch}")
            print(f"  seq_len      : {args.seq_len}")
            print(f"  beta         : {args.beta}")
            print(f"  loss_type    : {args.loss_type}")
            print(f"  label_smooth : {args.label_smoothing}")
            print(f"  len_normalise: {args.length_normalise}")
            print(f"  lr           : {args.lr}")
            print(f"  epochs       : {args.epochs}")
            print(f"  steps(cap)   : {args.steps}")
            print(f"  save_dir     : {args.save_dir}")
            print("=" * 56, flush=True)

        # W&B
        if self.accelerator.is_main_process and args.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name or None,
                    config={
                        "model": args.model,
                        "num_layers": args.num_layers,
                        "train_layers": train_layer_indices,
                        "train_modules": args.train_modules,
                        "use_lora": args.use_lora,
                        "lora_rank": args.lora_rank if args.use_lora else None,
                        "dataset": args.dataset,
                        "batch_size": args.batch_size,
                        "grad_accum": gas,
                        "eff_batch": eff_batch,
                        "seq_len": args.seq_len,
                        "beta": args.beta,
                        "loss_type": args.loss_type,
                        "label_smoothing": args.label_smoothing,
                        "length_normalise": args.length_normalise,
                        "lr": args.lr,
                        "epochs": args.epochs,
                        "world_size": self.world_size,
                    },
                    resume="allow",
                )
                self.use_wandb = True
                print(f"[WANDB] Initialized: project={args.wandb_project}, "
                      f"run={wandb.run.name}", flush=True)
            except Exception as e:
                print(f"[WANDB] Init failed: {e}. Continuing without wandb.", flush=True)

    # ==========================================================
    # Reference log-prob precomputation
    # ==========================================================

    @torch.no_grad()
    def _precompute_ref_logprobs(self):
        """Compute reference log-probs using concatenated forward passes."""
        self.model.eval()

        n = len(self.dataset)
        n_padded = math.ceil(n / self.world_size) * self.world_size
        all_indices = list(range(n)) + [0] * (n_padded - n)

        per_rank = n_padded // self.world_size
        my_start = self.rank * per_rank
        my_end = my_start + per_rank
        my_indices = all_indices[my_start:my_end]

        ref_bs = 1
        _collate = partial(dpo_collate_fn, pad_id=self.pad_token_id, seq_len=self.args.seq_len)

        local_results = {}

        t0 = time.time()
        num_batches = math.ceil(len(my_indices) / ref_bs)

        for bi in range(num_batches):
            batch_indices = my_indices[bi * ref_bs : (bi + 1) * ref_bs]
            batch_items = [self.dataset[i] for i in batch_indices]
            batch = _collate(batch_items)

            c_ids = batch["chosen_input_ids"].to(self.device, non_blocking=True)
            c_mask = batch["chosen_attention_mask"].to(self.device, non_blocking=True)
            c_labels = batch["chosen_labels"].to(self.device, non_blocking=True)
            r_ids = batch["rejected_input_ids"].to(self.device, non_blocking=True)
            r_mask = batch["rejected_attention_mask"].to(self.device, non_blocking=True)
            r_labels = batch["rejected_labels"].to(self.device, non_blocking=True)

            all_ids = torch.cat([c_ids, r_ids], dim=0)
            all_mask = torch.cat([c_mask, r_mask], dim=0)
            all_labels = torch.cat([c_labels, r_labels], dim=0)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=all_ids, attention_mask=all_mask,
                    use_cache=False, output_router_logits=False, return_dict=True,
                )
            all_logps = compute_per_sequence_logps(outputs.logits, all_labels)
            bs = c_ids.size(0)
            c_logps = all_logps[:bs]
            r_logps = all_logps[bs:]
            del outputs, all_ids, all_mask, all_labels, all_logps

            for j, idx in enumerate(batch_indices):
                if idx < n:
                    local_results[idx] = (c_logps[j].item(), r_logps[j].item())

            if self.accelerator.is_main_process and (bi + 1) % max(1, num_batches // 5) == 0:
                elapsed = time.time() - t0
                print(f"  [REF] {bi+1}/{num_batches} batches, "
                      f"elapsed={elapsed:.1f}s", flush=True)

        # All-gather
        all_results = [None] * self.world_size
        dist.all_gather_object(all_results, local_results)

        # Merge
        ref_chosen = [0.0] * n
        ref_rejected = [0.0] * n
        for rank_dict in all_results:
            for idx, (c, r) in rank_dict.items():
                ref_chosen[idx] = c
                ref_rejected[idx] = r

        self.dataset.ref_chosen_logps = ref_chosen
        self.dataset.ref_rejected_logps = ref_rejected


        if self.accelerator.is_main_process:
            elapsed = time.time() - t0
            print(f"  [REF] Precomputation finished in {elapsed:.1f}s", flush=True)

    # ==========================================================
    # Training step
    # ==========================================================

    def step(self, batch):
        self.model.eval()

        c_ids = batch["chosen_input_ids"].to(self.device, non_blocking=True)
        c_mask = batch["chosen_attention_mask"].to(self.device, non_blocking=True)
        c_labels = batch["chosen_labels"].to(self.device, non_blocking=True)
        r_ids = batch["rejected_input_ids"].to(self.device, non_blocking=True)
        r_mask = batch["rejected_attention_mask"].to(self.device, non_blocking=True)
        r_labels = batch["rejected_labels"].to(self.device, non_blocking=True)
        ref_c = batch["ref_chosen_logps"].to(self.device, non_blocking=True)
        ref_r = batch["ref_rejected_logps"].to(self.device, non_blocking=True)

        if self._micro_step == 0:
            torch.cuda.reset_peak_memory_stats()

        with self.accelerator.accumulate(self.model):
            all_ids = torch.cat([c_ids, r_ids], dim=0)
            all_mask = torch.cat([c_mask, r_mask], dim=0)
            all_labels = torch.cat([c_labels, r_labels], dim=0)
            B = c_ids.size(0)
            del c_ids, r_ids, c_mask, r_mask, c_labels, r_labels

            torch.cuda.empty_cache()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=all_ids, attention_mask=all_mask,
                    use_cache=False, output_router_logits=False, return_dict=True,
                )
            del all_ids, all_mask

            logits = outputs.logits
            del outputs

            if self.args.length_normalise:
                all_logps, all_lengths = compute_per_sequence_logps_with_length(
                    logits, all_labels)
                pc_logps = all_logps[:B]
                pr_logps = all_logps[B:]
                c_lengths = all_lengths[:B]
                r_lengths = all_lengths[B:]
            else:
                all_logps = compute_per_sequence_logps(logits, all_labels)
                pc_logps = all_logps[:B]
                pr_logps = all_logps[B:]
                c_lengths = r_lengths = None
            del logits, all_labels, all_logps

            if self._micro_step == 0 and self.accelerator.is_local_main_process:
                _alloc = torch.cuda.memory_allocated() / 1e9
                _peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[MEM] after fwd+logps: "
                      f"alloc={_alloc:.1f}GB peak={_peak:.1f}GB", flush=True)

            loss, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps=pc_logps,
                policy_rejected_logps=pr_logps,
                ref_chosen_logps=ref_c,
                ref_rejected_logps=ref_r,
                beta=self.args.beta,
                label_smoothing=self.args.label_smoothing,
                loss_type=self.args.loss_type,
                length_normalise=self.args.length_normalise,
                chosen_lengths=c_lengths,
                rejected_lengths=r_lengths if self.args.length_normalise else None,
            )
            del pc_logps, pr_logps

            torch.cuda.empty_cache()
            self.accelerator.backward(loss)

            if self._micro_step == 0 and self.accelerator.is_local_main_process:
                _alloc = torch.cuda.memory_allocated() / 1e9
                _peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"[MEM] after backward: "
                      f"alloc={_alloc:.1f}GB peak={_peak:.1f}GB", flush=True)

            if self.accelerator.sync_gradients and self.args.grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    [p for _, p in self.model.named_parameters() if p.requires_grad],
                    self.args.grad_clip,
                )
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        loss_val = loss.item()
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        self._micro_step += 1

        if not (loss_val != loss_val):
            self._micro_loss_accum += loss_val
            self._micro_margin_accum += reward_margin
            self._micro_acc_accum += accuracy
            self._micro_count += 1

        gas = self.args.gradient_accumulation_steps
        if self._micro_step % gas == 0:
            self.global_step += 1
            current_step = self.global_step

            avg_loss = self._micro_loss_accum / max(1, self._micro_count)
            avg_margin = self._micro_margin_accum / max(1, self._micro_count)
            avg_acc = self._micro_acc_accum / max(1, self._micro_count)

            self._epoch_loss_accum += avg_loss
            self._epoch_margin_accum += avg_margin
            self._epoch_acc_accum += avg_acc
            self._epoch_count += 1

            self._micro_loss_accum = 0.0
            self._micro_margin_accum = 0.0
            self._micro_acc_accum = 0.0
            self._micro_count = 0

            step_end = time.time()
            self._step_times.append(step_end)

            if current_step % self.args.log_every == 0 and self.accelerator.is_main_process:
                elapsed = step_end - self._train_start_time if self._train_start_time else 0
                run_avg_loss = self._epoch_loss_accum / max(1, self._epoch_count)

                if len(self._step_times) >= 2:
                    window = min(50, len(self._step_times))
                    recent = self._step_times[-1] - self._step_times[-window]
                    steps_per_sec = (window - 1) / max(recent, 1e-6)
                    sec_per_step = 1.0 / max(steps_per_sec, 1e-6)
                else:
                    steps_per_sec = sec_per_step = 0

                eta_str = "?"
                pct_str = ""
                if self.estimated_total_steps > 0:
                    pct = current_step / self.estimated_total_steps * 100
                    pct_str = f" ({pct:.1f}%)"
                    remaining = max(0, self.estimated_total_steps - current_step)
                    if steps_per_sec > 0:
                        eta_sec = remaining / steps_per_sec
                        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

                print(
                    f"  step {current_step}{pct_str} | "
                    f"loss={avg_loss:.4f} (avg={run_avg_loss:.4f}) | "
                    f"margin={avg_margin:.3f} | acc={avg_acc:.3f} | "
                    f"lr={self.opt.param_groups[0]['lr']:.2e} | "
                    f"{sec_per_step:.1f}s/step | "
                    f"elapsed={str(datetime.timedelta(seconds=int(elapsed)))} | "
                    f"ETA={eta_str}",
                    flush=True,
                )

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/avg_loss": run_avg_loss,
                        "train/reward_margin": avg_margin,
                        "train/accuracy": avg_acc,
                        "train/lr": self.opt.param_groups[0]["lr"],
                        "train/epoch": self.global_epoch,
                        "perf/sec_per_step": sec_per_step,
                    }, step=current_step)

            if (self.args.save_every and current_step % self.args.save_every == 0
                    and current_step > 0):
                self.save(os.path.join(self.args.save_dir, f"step-{current_step}"))

    def _should_stop(self):
        return (self.args.steps and self.args.steps > 0
                and self.global_step >= self.args.steps)

    # ==========================================================
    # Save (identical to SFT)
    # ==========================================================

    @torch.no_grad()
    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        def clean_key(k):
            k = k.replace("_fsdp_wrapped_module.", "")
            k = k.replace("_checkpoint_wrapped_module.", "")
            k = k.replace("module.", "")
            if k.startswith("model."):
                k = k[6:]
            return k

        if self.is_fsdp:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                sd = self.model.state_dict()

            unwrapped = self.accelerator.unwrap_model(self.model)
            trainable_names = {
                clean_key(n) for n, p in unwrapped.named_parameters()
                if p.requires_grad
            }
            sub = {k: v for k, v in sd.items()
                   if clean_key(k) in trainable_names}

            path = os.path.join(save_dir, f"params.rank{self.rank}.pt")
            torch.save(sub, path)

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                meta = {
                    "format": "dpo_lora_sharded" if self.args.use_lora else "dpo_partial_layer_sharded",
                    "pattern": "params.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "model": self.args.model,
                    "train_layers": list(self.train_layer_indices),
                    "train_modules": self.args.train_modules,
                    "use_lora": self.args.use_lora,
                    "lora_rank": self.args.lora_rank if self.args.use_lora else None,
                    "lora_alpha": self.args.lora_alpha if self.args.use_lora else None,
                    "beta": self.args.beta,
                    "loss_type": self.args.loss_type,
                    "matched_keys": len(sub),
                }
                with open(os.path.join(save_dir, "params_meta.json"), "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"[SAVE] -> {save_dir} ({len(sub)} keys)", flush=True)
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped = self.accelerator.unwrap_model(self.model)
                sub = {n: p.detach().cpu()
                       for n, p in unwrapped.named_parameters() if p.requires_grad}
                path = os.path.join(save_dir, "params.pt")
                torch.save(sub, path)
                print(f"[SAVE] -> {path} ({len(sub)} keys)", flush=True)

    # ==========================================================
    # Training loop
    # ==========================================================

    def train(self):
        self.model.train()
        self._train_start_time = time.time()
        self._step_times = [self._train_start_time]
        self._epoch_loss_accum = 0.0
        self._epoch_margin_accum = 0.0
        self._epoch_acc_accum = 0.0
        self._epoch_count = 0
        self._micro_step = 0
        self._micro_loss_accum = 0.0
        self._micro_margin_accum = 0.0
        self._micro_acc_accum = 0.0
        self._micro_count = 0

        if self.args.epochs and self.args.epochs > 0:
            for ep in range(1, self.args.epochs + 1):
                self.global_epoch = ep
                self.train_sampler.set_epoch(ep)
                epoch_start = time.time()
                epoch_step_start = self.global_step

                self._epoch_loss_accum = 0.0
                self._epoch_margin_accum = 0.0
                self._epoch_acc_accum = 0.0
                self._epoch_count = 0

                if self.accelerator.is_main_process:
                    print(f"\n[EPOCH] start {ep}/{self.args.epochs}", flush=True)

                for batch in self.loader:
                    self.step(batch)
                    if self._should_stop():
                        break

                if self.accelerator.is_main_process:
                    epoch_elapsed = time.time() - epoch_start
                    epoch_steps = self.global_step - epoch_step_start
                    avg_loss = self._epoch_loss_accum / max(1, self._epoch_count)
                    avg_margin = self._epoch_margin_accum / max(1, self._epoch_count)
                    avg_acc = self._epoch_acc_accum / max(1, self._epoch_count)
                    print(
                        f"\n[EPOCH] finished {ep}/{self.args.epochs} "
                        f"| steps={epoch_steps} "
                        f"| avg_loss={avg_loss:.4f} "
                        f"| avg_margin={avg_margin:.3f} "
                        f"| avg_acc={avg_acc:.3f} "
                        f"| time={str(datetime.timedelta(seconds=int(epoch_elapsed)))}",
                        flush=True,
                    )
                if self._should_stop():
                    break

        if not self.args.epochs or self.args.epochs <= 0:
            if self.accelerator.is_main_process:
                print(f"\n[TRAIN] step-only mode (steps={self.args.steps})", flush=True)
            while not self._should_stop():
                self.train_sampler.set_epoch(self.global_step)
                for batch in self.loader:
                    self.step(batch)
                    if self._should_stop():
                        break

        if self.accelerator.is_main_process:
            total_elapsed = time.time() - self._train_start_time
            avg_loss = self._epoch_loss_accum / max(1, self._epoch_count)
            print(
                f"\n[DONE] steps={self.global_step} "
                f"| avg_loss={avg_loss:.4f} "
                f"| total_time={str(datetime.timedelta(seconds=int(total_elapsed)))}",
                flush=True,
            )

        self.save(self.args.save_dir)

        if self.use_wandb:
            import wandb
            wandb.finish()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 Partial-Layer DPO")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=60)
    parser.add_argument("--train_layers", type=int, default=1)
    parser.add_argument("--train_modules", type=str, default="attn",
                        choices=list(TRAIN_MODULE_PATTERNS.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Micro batch size per rank")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./output/qwen35-lastlayer-dpo")
    parser.add_argument("--seed", type=int, default=42)
    # DPO-specific
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo"])
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--length_normalise", action="store_true")
    parser.add_argument("--ref_batch_size", type=int, default=1,
                        help="Ignored (forced to 1 for MoE routing consistency)")
    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    # W&B
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    args = parser.parse_args()

    trainer = DPOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
