#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Partial-layer SFT for Qwen3.5-397B-A17B (FSDP, BF16).

Supports:
  - Full fine-tuning of selected modules (attn, mlp, or full layer)
  - LoRA adaptation (--use_lora) — base weights stay frozen
  - Configurable number of trainable layers (--train_layers N)
  - Sequence packing (--packing) for GPU utilization
  - Gradient accumulation (--gradient_accumulation_steps)
  - W&B logging (--wandb_project)

Model: Qwen3.5-397B-A17B
  - 60 layers, hybrid attention (45 Gated DeltaNet + 15 Gated Attention, 3:1)
  - 512 MoE experts (10 routed + 1 shared), expert_dim=1024
  - BF16 native — no FP8 dequant needed
  - Vision encoder (ViT) — frozen for text SFT

Loading strategy:
  - rank 0: from_pretrained() loads BF16 weights to CPU (~794GB)
  - rank != 0: meta context → zero CPU RAM
  - FSDP sync_module_states=True broadcasts from rank 0
"""

import os
import json
import math
import random
import argparse
import datetime
import time
from typing import List, Tuple
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs
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
# Dataset: On-the-fly ChatML tokenization
# ============================================================

def parse_chatml(text):
    """Parse ChatML format into list of (role, content)."""
    messages = []
    for block in text.split("<|im_start|>"):
        block = block.strip()
        if not block:
            continue
        block = block.replace("<|im_end|>", "").strip()
        if "\n" in block:
            role, content = block.split("\n", 1)
            messages.append({"role": role.strip(), "content": content.strip()})
        else:
            messages.append({"role": block.strip(), "content": ""})
    return messages


class OnTheFlyTokenizedChatMLJsonl(IterableDataset):
    """Reads .jsonl with ChatML text, tokenizes on-the-fly."""

    def __init__(self, path, tok, seq_len, rank=0, world_size=1):
        self.path = path
        self.tok = tok
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size

        with open(path, "r", encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if l.strip()]
        self.total = len(self.lines)

    def _tokenize_line(self, line):
        """Tokenize a single JSONL line with manual assistant-turn masking.

        Qwen3.5 does NOT support return_assistant_tokens_mask (the Qwen team
        explicitly rejected adding {% generation %} to the chat template).

        Approach (Qwen-recommended: tokenize=False + framework-level masking):
        1. Use tokenize=False + enable_thinking=False to get formatted text
        2. Find assistant turn boundaries via <|im_start|>assistant / <|im_end|>
        3. Tokenize the full string, build labels mask from char→token mapping
        """
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None

        text = obj.get("text", None)
        messages = obj.get("messages", None)

        if messages:
            pass
        elif text:
            messages = parse_chatml(text)
        else:
            return None

        if not messages:
            return None

        # Step 1: Get formatted text string (no tokenization yet)
        try:
            formatted = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizer without enable_thinking support
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

        # Step 2: Tokenize the full formatted text
        enc = self.tok(formatted, add_special_tokens=False)
        input_ids = enc["input_ids"]
        if not input_ids:
            return None

        # Step 3: Build assistant mask from string positions
        # Qwen ChatML format:
        #   <|im_start|>assistant\n{content}<|im_end|>\n
        # We mark {content}<|im_end|> as trainable labels (not the header)
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
            # Include <|im_end|> token as part of trainable
            content_end = end + len(im_end_str)

            # Convert char positions → token positions via prefix tokenization
            prefix_before = formatted[:content_start]
            prefix_after = formatted[:content_end]
            tok_start = len(self.tok(prefix_before, add_special_tokens=False)["input_ids"])
            tok_end = len(self.tok(prefix_after, add_special_tokens=False)["input_ids"])

            for j in range(tok_start, min(tok_end, len(labels))):
                labels[j] = input_ids[j]

            pos = content_end

        # Sanity: if no assistant tokens found, train on everything
        if all(l == -100 for l in labels):
            labels = list(input_ids)

        return {"input_ids": input_ids, "labels": labels}

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            per_worker = len(self.lines) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.lines)
            lines = self.lines[start:end]
        else:
            lines = self.lines

        for idx, line in enumerate(lines):
            if idx % self.world_size != self.rank:
                continue
            result = self._tokenize_line(line)
            if result is None:
                continue
            ids = result["input_ids"][:self.seq_len]
            labs = result["labels"][:self.seq_len]
            yield {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labs, dtype=torch.long),
                "length": len(ids),
            }


class PackedDataset(IterableDataset):
    """Packs multiple samples into fixed-length sequences."""

    def __init__(self, base_dataset, seq_len, batch_size, pad_token_id):
        self.base = base_dataset
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pad_id = pad_token_id

    def _pack_iter(self):
        buf_ids, buf_labels = [], []
        for sample in self.base:
            ids = sample["input_ids"].tolist()
            labs = sample["labels"].tolist()
            buf_ids.extend(ids)
            buf_labels.extend(labs)
            while len(buf_ids) >= self.seq_len:
                chunk_ids = buf_ids[:self.seq_len]
                chunk_labels = buf_labels[:self.seq_len]
                buf_ids = buf_ids[self.seq_len:]
                buf_labels = buf_labels[self.seq_len:]
                yield {
                    "input_ids": torch.tensor(chunk_ids, dtype=torch.long),
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                    "labels": torch.tensor(chunk_labels, dtype=torch.long),
                }
        # Flush remaining with padding
        if buf_ids:
            pad_len = self.seq_len - len(buf_ids)
            buf_ids.extend([self.pad_id] * pad_len)
            buf_labels.extend([-100] * pad_len)
            yield {
                "input_ids": torch.tensor(buf_ids[:self.seq_len], dtype=torch.long),
                "attention_mask": torch.tensor(
                    [1] * (self.seq_len - pad_len) + [0] * pad_len, dtype=torch.long),
                "labels": torch.tensor(buf_labels[:self.seq_len], dtype=torch.long),
            }

    def __iter__(self):
        batch = []
        for packed in self._pack_iter():
            batch.append(packed)
            if len(batch) >= self.batch_size:
                yield {
                    "input_ids": torch.stack([b["input_ids"] for b in batch]),
                    "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                    "labels": torch.stack([b["labels"] for b in batch]),
                }
                batch = []
        if batch:
            yield {
                "input_ids": torch.stack([b["input_ids"] for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                "labels": torch.stack([b["labels"] for b in batch]),
            }


# ============================================================
# Trainable parameter selection
# ============================================================

# Module name patterns for each training mode.
# Qwen3.5 layers have either:
#   - linear_attention (Gated DeltaNet): may appear as gated_deltanet, linear_attn, etc.
#   - full_attention (Gated Attention + GQA): may appear as self_attn, attention, etc.
# Both types have: mlp (MoE), input_layernorm, post_attention_layernorm
# We use broad patterns to match both attention types.
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
        "include": None,  # None = all params in the layer
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
            print(f"  [LoRA] {name}: ({out_features}, {in_features}) → r={lora_rank}",
                  flush=True)

    if rank == 0:
        if injected > 5:
            print(f"  [LoRA] ... and {injected - 5} more modules", flush=True)
        print(f"[LoRA] Injected {injected} adapters | r={lora_rank} α={lora_alpha} "
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
        mode_str = f"lora({train_modules})"
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
            if include_patterns is not None:
                match = any(pref in name for pref in all_prefixes)
            else:
                match = any(pref in name for pref in all_prefixes)
            if match:
                p.requires_grad = True
                trainable += p.numel()
            else:
                p.requires_grad = False
        mode_str = train_modules

    layers_str = ",".join(str(i) for i in sorted(train_set))
    print(f"[INFO] trainable({mode_str})={trainable:,} / total={total:,} "
          f"({trainable/total*100:.4f}%) [layers {layers_str}, "
          f"modules={description}]", flush=True)

    return trainable


# ============================================================
# Trainer
# ============================================================

class Trainer:
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

        # Get the text model's layer class for FSDP wrapping
        text_config = getattr(config, "text_config", config)
        model_type = getattr(text_config, "model_type", "")

        # Try to import the layer class from transformers
        wrap_class = None
        # For Qwen3.5, try known class names
        for cls_name in [
            "Qwen3_5MoeDecoderLayer",
            "Qwen3_5MoeTextDecoderLayer",
            "Qwen3NextMoeDecoderLayer",
        ]:
            try:
                import transformers.models
                # Walk transformers to find the class
                for attr_name in dir(transformers.models):
                    mod = getattr(transformers.models, attr_name, None)
                    if mod and hasattr(mod, cls_name):
                        wrap_class = getattr(mod, cls_name)
                        break
                if wrap_class:
                    break
            except Exception:
                continue

        # Broader fallback: load model config and find layer class
        if wrap_class is None:
            try:
                from transformers import AutoModel
                # Try to get the class from the architecture string
                arch = config.architectures[0] if hasattr(config, "architectures") else ""
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

        # BF16 native: simple mixed precision
        fsdp_plugin.mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # param_init_fn: only materialize meta tensors
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
        self.accelerator = Accelerator(
            mixed_precision="no",  # FSDP MixedPrecision handles this
            fsdp_plugin=fsdp_plugin,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kwargs_handlers=[pg_kwargs],
        )

        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.device = self.accelerator.device

        if self.accelerator.is_main_process:
            print(f"[DIST] rank={self.rank} / world_size={self.world_size}", flush=True)

        # --------------------------------------------------
        # 3) Model loading
        # --------------------------------------------------
        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        if self.accelerator.is_main_process:
            print(f"[LOAD] model: {args.model}", flush=True)

        if self.rank == 0:
            print("[LOAD] rank 0: loading BF16 weights to CPU...", flush=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                config=config,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
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
                        args.model,
                        config=config,
                        device_map=None,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
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

        # --------------------------------------------------
        # 3.5) Discover wrap class from loaded model if needed
        # --------------------------------------------------
        if wrap_class is None:
            # Find the layer class from the loaded model's layers
            layers_list = None
            for mod in base_model.modules():
                if isinstance(mod, nn.ModuleList) and len(mod) == args.num_layers:
                    layers_list = mod
                    break
            if layers_list is not None and len(layers_list) > 0:
                wrap_class = type(layers_list[0])
                print(f"[FSDP] Discovered wrap class from model: {wrap_class.__name__}", flush=True)
                fsdp_plugin.auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={wrap_class},
                )

        # --------------------------------------------------
        # 3.6) Print model structure (rank 0, first run diagnostic)
        # --------------------------------------------------
        if self.rank == 0:
            print("\n[MODEL] Architecture diagnostic:", flush=True)
            for name, mod in base_model.named_modules():
                depth = name.count(".")
                if depth <= 2:
                    print(f"  {'  ' * depth}{name}: {type(mod).__name__}", flush=True)
                elif "layers.0." in name and depth <= 4:
                    short = name.replace("model.layers.0.", "L0.")
                    print(f"    {short}: {type(mod).__name__}", flush=True)

            # Show last layer structure
            print(f"\n[MODEL] Last layer (layer {args.num_layers - 1}):", flush=True)
            last_prefix = f"model.layers.{args.num_layers - 1}."
            for name, mod in base_model.named_modules():
                if name.startswith(last_prefix):
                    short = name.replace(last_prefix, "  ")
                    if short.count(".") <= 1:
                        print(f"    {short}: {type(mod).__name__}", flush=True)

            # Param dtype summary
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

        # Freeze all except trainable
        trainable_count = mark_trainable(
            base_model, train_layer_indices, args.train_modules,
            use_lora=args.use_lora,
        )
        if trainable_count == 0:
            raise RuntimeError(f"No trainable params for layers {train_layer_indices}")

        # Note: No buffer conversion needed for BF16 native model.
        # FSDP shards all Parameters (including frozen ones) across GPUs.
        # Buffer conversion would cause OOM because FSDP replicates buffers.

        # --------------------------------------------------
        # 5) FSDP wrap
        # --------------------------------------------------
        # Materialize meta buffers on non-rank-0
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
        # 6) Post-FSDP detach hooks
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
                      f"target: index {first_train_idx} → {type(fsdp_train_layer).__name__}",
                      flush=True)

        # Hook: detach + requires_grad on first trainable layer
        if fsdp_train_layer is not None:
            def _detach_and_enable_grad(module, args):
                new_args = []
                for a in args:
                    if isinstance(a, torch.Tensor):
                        new_args.append(a.detach().requires_grad_(True))
                    else:
                        new_args.append(a)
                return tuple(new_args)

            fsdp_train_layer.register_forward_pre_hook(_detach_and_enable_grad)
            if self.accelerator.is_main_process:
                print(f"[DETACH] Installed detach+grad hook on layer {first_train_idx} "
                      f"(frozen layers 0-{first_train_idx-1} run without grad)",
                      flush=True)

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
            print(f"[DETACH] MLP detach: {mlp_detach_count} layers "
                  f"(approx gradients via residual identity path)", flush=True)

        # --------------------------------------------------
        # 7) Optimizer
        # --------------------------------------------------
        trainable_params = [p for _, p in self.model.named_parameters()
                           if p.requires_grad and p.numel() > 0]

        if self.accelerator.is_main_process:
            print(f"[OPT] Trainable params for optimizer: {len(trainable_params)}", flush=True)
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.numel() > 0:
                    print(f"  {n}: shape={tuple(p.shape)}, numel={p.numel()}", flush=True)

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
            weight_decay=args.weight_decay,
            foreach=False,
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
            print(f"[DATA] packing: {args.packing}", flush=True)

        base_ds = OnTheFlyTokenizedChatMLJsonl(
            args.dataset, tok=self.tok, seq_len=args.seq_len,
            rank=self.rank, world_size=self.world_size,
        )

        if args.packing:
            ds = PackedDataset(base_ds, args.seq_len, args.batch_size, self.pad_token_id)
            self.loader = DataLoader(ds, batch_size=None, num_workers=0, pin_memory=True)
            # Packing estimate: sample a few lines for avg token length
            avg_tokens = 500  # default estimate
            if self.rank == 0:
                sample_n = min(50, len(base_ds.lines))
                sample_lens = []
                for i in range(sample_n):
                    r = base_ds._tokenize_line(base_ds.lines[i])
                    if r is not None:
                        sample_lens.append(len(r["input_ids"]))
                        # Print first sample diagnostic
                        if len(sample_lens) == 1:
                            ids = r["input_ids"]
                            labs = r["labels"]
                            n_train = sum(1 for l in labs if l != -100)
                            n_mask = sum(1 for l in labs if l == -100)
                            print(f"[DATA] sample 0: {len(ids)} tokens, "
                                  f"{n_train} trainable, {n_mask} masked",
                                  flush=True)
                            # Show first few tokens of each region
                            in_train = False
                            for j, (t, l) in enumerate(zip(ids, labs)):
                                is_train = (l != -100)
                                if is_train != in_train:
                                    region = "TRAIN" if is_train else "MASK"
                                    snippet = self.tok.decode(ids[j:j+10])
                                    print(f"  [{region}] @tok {j}: "
                                          f"{repr(snippet[:60])}...",
                                          flush=True)
                                    in_train = is_train
                if sample_lens:
                    avg_tokens = sum(sample_lens) // len(sample_lens)
                    print(f"[DATA] sampled {len(sample_lens)} lines → avg {avg_tokens} tokens/sample",
                          flush=True)

            rank_samples = base_ds.total // self.world_size
            rank_tokens = rank_samples * avg_tokens
            packed_seqs = max(1, rank_tokens // args.seq_len)
            if self.accelerator.is_main_process:
                print(f"[DATA] packing estimate: ~{rank_tokens:,} tokens/rank, "
                      f"~{packed_seqs} packed_seqs/rank", flush=True)
        else:
            def collate_fn(batch):
                max_len = max(b["length"] for b in batch)
                max_len = min(max_len, args.seq_len)
                ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
                mask = torch.zeros(len(batch), max_len, dtype=torch.long)
                labs = torch.full((len(batch), max_len), -100, dtype=torch.long)
                for i, b in enumerate(batch):
                    slen = min(b["length"], max_len)
                    ids[i, :slen] = b["input_ids"][:slen]
                    mask[i, :slen] = 1
                    labs[i, :slen] = b["labels"][:slen]
                return {"input_ids": ids, "attention_mask": mask, "labels": labs}

            self.loader = DataLoader(
                base_ds, batch_size=args.batch_size, collate_fn=collate_fn,
                num_workers=0, pin_memory=True, drop_last=False,
            )

        # Estimate steps
        gas = args.gradient_accumulation_steps
        eff_batch = args.batch_size * gas * self.world_size
        if args.packing:
            total_tokens_est = base_ds.total * avg_tokens
            total_packed = max(1, total_tokens_est // args.seq_len)
            micro_per_epoch = max(1, total_packed // (args.batch_size * self.world_size))
        else:
            lines_per_rank = base_ds.total // self.world_size
            micro_per_epoch = max(1, lines_per_rank // args.batch_size)
        optim_per_epoch = micro_per_epoch // gas
        total_steps = optim_per_epoch * (args.epochs or 1)
        self.estimated_total_steps = max(1, total_steps)

        if self.accelerator.is_main_process:
            print(f"[DATA] total_lines={base_ds.total}, "
                  f"micro_batches/epoch≈{micro_per_epoch}, "
                  f"optim_steps/epoch≈{optim_per_epoch}, "
                  f"estimated_total_steps≈{total_steps}, "
                  f"effective_batch_size={eff_batch}", flush=True)

        # Summary
        if self.accelerator.is_main_process:
            # Determine layer types
            layer_types = getattr(text_config, "layer_types", [])
            train_layer_types = [
                layer_types[i] if i < len(layer_types) else "unknown"
                for i in train_layer_indices
            ]

            print("=" * 56)
            print(f"  Qwen3.5 Partial-Layer SFT "
                  f"({'LoRA' if args.use_lora else 'Full'})")
            print(f"  model        : {args.model}")
            print(f"  num_layers   : {args.num_layers}")
            print(f"  train_layers : {layers_str} ({len(train_layer_indices)} layers)")
            print(f"  layer_types  : {train_layer_types}")
            print(f"  train_modules: {args.train_modules} ({TRAIN_MODULE_PATTERNS[args.train_modules]['description']})")
            print(f"  dataset      : {args.dataset}")
            print(f"  packing      : {args.packing}")
            print(f"  batch_size   : {args.batch_size} (micro)")
            print(f"  grad_accum   : {gas}")
            print(f"  eff_batch    : {eff_batch} "
                  f"(micro×accum×GPUs={args.batch_size}×{gas}×{self.world_size})")
            print(f"  seq_len      : {args.seq_len}")
            print(f"  optim        : adamw")
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
                        "packing": args.packing,
                        "batch_size": args.batch_size,
                        "grad_accum": gas,
                        "eff_batch": eff_batch,
                        "seq_len": args.seq_len,
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

    def step(self, batch):
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)

        self.model.train()

        with self.accelerator.accumulate(self.model):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                    output_router_logits=False,
                    return_dict=True,
                )
                loss = outputs.loss

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients and self.args.grad_clip > 0:
                self.accelerator.clip_grad_norm_(
                    [p for _, p in self.model.named_parameters() if p.requires_grad],
                    self.args.grad_clip,
                )
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        loss_val = loss.item()
        self._micro_step += 1

        if not (loss_val != loss_val):  # not NaN
            self._micro_loss_accum += loss_val
            self._micro_loss_count += 1

        gas = self.args.gradient_accumulation_steps
        if self._micro_step % gas == 0:
            self.global_step += 1
            current_step = self.global_step

            accum_avg_loss = self._micro_loss_accum / max(1, self._micro_loss_count)
            self._micro_loss_accum = 0.0
            self._micro_loss_count = 0
            self._loss_accum += accum_avg_loss
            self._loss_count += 1

            step_end = time.time()
            if self._train_start_time is not None:
                self._step_times.append(step_end)

            if current_step % self.args.log_every == 0 and self.accelerator.is_main_process:
                elapsed = step_end - self._train_start_time if self._train_start_time else 0
                avg_loss = self._loss_accum / max(1, self._loss_count)

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
                    f"loss={accum_avg_loss:.4f} (avg={avg_loss:.4f}) | "
                    f"lr={self.opt.param_groups[0]['lr']:.2e} | "
                    f"{sec_per_step:.1f}s/step | "
                    f"elapsed={str(datetime.timedelta(seconds=int(elapsed)))} | "
                    f"ETA={eta_str}",
                    flush=True,
                )

                if self.use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": accum_avg_loss,
                        "train/avg_loss": avg_loss,
                        "train/lr": self.opt.param_groups[0]["lr"],
                        "train/epoch": self.global_epoch,
                        "perf/sec_per_step": sec_per_step,
                    }, step=current_step)

            # Save periodically
            if (self.args.save_every and current_step % self.args.save_every == 0
                    and current_step > 0):
                self.save(os.path.join(self.args.save_dir, f"step-{current_step}"))

    def _should_stop(self):
        return (self.args.steps and self.args.steps > 0
                and self.global_step >= self.args.steps)

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
                    "format": "lora_sharded" if self.args.use_lora else "partial_layer_sharded",
                    "pattern": "params.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "model": self.args.model,
                    "train_layers": list(self.train_layer_indices),
                    "train_modules": self.args.train_modules,
                    "use_lora": self.args.use_lora,
                    "lora_rank": self.args.lora_rank if self.args.use_lora else None,
                    "lora_alpha": self.args.lora_alpha if self.args.use_lora else None,
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

    def _rank_synced_batches(self, loader):
        """Yield batches only while EVERY rank still has one.

        With IterableDataset + packing, the number of batches per rank can
        differ (line sharding remainder, packing boundaries). A rank that
        runs out early stops calling the FSDP collectives and the others
        hang. Before consuming each batch, all ranks all_reduce(MIN) a
        "has next batch" flag; as soon as any rank is exhausted, every rank
        stops together (trailing batches on longer ranks are dropped).
        """
        it = iter(loader)
        use_dist = (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        )
        while True:
            try:
                batch = next(it)
                has_batch = 1
            except StopIteration:
                batch = None
                has_batch = 0
            if use_dist:
                flag = torch.tensor(
                    [has_batch], dtype=torch.long, device=self.device
                )
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                has_batch = int(flag.item())
            if not has_batch:
                if batch is not None and self.accelerator.is_main_process:
                    print(
                        "[DATA] dropping trailing batch(es) to keep ranks "
                        "aligned (another rank exhausted its shard)",
                        flush=True,
                    )
                break
            yield batch

    def train(self):
        self.model.train()
        self._train_start_time = time.time()
        self._step_times = [self._train_start_time]
        self._loss_accum = 0.0
        self._loss_count = 0
        self._micro_step = 0
        self._micro_loss_accum = 0.0
        self._micro_loss_count = 0

        if self.args.epochs and self.args.epochs > 0:
            for ep in range(1, self.args.epochs + 1):
                self.global_epoch = ep
                epoch_start = time.time()
                epoch_step_start = self.global_step
                if self.accelerator.is_main_process:
                    print(f"\n[EPOCH] start {ep}/{self.args.epochs}", flush=True)
                for batch in self._rank_synced_batches(self.loader):
                    self.step(batch)
                    if self._should_stop():
                        break
                if self.accelerator.is_main_process:
                    epoch_elapsed = time.time() - epoch_start
                    epoch_steps = self.global_step - epoch_step_start
                    avg_loss = self._loss_accum / max(1, self._loss_count)
                    print(
                        f"\n[EPOCH] finished {ep}/{self.args.epochs} "
                        f"| steps={epoch_steps} "
                        f"| avg_loss={avg_loss:.4f} "
                        f"| time={str(datetime.timedelta(seconds=int(epoch_elapsed)))}",
                        flush=True,
                    )
                if self._should_stop():
                    break

        # Step-only mode
        if not self.args.epochs or self.args.epochs <= 0:
            if self.accelerator.is_main_process:
                print(f"\n[TRAIN] step-only mode (steps={self.args.steps})", flush=True)
            while not self._should_stop():
                for batch in self._rank_synced_batches(self.loader):
                    self.step(batch)
                    if self._should_stop():
                        break

        # Final save
        if self.accelerator.is_main_process:
            total_elapsed = time.time() - self._train_start_time
            avg_loss = self._loss_accum / max(1, self._loss_count)
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


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 Partial-Layer SFT")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_layers", type=int, default=60)
    parser.add_argument("--train_layers", type=int, default=1)
    parser.add_argument("--train_modules", type=str, default="attn",
                        choices=list(TRAIN_MODULE_PATTERNS.keys()))
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--bucket_buffer", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw", choices=["adamw"])
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./output/qwen35-lastlayer-sft")
    parser.add_argument("--seed", type=int, default=42)
    # LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    # W&B
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
