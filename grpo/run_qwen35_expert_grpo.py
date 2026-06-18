#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expert-only full-parameter RL with GRPO-style objective and LLM-as-judge reward.
Adapted for Qwen3.5-397B-A17B (60 layers, 512 MoE experts, hybrid attention, BF16 native).

Key differences from Qwen3-235B version:
  - 60 layers (was 94), 512 experts (was 128)
  - 10 routed + 1 shared expert per layer
  - Hybrid attention: 45 Gated DeltaNet + 15 Gated Attention (3:1 ratio)
  - BF16 native (no FP8 dequant)
  - Vision encoder (ViT) — frozen
  - Chat template: enable_thinking=False
  - Memory-efficient loading: rank0 loads to CPU, others use meta device
"""

import os
import re
import json
import math
import random
import argparse
import hashlib
import time
import datetime
import threading
from typing import List, Tuple, Optional, Set, Any
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Standalone FSDP vLLM client - no TRL dependency required
# Place fsdp_vllm_client.py alongside this script
try:
    from fsdp_vllm_client import FSDPVLLMClient as VLLMClient
except ImportError:
    try:
        from trl.generation.vllm_client import VLLMClient
    except ImportError:
        try:
            from trl.extras.vllm_client import VLLMClient
        except ImportError:
            VLLMClient = None  # vLLM support disabled

from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import InitProcessGroupKwargs
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    StateDictType,
    ShardedStateDictConfig,
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


def extract_text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [extract_text_from_content(x) for x in content]
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        inner = content.get("content")
        if isinstance(inner, str):
            return inner
        if isinstance(inner, (list, dict)):
            return extract_text_from_content(inner)
        parts = [v for v in content.values() if isinstance(v, str)]
        return "\n".join(parts)
    return ""


class ChatMLJsonl(IterableDataset):
    def __init__(self, path: str, rank: int = 0, world_size: int = 1):
        self.path = path
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % self.world_size != self.rank:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                msgs = rec.get("messages") or []
                umsgs = [m for m in msgs if m.get("role") == "user"]
                amsgs = [m for m in msgs if m.get("role") == "assistant"]
                if not umsgs:
                    continue
                user = extract_text_from_content(umsgs[0].get("content")).strip()
                ref = extract_text_from_content(amsgs[0].get("content")).strip() if amsgs else ""
                if not user:
                    continue
                yield {"prompt": user, "reference": ref}


# ============================================================
# Qwen3.5 Expert/Router param detection
# ============================================================

# Qwen3.5 expert naming patterns:
#   model.layers.{L}.mlp.experts.{E}.gate_proj.weight
#   model.layers.{L}.mlp.experts.{E}.up_proj.weight
#   model.layers.{L}.mlp.experts.{E}.down_proj.weight
#   model.layers.{L}.mlp.shared_expert.gate_proj.weight  (shared expert)
#   model.layers.{L}.mlp.shared_expert.up_proj.weight
#   model.layers.{L}.mlp.shared_expert.down_proj.weight
#   model.layers.{L}.mlp.gate.weight  (router)

_EXPERT_RE = re.compile(r"\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
_SHARED_EXPERT_RE = re.compile(r"\.layers\.(\d+)\.mlp\.shared_expert[s]?\.")
_ATTN_RE = re.compile(r"\.layers\.(\d+)\.self_attn\.")
_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def parse_layer_idx_from_name(name: str) -> Optional[int]:
    """Extract layer index from param name."""
    m = _LAYER_IDX_RE.search(name)
    return int(m.group(1)) if m else None


def is_attention_param(name: str) -> bool:
    """Check if param belongs to self-attention (q/k/v/o_proj, gate, etc.)."""
    return bool(_ATTN_RE.search(name))


def is_layernorm_param(name: str) -> bool:
    """Check if param is a layer norm weight/bias."""
    lname = name.lower()
    return "layernorm" in lname or "layer_norm" in lname or "rmsnorm" in lname


def build_attention_layer_set(
    mode: str,
    n_layers: int,
    layer_types: Optional[List[str]] = None,
    attn_layer_indices: Optional[str] = None,
) -> Optional[Set[int]]:
    """
    Build the set of layer indices whose attention params should be trained.

    Qwen3.5 layer_types: 60-element list of "linear_attention" and "full_attention"
      - "linear_attention" → Gated DeltaNet (45 layers)
      - "full_attention"   → Gated Attention (15 layers)

    mode:
      "none"     → return None (don't train attention)
      "full"     → only full_attention layers
      "linear"   → only linear_attention (DeltaNet) layers
      "all"      → all layers
      "indices"  → specific indices from --attn_layer_indices "0,1,15,30,45,59"

    Returns:
      Set of layer indices, or None if mode="none"
    """
    if mode == "none":
        return None

    if mode == "all":
        return set(range(n_layers))

    if mode == "indices":
        if not attn_layer_indices:
            raise ValueError("--train_attention=indices requires --attn_layer_indices")
        return {int(x.strip()) for x in attn_layer_indices.split(",") if x.strip()}

    if mode in ("full", "linear"):
        if layer_types is None or len(layer_types) == 0:
            print(f"[WARN] layer_types not available in config; "
                  f"--train_attention={mode} falls back to 'all'", flush=True)
            return set(range(n_layers))

        target = set()
        for i, lt in enumerate(layer_types):
            lt_lower = str(lt).lower()
            if mode == "full" and ("full" in lt_lower or lt_lower == "attention"):
                target.add(i)
            elif mode == "linear" and "linear" in lt_lower:
                target.add(i)
        return target

    raise ValueError(f"Unknown --train_attention mode: {mode}")


def parse_layer_expert_from_name(name: str) -> Optional[Tuple[int, int]]:
    """Parse (layer_idx, expert_id) from routed expert param name."""
    m = _EXPERT_RE.search(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def is_shared_expert_param(name: str) -> bool:
    """Check if param belongs to a shared expert."""
    return bool(_SHARED_EXPERT_RE.search(name))


def is_router_param_name(name: str) -> bool:
    """Check if param is a router/gate weight."""
    lname = name.lower()
    # Qwen3.5: model.layers.{L}.mlp.gate.weight
    if lname.endswith("mlp.gate.weight") or ".mlp.gate.weight" in lname:
        return True
    if "router" in lname and "router_logits" not in lname:
        return True
    if "routing" in lname:
        return True
    return False


def is_selected_expert_param(name: str, layer_masks: Optional[List[torch.Tensor]],
                              include_shared: bool = True) -> bool:
    """Check if param belongs to a selected routed expert (or shared expert if enabled)."""
    if layer_masks is None:
        return False

    # Shared expert: always included if include_shared=True
    if include_shared and is_shared_expert_param(name):
        m = _SHARED_EXPERT_RE.search(name)
        if m:
            li = int(m.group(1))
            if 0 <= li < len(layer_masks):
                # Include shared expert if ANY expert in this layer is selected
                return layer_masks[li].any().item()
        return False

    # Routed expert
    parsed = parse_layer_expert_from_name(name)
    if parsed is None:
        return False
    li, ei = parsed
    if li < 0 or li >= len(layer_masks):
        return False
    mask = layer_masks[li]
    if ei < 0 or ei >= mask.numel():
        return False
    return bool(mask[ei].item())


def load_mask(mask_json: Optional[str], n_layers: int, n_experts: int) -> Optional[List[torch.Tensor]]:
    """Load expert selection mask from JSON."""
    if not mask_json:
        return None
    with open(mask_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    layer_masks = [torch.zeros(n_experts, dtype=torch.bool) for _ in range(n_layers)]

    sel = obj.get("selected_layer_experts")
    if sel:
        for it in sel:
            li = int(it["layer_idx"])
            ei = int(it["expert_id"])
            if 0 <= li < n_layers and 0 <= ei < n_experts:
                layer_masks[li][ei] = True
    else:
        ex = (obj.get("selected_experts_unique")
              or obj.get("selected_experts")
              or obj.get("picked_experts"))
        if not ex:
            raise RuntimeError("Mask JSON lacks selected_* fields.")
        for li in range(n_layers):
            for ei in ex:
                ei = int(ei)
                if 0 <= ei < n_experts:
                    layer_masks[li][ei] = True

    # Fallback: if any layer has zero selected, enable all
    for li in range(n_layers):
        if layer_masks[li].sum().item() == 0:
            layer_masks[li] = torch.ones(n_experts, dtype=torch.bool)

    return layer_masks


def mark_trainable_experts_only(model: nn.Module,
                                 layer_masks: Optional[List[torch.Tensor]],
                                 include_shared: bool = True,
                                 train_router: bool = False,
                                 attn_layers: Optional[Set[int]] = None,
                                 train_layernorm: bool = False):
    """Mark only selected expert params (+ optionally shared experts, routers, attention) as trainable.

    Args:
        attn_layers: Set of layer indices whose attention params should be trained.
                     None = don't train attention.
        train_layernorm: Also train layernorm in attention-trained layers.
    """
    total = 0
    trainable = 0
    shared_count = 0
    router_count = 0
    attn_count = 0
    norm_count = 0

    for name, p in model.named_parameters():
        total += p.numel()

        # 1) Router
        if train_router and is_router_param_name(name):
            p.requires_grad = True
            trainable += p.numel()
            router_count += p.numel()
            continue

        # 2) Selected experts (routed + shared)
        if is_selected_expert_param(name, layer_masks, include_shared=include_shared):
            p.requires_grad = True
            trainable += p.numel()
            if is_shared_expert_param(name):
                shared_count += p.numel()
            continue

        # 3) Attention params
        if attn_layers is not None and is_attention_param(name):
            li = parse_layer_idx_from_name(name)
            if li is not None and li in attn_layers:
                p.requires_grad = True
                trainable += p.numel()
                attn_count += p.numel()
                continue

        # 4) LayerNorm (only in attention-trained layers)
        if train_layernorm and attn_layers is not None and is_layernorm_param(name):
            li = parse_layer_idx_from_name(name)
            if li is not None and li in attn_layers:
                p.requires_grad = True
                trainable += p.numel()
                norm_count += p.numel()
                continue

        # Freeze everything else
        p.requires_grad = False

    attn_mode_str = "none" if attn_layers is None else f"{len(attn_layers)} layers"
    print(
        f"[INFO] trainable={trainable:,} / total={total:,} "
        f"({trainable / max(1, total) * 100:.4f}%) "
        f"[experts={trainable - shared_count - router_count - attn_count - norm_count:,}, "
        f"shared={shared_count:,}, router={router_count:,}, "
        f"attn={attn_count:,} ({attn_mode_str}), norm={norm_count:,}]",
        flush=True,
    )
    return [p for p in model.parameters() if p.requires_grad]


# ============================================================
# Layer class discovery for FSDP wrapping (from Qwen3.5 SFT script)
# ============================================================

def discover_layer_class(config, num_layers: int):
    """Try to find the Qwen3.5 decoder layer class for FSDP wrapping."""
    wrap_class = None

    # Known Qwen3.5 class names
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

    # Broader fallback
    if wrap_class is None:
        for mod_name in ["qwen3_5_moe", "qwen3_5", "qwen3_next_moe", "qwen3_next"]:
            try:
                mod = __import__(
                    f"transformers.models.{mod_name}.modeling_{mod_name}",
                    fromlist=["_"],
                )
                for name in dir(mod):
                    if "DecoderLayer" in name:
                        wrap_class = getattr(mod, name)
                        break
                if wrap_class:
                    break
            except ImportError:
                continue

    return wrap_class


def discover_layer_class_from_model(model, num_layers: int):
    """Discover layer class from an instantiated model."""
    for mod in model.modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == num_layers:
            if len(mod) > 0:
                return type(mod[0])
    return None


# ============================================================
# Judge
# ============================================================

JUDGE_SYSTEM = (
    "You are a strict grading assistant.\n"
    "You evaluate the candidate answer for correctness, completeness, safety (no hallucinations), and clarity.\n"
    "You MUST follow this procedure:\n"
    "1. First, think step by step in detail about the question and the candidate answer.\n"
    "   - Carefully compare it with the reference answer if provided.\n"
    "   - Identify correct points, missing points, and any unsafe or incorrect statements.\n"
    "2. After you finish your reasoning, on the FINAL line output exactly ONE of:\n"
    "   [1], [0.5], [0]\n"
    "   - The final line must contain ONLY the score token (no extra text).\n"
    "Scoring rule:\n"
    "- [1]   : completely correct and safe. No important errors or omissions.\n"
    "- [0.5] : partially correct. Some important parts are missing or slightly wrong, "
    "but parts of the answer are still useful.\n"
    "- [0]   : incorrect, unsafe, or mostly wrong / misleading.\n"
)

JUDGE_TEMPLATE = (
    "User question:\n{prompt}\n\n"
    "{ref_block}"
    "Candidate answer:\n{candidate}\n\n"
    "Think step by step according to the rubric above. "
    "After you finish your reasoning, output the final score on the LAST line as one of [1], [0.5], or [0] "
    "with no additional text on that line."
)


def build_judge_text(prompt: str, candidate: str, reference: Optional[str]) -> str:
    ref_block = ""
    if reference:
        ref_block = f"Ideal reference answer (for evaluator's eyes only):\n{reference}\n\n"
    return JUDGE_TEMPLATE.format(prompt=prompt, ref_block=ref_block, candidate=candidate)


class ScoreCache:
    def __init__(self, cache_path: Optional[str]):
        self.path = cache_path
        self.mem = {}
        self._lock = threading.Lock()
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        self.mem[rec["key"]] = rec["score"]
            except Exception:
                pass

    def key(self, prompt: str, candidate: str, reference: str) -> str:
        h = hashlib.sha1()
        h.update(prompt.encode("utf-8"))
        h.update(b"\x00")
        h.update(candidate.encode("utf-8"))
        h.update(b"\x00")
        h.update((reference or "").encode("utf-8"))
        return h.hexdigest()

    def get(self, k: str) -> Optional[float]:
        return self.mem.get(k)

    def put(self, k: str, score: float):
        with self._lock:
            self.mem[k] = float(score)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": k, "score": float(score)}, ensure_ascii=False) + "\n")


# ============================================================
# Trainer
# ============================================================

class Qwen35ExpertGRPOTrainer:
    """
    Expert-only GRPO trainer adapted for Qwen3.5-397B-A17B.

    Key adaptations from Qwen3-235B version:
    - Memory-efficient loading: rank0 CPU, others meta device
    - Layer class auto-discovery for FSDP
    - Shared expert support
    - enable_thinking=False for chat template
    - 60 layers, 512 experts (10 routed + 1 shared)
    """

    def __init__(self, args):
        self.args = args
        self.wandb_run = None

        if not _FSDP_AVAILABLE:
            raise RuntimeError("[FSDP] torch.distributed.fsdp が使えません。")
        if not torch.cuda.is_available():
            raise RuntimeError("[CUDA] GPU が必要です。")

        rank = int(os.environ.get("RANK", "0"))

        # ---- 1) Discover layer class ----
        if rank == 0:
            print("[INIT] Loading model config to detect layer class...", flush=True)

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        wrap_class = discover_layer_class(config, args.layers)

        # Fallback: instantiate meta model to discover class
        if wrap_class is None:
            with torch.device("meta"):
                try:
                    temp_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
                    wrap_class = discover_layer_class_from_model(temp_model, args.layers)
                    del temp_model
                except Exception:
                    pass

        if wrap_class is not None and rank == 0:
            print(f"[INIT] Detected Layer Class: {wrap_class.__name__}", flush=True)
        elif rank == 0:
            print("[WARN] Could not find decoder layer class. Will discover from loaded model.", flush=True)

        # ---- 2) FSDP / Accelerator setup ----
        auto_wrap_policy = None
        if wrap_class is not None:
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={wrap_class},
            )

        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            sync_module_states=True,
            limit_all_gathers=True,
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing=True,
            cpu_offload=False,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
        )

        # BF16 native mixed precision
        fsdp_plugin.mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # param_init_fn: materialize meta tensors only
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

        pg_kwargs = InitProcessGroupKwargs(
            timeout=datetime.timedelta(seconds=7200)
        )

        self.accelerator = Accelerator(
            mixed_precision="no",  # FSDP MixedPrecision handles this
            fsdp_plugin=fsdp_plugin,
            kwargs_handlers=[pg_kwargs],
        )

        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        self.device = self.accelerator.device
        self.model_device = self.device

        if self.accelerator.is_main_process:
            print(
                f"[DIST] rank={self.rank} / world_size={self.world_size}",
                flush=True,
            )

        self.use_vllm = bool(getattr(args, "use_vllm", False))
        self.vllm_client: Optional[VLLMClient] = None
        self.vllm_clients: List[VLLMClient] = []

        # ---- 3) Tokenizer ----
        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        # ---- 4) Model loading (memory-efficient: rank0 CPU, others meta) ----
        if self.accelerator.is_main_process:
            print(f"[LOAD] model: {args.model}", flush=True)

        if self.rank == 0:
            print("[LOAD] rank 0: loading BF16 weights to CPU...", flush=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                config=config,
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
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
                        torch_dtype=torch.bfloat16,
                        device_map=None,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
            finally:
                _tm_utils.PreTrainedModel._load_pretrained_model = _orig_load
                _tm_utils.PreTrainedModel._finalize_model_loading = _orig_finalize
                if _orig_check is not None:
                    _tm_utils.check_and_set_device_map = _orig_check
            print(f"[LOAD] rank {self.rank}: meta model created.", flush=True)

        # Discover wrap class from loaded model if needed
        if wrap_class is None:
            wrap_class = discover_layer_class_from_model(base_model, args.layers)
            if wrap_class is not None:
                if self.rank == 0:
                    print(f"[FSDP] Discovered wrap class from model: {wrap_class.__name__}", flush=True)
                fsdp_plugin.auto_wrap_policy = partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={wrap_class},
                )

        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = True
        if hasattr(base_model.config, "output_router_logits"):
            base_model.config.output_router_logits = False

        # Gradient checkpointing + frozen embedding support
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        base_model.train()

        # ---- 5) Expert mask + attention mode + trainable marking ----
        self.layer_masks = None
        if args.mask_json:
            self.layer_masks = load_mask(
                args.mask_json,
                n_layers=args.layers,
                n_experts=args.experts,
            )
            if self.accelerator.is_main_process:
                allowed = int(sum(m.sum().item() for m in self.layer_masks))
                print(f"[INFO] loaded coverage-mask allow entries = {allowed}", flush=True)

        # Attention training: extract layer_types from config
        tc = getattr(config, "text_config", config)
        layer_types = getattr(tc, "layer_types", None)
        if layer_types is not None and self.accelerator.is_main_process:
            n_linear = sum(1 for t in layer_types if "linear" in str(t).lower())
            n_full = sum(1 for t in layer_types if "full" in str(t).lower() or str(t).lower() == "attention")
            print(f"[INFO] layer_types: {n_linear} DeltaNet + {n_full} FullAttn (total {len(layer_types)})")

        attn_mode = getattr(args, "train_attention", "none")
        attn_layers = build_attention_layer_set(
            mode=attn_mode,
            n_layers=args.layers,
            layer_types=layer_types,
            attn_layer_indices=getattr(args, "attn_layer_indices", None),
        )
        self.attn_layers = attn_layers

        if attn_layers is not None and self.accelerator.is_main_process:
            sorted_layers = sorted(attn_layers)
            print(f"[INFO] attention training: mode={attn_mode}, {len(attn_layers)} layers")
            if len(sorted_layers) <= 20:
                print(f"[INFO] attention layer indices: {sorted_layers}")
            else:
                print(f"[INFO] attention layer indices: {sorted_layers[:10]} ... {sorted_layers[-5:]}")

        _ = mark_trainable_experts_only(
            base_model,
            self.layer_masks,
            include_shared=bool(args.include_shared_expert),
            train_router=bool(args.train_router),
            attn_layers=attn_layers,
            train_layernorm=bool(getattr(args, "train_layernorm", False)),
        )

        # ---- 6) Materialize meta buffers on non-rank-0 ----
        if self.rank != 0:
            for module in base_model.modules():
                for bname in list(module._buffers.keys()):
                    buf = module._buffers[bname]
                    if buf is not None and buf.device.type == "meta":
                        module._buffers[bname] = torch.empty(
                            buf.shape, dtype=buf.dtype, device="cpu")

        self.accelerator.wait_for_everyone()

        # ---- 7) DataLoader ----
        self.ds = ChatMLJsonl(args.dataset, rank=self.rank, world_size=self.world_size)
        loader = DataLoader(
            self.ds,
            batch_size=args.batch_size,
            drop_last=True,
        )

        # ---- 8) FSDP wrap ----
        if self.accelerator.is_main_process:
            print("[FSDP] Wrapping model (sync_module_states=True)...", flush=True)

        self.model = self.accelerator.prepare(base_model)
        self.loader = loader

        self.is_fsdp = isinstance(self.model, FSDP)

        if self.accelerator.is_main_process:
            print("[FSDP] Model wrapped successfully.", flush=True)

        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
        except Exception:
            unwrapped = self.model
        self.decoder = getattr(unwrapped, "model", None)

        # ---- 9) Optimizer ----
        expert_params = [p for p in self.model.parameters() if p.requires_grad]
        if not expert_params:
            raise RuntimeError("No trainable expert parameters found.")
        self.opt = torch.optim.AdamW(
            expert_params,
            lr=args.lr,
            betas=(0.9, 0.98),
            weight_decay=args.weight_decay,
        )

        # NOTE: eos may double as the pad token here.  Attention masks must
        # therefore be built from sequence lengths (see _build_batch), never
        # from `ids != pad_token_id`, or real <|im_end|> tokens get masked.
        self.pad_token_id = (
            self.tok.eos_token_id
            if self.tok.eos_token_id is not None
            else self.tok.pad_token_id
        )

        # ---- 10) Judge setup ----
        if self.accelerator.is_main_process:
            print("[INFO] judge backend:", args.judge_backend, flush=True)
            if args.judge_backend != "local":
                print(
                    f"[INFO] external judge model='{self.args.judge_model_name}' "
                    f"api_base='{self.args.judge_api_base}'",
                    flush=True,
                )

        # Per-rank reward cache
        if getattr(args, "reward_cache", None):
            base = args.reward_cache
            root, ext = os.path.splitext(base)
            ext = ext if ext else ".jsonl"
            args.reward_cache = f"{root}.rank{self.rank}{ext}"

        self.cache = ScoreCache(args.reward_cache)

        # Thread-local storage for per-thread HTTP sessions
        self._thread_local = threading.local()

        # ---- 11) vLLM client ----
        if self.use_vllm:
            if VLLMClient is None:
                raise RuntimeError(
                    "[VLLM] Cannot use vLLM: neither fsdp_vllm_client.py nor "
                    "trl.extras.vllm_client is available.\n"
                    "  Place fsdp_vllm_client.py alongside this script, or "
                    "install a TRL version with vllm_client support."
                )
            self._init_vllm_client()

        self.global_step = 0
        self.baseline: Optional[float] = None

        # ---- 12) W&B ----
        if self.args.wandb and self.accelerator.is_main_process:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.args.wandb_project,
                    entity=self.args.wandb_entity or None,
                    name=self.args.wandb_run_name,
                    config=vars(self.args),
                )
            except Exception as e:
                print(f"[WARN] wandb init failed: {e}", flush=True)
                self.wandb_run = None

        if self.accelerator.is_main_process:
            attn_mode_display = getattr(args, "train_attention", "none")
            attn_count = len(self.attn_layers) if self.attn_layers else 0
            print("=" * 60)
            print("  Qwen3.5-397B Expert Full-Param GRPO w/ LLM-as-judge")
            print(f"  actor model       : {self.args.model}")
            print(f"  judge backend     : {self.args.judge_backend}")
            if self.args.judge_backend != "local":
                print(f"  judge model       : {self.args.judge_model_name}")
                print(f"  judge concurrency : {getattr(self.args, 'judge_concurrency', 1)}")
            print(f"  dataset           : {self.args.dataset}")
            print(f"  layers/experts    : {self.args.layers}/{self.args.experts}")
            print(f"  include_shared    : {self.args.include_shared_expert}")
            print(f"  train_router      : {self.args.train_router}")
            print(f"  train_attention   : {attn_mode_display} ({attn_count} layers)")
            print(f"  train_layernorm   : {getattr(self.args, 'train_layernorm', False)}")
            print(f"  grpo_group_size   : {self.args.grpo_group_size}")
            print(f"  use_vllm          : {self.args.use_vllm}")
            print(f"  wandb             : {self.args.wandb}")
            print("=" * 60, flush=True)

    # ============================================================
    # vLLM Client
    # ============================================================

    def _init_vllm_client(self):
        timeout = self.args.vllm_connection_timeout

        if self.accelerator.is_main_process:
            print("[VLLM] initializing VLLMClient(s)...", flush=True)

        urls: List[Optional[str]] = []
        if getattr(self.args, "vllm_base_urls", None):
            urls = [u.strip() for u in self.args.vllm_base_urls.split(",") if u.strip()]
        elif self.args.vllm_base_url:
            urls = [self.args.vllm_base_url]
        else:
            urls = [None]

        clients: List[VLLMClient] = []
        for u in urls:
            if u:
                c = VLLMClient(u, connection_timeout=timeout)
            else:
                c = VLLMClient(connection_timeout=timeout)
            c.init_communicator(self.device)
            clients.append(c)

        self.vllm_clients = clients
        self.vllm_client = clients[0] if clients else None

        self._sync_weights_to_vllm("initial")

        if self.accelerator.is_main_process:
            print(f"[VLLM] ready clients={len(self.vllm_clients)}", flush=True)

    def _sync_weights_to_vllm(self, label: str) -> bool:
        """Sync model weights to all vLLM clients, deadlock-safe.

        The FSDP streaming sync contains collective ops (summon_full_params,
        barrier), so every rank must execute the same sequence of calls.
        A per-rank try/except is NOT enough: if only rank 0 fails and bails
        out, the other ranks keep waiting inside the collectives.

        The streaming path in fsdp_vllm_client keeps all collectives aligned
        even on send errors and only re-raises on rank 0 after its final
        barrier. Here we then broadcast the per-client outcome from rank 0
        so that ALL ranks agree to continue (or report failure) together.
        """
        all_ok = True
        for ci, c in enumerate(self.vllm_clients):
            err = None
            ok = 1
            try:
                if self.is_fsdp:
                    # Collective: called by ALL ranks.
                    c.update_model_params_streaming_fsdp(
                        self.model, trainable_only=True
                    )
                elif self.rank == 0:
                    # Non-FSDP: only rank 0 talks to the server.
                    c.update_model_params(self.model, trainable_only=True)
            except Exception as e:
                err = e
                ok = 0

            # Agree on the outcome across ranks: rank 0 performs the actual
            # NCCL/HTTP transfer, so its verdict is authoritative.
            if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                flag = torch.tensor([ok], dtype=torch.long, device=self.device)
                dist.broadcast(flag, src=0)
                ok = int(flag.item())

            if ok:
                if self.accelerator.is_main_process:
                    print(f"[VLLM] weight sync OK ({label}, client={ci})", flush=True)
            else:
                all_ok = False
                if self.accelerator.is_main_process:
                    print(
                        f"[WARN] VLLM weight sync failed ({label}, client={ci}): {err}",
                        flush=True,
                    )
        return all_ok

    # ============================================================
    # Generation
    # ============================================================

    def _apply_chat_template(self, prompt: str) -> str:
        """Apply Qwen3.5 chat template with enable_thinking=False."""
        if hasattr(self.tok, "apply_chat_template") and getattr(self.tok, "chat_template", None):
            try:
                return self.tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except TypeError:
                # Fallback without enable_thinking
                return self.tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
        return prompt

    @torch.no_grad()
    def _generate_with_vllm_grpo(self, prompts: List[str], group_size: int) -> List[List[str]]:
        assert self.vllm_client is not None or self.vllm_clients
        clients = self.vllm_clients if self.vllm_clients else [self.vllm_client]

        orig_B = len(prompts)
        force_n1 = bool(getattr(self.args, "vllm_force_n1", False))

        call_prompts = prompts
        call_n = int(group_size)

        if force_n1:
            expanded: List[str] = []
            for p in prompts:
                expanded.extend([p] * int(group_size))
            call_prompts = expanded
            call_n = 1

        # Apply chat template for vLLM
        call_prompts = [self._apply_chat_template(p) for p in call_prompts]

        gen_kwargs = {
            "max_tokens": int(self.args.gen_max_new_tokens),
            "repetition_penalty": float(self.args.gen_repetition_penalty),
            "n": int(call_n),
            "temperature": float(self.args.gen_temperature),
            "top_p": float(self.args.gen_top_p),
            "min_p": 0.0,
        }

        def _decode_resp(resp, local_prompts, local_n) -> List[List[str]]:
            # Handle FSDPVLLMClient's pre-grouped text format
            if isinstance(resp, dict) and "completion_texts" in resp:
                grouped = resp["completion_texts"]
                B = len(local_prompts)
                result = []
                for i in range(B):
                    if i < len(grouped):
                        group = grouped[i]
                        if len(group) < int(local_n):
                            group = group + [""] * (int(local_n) - len(group))
                        result.append([t.strip() for t in group[:int(local_n)]])
                    else:
                        result.append([""] * int(local_n))
                return result

            token_seqs = resp.get("completion_ids", resp) if isinstance(resp, dict) else resp

            outs_flat: List[str] = []

            def _to_list(x):
                return x.tolist() if isinstance(x, torch.Tensor) else x

            def _decode_ids(ids):
                ids = _to_list(ids)
                if not ids:
                    outs_flat.append("")
                    return
                if isinstance(ids, list) and ids and isinstance(ids[0], list):
                    for sub in ids:
                        try:
                            txt = self.tok.decode(_to_list(sub), skip_special_tokens=True)
                        except Exception:
                            txt = ""
                        outs_flat.append((txt or "").strip())
                    return
                try:
                    txt = self.tok.decode(ids, skip_special_tokens=True)
                except Exception:
                    txt = ""
                outs_flat.append((txt or "").strip())

            def _handle_seq(seq):
                seq = _to_list(seq)
                if seq is None:
                    outs_flat.append("")
                    return
                if isinstance(seq, dict):
                    ids = None
                    for k in ["output_token_ids", "token_ids", "tokens",
                              "output_ids", "generated_token_ids", "completion_ids"]:
                        if k in seq:
                            ids = seq[k]
                            break
                    text = seq.get("text") or seq.get("generated_text")
                    if ids is not None:
                        _decode_ids(ids)
                    else:
                        outs_flat.append((text or "").strip())
                    return
                if isinstance(seq, list) and seq and isinstance(seq[0], (dict, list)):
                    for sub in seq:
                        _handle_seq(sub)
                    return
                _decode_ids(seq)

            _handle_seq(token_seqs)

            B = len(local_prompts)
            expected = B * int(local_n)
            limit = min(len(outs_flat), expected)
            outs_flat_trimmed = outs_flat[:limit]

            grouped: List[List[str]] = []
            idx = 0
            for _ in range(B):
                group = outs_flat_trimmed[idx: idx + int(local_n)]
                if len(group) < int(local_n):
                    group = group + [""] * (int(local_n) - len(group))
                grouped.append(group)
                idx += int(local_n)
            return grouped

        # Multi-server support
        total = len(call_prompts) * call_n
        if self.accelerator.is_main_process:
            print(f"[GEN] {len(call_prompts)} prompts × n={call_n} = {total} completions", flush=True)

        _gen_t0 = time.time()

        if len(clients) == 1:
            resp = clients[0].generate(call_prompts, **gen_kwargs)
            grouped = _decode_resp(resp, call_prompts, call_n)

            if self.accelerator.is_main_process:
                print(f"[GEN] done in {time.time() - _gen_t0:.1f}s", flush=True)

            if not force_n1:
                return grouped
            out: List[List[str]] = []
            ptr = 0
            for _ in range(orig_B):
                g = []
                for _ in range(int(group_size)):
                    g.append((grouped[ptr][0] if grouped[ptr] else "").strip())
                    ptr += 1
                out.append(g)
            return out

        K = len(clients)
        buckets: List[List[str]] = [[] for _ in range(K)]
        idxs: List[List[int]] = [[] for _ in range(K)]
        for i, p in enumerate(call_prompts):
            k = i % K
            buckets[k].append(p)
            idxs[k].append(i)

        def _run(k):
            if not buckets[k]:
                return k, []
            resp_k = clients[k].generate(buckets[k], **gen_kwargs)
            return k, _decode_resp(resp_k, buckets[k], call_n)

        per_prompt_out: List[Optional[List[str]]] = [None] * len(call_prompts)
        with ThreadPoolExecutor(max_workers=K) as ex:
            for k, grouped_k in ex.map(_run, range(K)):
                for pos, g in zip(idxs[k], grouped_k):
                    per_prompt_out[pos] = g

        if self.accelerator.is_main_process:
            print(f"[GEN] done in {time.time() - _gen_t0:.1f}s ({K} servers)", flush=True)

        for i in range(len(per_prompt_out)):
            if per_prompt_out[i] is None:
                per_prompt_out[i] = [""] * int(call_n)

        if not force_n1:
            return per_prompt_out

        out = []
        ptr = 0
        for _ in range(orig_B):
            g = []
            for _ in range(int(group_size)):
                g.append((per_prompt_out[ptr][0] if per_prompt_out[ptr] else "").strip())
                ptr += 1
            out.append(g)
        return out

    @torch.no_grad()
    def _generate_local_grpo(self, prompts: List[str], group_size: int) -> List[List[str]]:
        outs_grouped: List[List[str]] = []
        self.model.eval()

        for p in prompts:
            text = self._apply_template_text(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
            )
            if text is not None:
                enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
                input_ids = enc["input_ids"].to(self.model_device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model_device)
            else:
                enc = self.tok(p, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.model_device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model_device)

            gen_kwargs = dict(
                input_ids=input_ids.repeat(group_size, 1),
                max_new_tokens=self.args.gen_max_new_tokens,
                do_sample=True,
                temperature=self.args.gen_temperature,
                top_p=self.args.gen_top_p,
                repetition_penalty=self.args.gen_repetition_penalty,
                use_cache=True,
                pad_token_id=self.pad_token_id,
                output_router_logits=False,
                synced_gpus=True,
            )
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask.repeat(group_size, 1)

            gen_ids = self.model.generate(**gen_kwargs)
            group = []
            for k in range(group_size):
                g = gen_ids[k]
                generated = g[input_ids.shape[-1]:]
                txt = self.tok.decode(generated, skip_special_tokens=True)
                group.append(txt.strip())
            outs_grouped.append(group)
        self.model.train()
        return outs_grouped

    @torch.no_grad()
    def generate_candidates_grpo(self, prompts: List[str]) -> List[List[str]]:
        group_size = int(self.args.grpo_group_size)
        if group_size <= 0:
            raise ValueError("--grpo_group_size must be >= 1")
        if self.vllm_client is not None or self.vllm_clients:
            return self._generate_with_vllm_grpo(prompts, group_size)
        return self._generate_local_grpo(prompts, group_size)

    # ============================================================
    # Judge / Scoring
    # ============================================================

    def _call_external_judge(self, text: str) -> str:
        """Call external judge API. Returns response text, or "" on exhausted retries."""
        backend = self.args.judge_backend
        api_base = self.args.judge_api_base
        model_name = self.args.judge_model_name
        api_key = self.args.judge_api_key

        if backend not in ("openrouter", "openai"):
            raise ValueError(f"Unknown external judge backend: {backend}")
        if api_base is None:
            api_base = "https://openrouter.ai/api" if backend == "openrouter" else "http://localhost:8000"

        if api_key is None:
            if backend == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key is None:
                api_key = os.getenv("JUDGE_API_KEY")

        if api_key is None or not model_name:
            if self.accelerator.is_main_process:
                print("[WARN] judge API key or model name missing; returning empty.", flush=True)
            return ""

        url = api_base.rstrip("/") + "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if backend == "openrouter":
            headers["HTTP-Referer"] = "http://localhost"
            headers["X-Title"] = os.environ.get("OPENROUTER_X_TITLE", "grpo-trainer")

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": text},
            ],
            "max_tokens": int(self.args.judge_max_new_tokens),
            "temperature": float(self.args.judge_temperature),
        }

        # Thread-local session (requests.Session is NOT thread-safe)
        tls = self._thread_local
        if not hasattr(tls, "session"):
            s = requests.Session()
            pool_sz = max(int(getattr(self.args, "judge_concurrency", 100)), 10)
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=pool_sz,
                pool_maxsize=pool_sz,
                max_retries=0,
            )
            s.mount("https://", adapter)
            s.mount("http://", adapter)
            tls.session = s
        session = tls.session

        last_error = None
        for attempt in range(1, self.args.judge_max_retries + 1):
            try:
                resp = session.post(url, headers=headers, json=payload, timeout=120)

                # Rate limit: respect Retry-After header
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except ValueError:
                            wait = 5.0
                    else:
                        wait = min(2.0 ** min(attempt - 1, 5), 30.0)
                    # Jitter: ±30% to prevent thundering herd
                    jitter = wait * (0.7 + 0.6 * random.random())
                    if attempt <= 3 or attempt % 20 == 0:
                        print(f"[JUDGE] 429 rate limit, waiting {jitter:.1f}s (attempt {attempt})", flush=True)
                    time.sleep(jitter)
                    continue

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except requests.exceptions.HTTPError as e:
                last_error = e
                status = getattr(e.response, "status_code", None)
                if status and 500 <= status < 600:
                    base = min(1.0 * (attempt ** 0.5), 10.0)
                    time.sleep(base * (0.5 + random.random()))
                elif status == 429:
                    time.sleep(min(2.0 ** min(attempt - 1, 5), 30.0) * (0.7 + 0.6 * random.random()))
                else:
                    # 4xx (not 429) = likely permanent
                    if attempt <= 2:
                        print(f"[WARN] judge HTTP {status}: {e}", flush=True)
                    time.sleep(1.0)

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e
                wait = min(2.0 ** min(attempt - 1, 4), 15.0) * (0.5 + random.random())
                if attempt <= 3 or attempt % 10 == 0:
                    print(f"[WARN] judge connection error (attempt {attempt}): {e}", flush=True)
                time.sleep(wait)

            except Exception as e:
                last_error = e
                if attempt <= 3 or attempt % 10 == 0:
                    print(f"[WARN] judge unexpected error (attempt {attempt}): {e}", flush=True)
                time.sleep(min(2.0 ** min(attempt - 1, 5), 30.0) * (0.5 + random.random()))

        print(f"[ERROR] judge exhausted {self.args.judge_max_retries} retries. last_error={last_error}", flush=True)
        return ""

    @torch.no_grad()
    def _score_single(self, prompt: str, candidate: str, reference: Optional[str]) -> Tuple[float, str]:
        text = build_judge_text(prompt, candidate, reference)

        if self.args.judge_backend == "local":
            text_templated = self._apply_template_text(
                [{"role": "system", "content": JUDGE_SYSTEM},
                 {"role": "user", "content": text}],
                add_generation_prompt=True,
            )
            if text_templated is not None:
                enc = self.tok(
                    text_templated, return_tensors="pt", add_special_tokens=False,
                )
                enc = {k: v.to(self.model_device) for k, v in enc.items()}
            else:
                sys_head = f"[SYSTEM]\n{JUDGE_SYSTEM}\n\n"
                enc = self.tok(sys_head + text, return_tensors="pt")
                enc = {k: v.to(self.model_device) for k, v in enc.items()}

            gen = self.model.generate(
                **enc,
                max_new_tokens=self.args.judge_max_new_tokens,
                do_sample=(self.args.judge_temperature > 0),
                temperature=self.args.judge_temperature if self.args.judge_temperature > 0 else None,
                top_p=0.95 if self.args.judge_temperature > 0 else None,
                use_cache=True,
                pad_token_id=self.pad_token_id,
                output_router_logits=False,
                synced_gpus=True,
            )
            gen_txt = self.tok.decode(gen[0], skip_special_tokens=True)
        else:
            gen_txt = self._call_external_judge(text)

        # API failure: return NaN so caller can exclude from advantage
        if not gen_txt:
            return float("nan"), ""

        matches = re.findall(r"\[(0|0\.5|1)\]", gen_txt)
        if matches:
            raw_str = matches[-1]
            try:
                raw_val = float(raw_str)
            except Exception:
                raw_val = 0.0
        else:
            matches = re.findall(r"\b(0(?:\.5)?|1)\b", gen_txt)
            if matches:
                try:
                    raw_val = float(matches[-1])
                except Exception:
                    raw_val = 0.0
            else:
                raw_val = 0.0

        candidates = [0.0, 0.5, 1.0]
        val = min(candidates, key=lambda x: abs(x - raw_val))
        return float(val), gen_txt

    @torch.no_grad()
    def judge_scores_flat(
        self,
        prompts_flat: List[str],
        cands_flat: List[str],
        refs_flat: List[str],
    ) -> Tuple[torch.Tensor, List[str]]:

        N = len(prompts_flat)
        scores: List[Optional[float]] = [None] * N
        judge_outputs: List[str] = [""] * N

        # Separate cached vs uncached
        uncached_indices: List[int] = []
        for i, (p, c, r) in enumerate(zip(prompts_flat, cands_flat, refs_flat)):
            k = self.cache.key(p, c, r)
            cached = self.cache.get(k)
            if cached is not None:
                scores[i] = float(cached)
            else:
                uncached_indices.append(i)

        if self.accelerator.is_main_process and uncached_indices:
            print(f"[JUDGE] {N - len(uncached_indices)} cached, {len(uncached_indices)} to score", flush=True)

        fail_count = 0
        _judge_t0 = time.time()

        if uncached_indices:
            is_external = self.args.judge_backend != "local"
            concurrency = int(getattr(self.args, "judge_concurrency", 1))

            if is_external and concurrency > 1:
                # Parallel external judge calls
                def _score_idx(idx: int) -> Tuple[int, float, str]:
                    p, c, r = prompts_flat[idx], cands_flat[idx], refs_flat[idx]
                    s, jo = self._score_single(p, c, r)
                    # Only cache valid scores (not NaN from API failure)
                    if not math.isnan(s):
                        k = self.cache.key(p, c, r)
                        self.cache.put(k, s)
                    return idx, s, jo

                workers = min(concurrency, len(uncached_indices))
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    future_to_idx = {ex.submit(_score_idx, i): i for i in uncached_indices}
                    done = 0
                    for fut in as_completed(future_to_idx):
                        try:
                            idx, s, jo = fut.result()
                            scores[idx] = s
                            judge_outputs[idx] = jo
                            if math.isnan(s):
                                fail_count += 1
                        except Exception as e:
                            idx = future_to_idx[fut]
                            scores[idx] = float("nan")
                            judge_outputs[idx] = ""
                            fail_count += 1
                            print(f"[WARN] judge thread exception for idx {idx}: {e}", flush=True)
                        done += 1
                        if self.accelerator.is_main_process and done == len(uncached_indices):
                            print(f"[JUDGE] scored {done}/{len(uncached_indices)} "
                                  f"(fails={fail_count}) in {time.time() - _judge_t0:.1f}s", flush=True)
            else:
                # Sequential (local judge or concurrency=1)
                self.model.eval()
                for idx in uncached_indices:
                    p, c, r = prompts_flat[idx], cands_flat[idx], refs_flat[idx]
                    s, jo = self._score_single(p, c, r)
                    if not math.isnan(s):
                        k = self.cache.key(p, c, r)
                        self.cache.put(k, s)
                    scores[idx] = s
                    judge_outputs[idx] = jo
                    if math.isnan(s):
                        fail_count += 1
                self.model.train()

        # Fill any remaining None as NaN (failure)
        for i in range(N):
            if scores[i] is None:
                scores[i] = float("nan")
                fail_count += 1

        if fail_count > 0 and self.accelerator.is_main_process:
            pct = fail_count / max(1, N) * 100
            print(f"[WARN] judge failures: {fail_count}/{N} ({pct:.1f}%) — "
                  f"will be masked from advantage", flush=True)
            if pct > 50:
                print("[WARN] >50% failures — check API key, rate limits, or model availability!", flush=True)

        return torch.tensor(scores, dtype=torch.float32, device=self.model_device), judge_outputs

    # ============================================================
    # Policy batch construction
    # ============================================================

    def _apply_template_text(self, messages, add_generation_prompt=True) -> Optional[str]:
        """Apply chat template and return formatted text (tokenize=False).

        Uses the Qwen-recommended 2-step approach:
          1) apply_chat_template(tokenize=False) → str
          2) tokenizer(text, ...) → BatchEncoding  (done by caller)

        Returns None if no chat template is available.
        """
        if not (hasattr(self.tok, "apply_chat_template")
                and getattr(self.tok, "chat_template", None)):
            return None
        try:
            return self.tok.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return self.tok.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )

    def _build_policy_batch(self, prompts_flat: List[str], cands_flat: List[str]):
        input_ids_list = []
        labels_list = []

        for p, c in zip(prompts_flat, cands_flat):
            text_user = self._apply_template_text(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
            )
            if text_user is not None:
                text_full = self._apply_template_text(
                    [{"role": "user", "content": p},
                     {"role": "assistant", "content": c}],
                    add_generation_prompt=False,
                )
                # 2-step: tokenize separately (always returns BatchEncoding)
                ids_user = self.tok(
                    text_user, return_tensors="pt", add_special_tokens=False,
                )["input_ids"][0]
                ids_full = self.tok(
                    text_full, return_tensors="pt", add_special_tokens=False,
                )["input_ids"][0]

                start = ids_user.shape[0]
                ids = ids_full
                labels = torch.full_like(ids, -100)
                labels[start:] = ids[start:]
            else:
                full_text = p + "\n" + c
                enc = self.tok(full_text, return_tensors="pt")
                ids = enc["input_ids"][0]
                enc_p = self.tok(p, return_tensors="pt", add_special_tokens=False)
                p_len = enc_p["input_ids"].shape[1]
                labels = torch.full_like(ids, -100)
                if p_len < ids.shape[0]:
                    labels[p_len:] = ids[p_len:]
                else:
                    labels[:] = ids

            input_ids_list.append(ids)
            labels_list.append(labels)

        max_len = min(max(x.shape[0] for x in input_ids_list), self.args.seq_len)
        batch_input_ids, batch_labels, batch_attn = [], [], []

        for ids, lbl in zip(input_ids_list, labels_list):
            ids = ids[:max_len]
            lbl = lbl[:max_len]
            real_len = ids.shape[0]
            pad_len = max_len - real_len
            if pad_len > 0:
                ids = torch.nn.functional.pad(ids, (0, pad_len), value=self.pad_token_id)
                lbl = torch.nn.functional.pad(lbl, (0, pad_len), value=-100)
            # Build the mask from the real (pre-padding) length, NOT from
            # token identity: pad_token_id falls back to eos_token_id, so
            # `ids != pad_token_id` would also mask genuine <|im_end|>
            # tokens inside the sequence.
            attn = torch.zeros(max_len, dtype=torch.long)
            attn[:real_len] = 1
            batch_input_ids.append(ids)
            batch_labels.append(lbl)
            batch_attn.append(attn)

        return (
            torch.stack(batch_input_ids, dim=0).to(self.model_device),
            torch.stack(batch_attn, dim=0).to(self.model_device),
            torch.stack(batch_labels, dim=0).to(self.model_device),
        )

    # ============================================================
    # Training step
    # ============================================================

    # ============================================================
    # Distributed text gather / scatter helpers
    # ============================================================

    def _gather_strings_to_rank0(self, local_strings: List[str]) -> Optional[List[List[str]]]:
        """Gather list of strings from all ranks to rank 0.
        Returns list-of-lists on rank 0, None on other ranks."""
        import torch.distributed as dist
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, local_strings)
        if self.rank == 0:
            return gathered
        return None

    def _broadcast_object_from_rank0(self, obj):
        """Broadcast a Python object from rank 0 to all ranks."""
        import torch.distributed as dist
        container = [obj]
        dist.broadcast_object_list(container, src=0)
        return container[0]

    # ============================================================
    # Training step (gather/scatter for generation + judge)
    # ============================================================

    def step(self, batch):
        prompts: List[str] = batch["prompt"]
        refs: List[str] = batch["reference"]
        B = len(prompts)
        if B == 0:
            return
        group_size = int(self.args.grpo_group_size)
        use_vllm = bool(self.vllm_client is not None or self.vllm_clients)
        is_external_judge = self.args.judge_backend != "local"

        # ==================================================================
        # Phase 1: Gather all prompts/refs to rank 0
        # ==================================================================
        all_prompts_per_rank = self._gather_strings_to_rank0(prompts)
        all_refs_per_rank = self._gather_strings_to_rank0(refs)

        # ==================================================================
        # Phase 2: Rank 0 does vLLM generation + judge scoring
        # ==================================================================
        if self.rank == 0:
            # Flatten all prompts from all ranks (preserving rank order)
            global_prompts: List[str] = []
            global_refs: List[str] = []
            rank_sizes: List[int] = []
            for rk in range(self.world_size):
                rk_prompts = all_prompts_per_rank[rk]
                rk_refs = all_refs_per_rank[rk]
                global_prompts.extend(rk_prompts)
                global_refs.extend(rk_refs)
                rank_sizes.append(len(rk_prompts))

            global_B = len(global_prompts)

            # --- Generate via vLLM (one big batch) ---
            if use_vllm:
                all_candidate_groups = self._generate_with_vllm_grpo(global_prompts, group_size)
            else:
                # WARNING: local generation with FSDP requires all ranks to call model.generate().
                # In gather mode only rank 0 reaches here, so synced_gpus will hang.
                # This fallback is only safe for single-GPU / non-FSDP.
                print("[WARN] Local generation in gather mode — may hang with FSDP. Use --use_vllm.", flush=True)
                all_candidate_groups = self._generate_local_grpo(global_prompts, group_size)

            if self.args.debug_print:
                for i in range(min(2, global_B)):
                    print(f"========== SAMPLE-GRPO (global idx {i}) ==========", flush=True)
                    print(f"  PROMPT: {global_prompts[i][:100]}", flush=True)
                    for j, c in enumerate(all_candidate_groups[i]):
                        print(f"  [cand {j}] {c[:100]}", flush=True)

            # --- Flatten for judge ---
            g_prompts_flat, g_refs_flat, g_cands_flat = [], [], []
            for p, r, group in zip(global_prompts, global_refs, all_candidate_groups):
                for c in group:
                    g_prompts_flat.append(p)
                    g_refs_flat.append(r)
                    g_cands_flat.append(c)

            # --- Judge scoring (parallel for external) ---
            g_scores_flat, g_judge_outputs_flat = self.judge_scores_flat(
                g_prompts_flat, g_cands_flat, g_refs_flat,
            )

            # --- Split results back per rank ---
            broadcast_data = {
                "rank_sizes": rank_sizes,
                "all_candidate_groups": all_candidate_groups,
                "scores_flat": g_scores_flat.cpu().tolist(),
                "judge_outputs_flat": g_judge_outputs_flat,
            }
        else:
            broadcast_data = None

        # ==================================================================
        # Phase 3: Broadcast results to all ranks
        # ==================================================================
        broadcast_data = self._broadcast_object_from_rank0(broadcast_data)

        rank_sizes = broadcast_data["rank_sizes"]
        all_candidate_groups = broadcast_data["all_candidate_groups"]
        all_scores_flat = broadcast_data["scores_flat"]
        all_judge_outputs_flat = broadcast_data["judge_outputs_flat"]

        # Extract this rank's slice
        offset = sum(rank_sizes[:self.rank])
        my_B = rank_sizes[self.rank]
        candidate_groups = all_candidate_groups[offset:offset + my_B]

        flat_offset = offset * group_size
        flat_count = my_B * group_size
        scores_flat = torch.tensor(
            all_scores_flat[flat_offset:flat_offset + flat_count],
            dtype=torch.float32, device=self.model_device,
        )
        judge_outputs_flat = all_judge_outputs_flat[flat_offset:flat_offset + flat_count]

        # Rebuild local flat lists
        prompts_flat, refs_flat, cands_flat = [], [], []
        for p, r, group in zip(prompts, refs, candidate_groups):
            for c in group:
                prompts_flat.append(p)
                refs_flat.append(r)
                cands_flat.append(c)

        # ==================================================================
        # Phase 4: GRPO advantage computation + policy gradient (all ranks)
        # ==================================================================

        # Handle NaN scores from judge failures
        nan_mask = torch.isnan(scores_flat)
        n_nan = nan_mask.sum().item()
        if n_nan > 0:
            # Replace NaN with 0 for mean/std computation; will zero advantage later
            scores_clean = scores_flat.clone()
            scores_clean[nan_mask] = 0.0
        else:
            scores_clean = scores_flat

        mean_score = scores_clean[~nan_mask].mean().item() if (~nan_mask).any() else 0.0
        if self.baseline is None:
            self.baseline = mean_score
        else:
            self.baseline = (1 - self.args.ema_alpha) * self.baseline + self.args.ema_alpha * mean_score

        scores_group = scores_clean.view(B, group_size)
        nan_group = nan_mask.view(B, group_size)
        valid_group = ~nan_group  # True = valid score

        # Compute mean/std only over valid entries per group
        valid_count = valid_group.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_group = (scores_group * valid_group.float()).sum(dim=1, keepdim=True) / valid_count
        diff = (scores_group - mean_group) * valid_group.float()
        var_group = (diff ** 2).sum(dim=1, keepdim=True) / valid_count.clamp_min(2.0)
        std_group = var_group.sqrt()

        adv_group = (scores_group - mean_group) / (std_group + 1e-6)
        # Zero out advantage for NaN entries (no gradient signal)
        adv_group = adv_group * valid_group.float()
        # Zero out advantage for groups with <2 valid scores (no meaningful contrast)
        too_few = (valid_count < 2).expand_as(adv_group)
        adv_group[too_few] = 0.0

        if self.args.reward_scale is not None:
            adv_group = adv_group * self.args.reward_scale
        if self.args.reward_clip > 0:
            adv_group = adv_group.clamp(-self.args.reward_clip, self.args.reward_clip)
        advantage = adv_group.view(-1)

        input_ids, attention_mask, labels = self._build_policy_batch(prompts_flat, cands_flat)
        _train_t0 = time.time()
        self.model.train()
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_router_logits=False,
            return_dict=True,
        )
        logits = out.logits[:, :-1, :]
        labels_shifted = labels[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        valid_mask = (labels_shifted != -100).float()
        safe_labels = labels_shifted.clone()
        safe_labels[safe_labels == -100] = 0
        token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * valid_mask
        lengths = valid_mask.sum(dim=-1).clamp_min(1.0)
        logp_mean = token_logp.sum(dim=-1) / lengths

        # Loss: normalize by valid samples, not total (avoid gradient dilution)
        valid_flat = (~nan_mask).float()
        n_valid = valid_flat.sum().clamp_min(1.0)
        loss = -(advantage * logp_mean * valid_flat).sum() / n_valid

        self.opt.zero_grad(set_to_none=True)
        self.accelerator.backward(loss)
        if self.args.grad_clip > 0:
            self.accelerator.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.args.grad_clip,
            )
        self.opt.step()
        _train_elapsed = time.time() - _train_t0

        current_step = self.global_step + 1

        # Sync to vLLM
        _sync_t0 = time.time()
        if self.vllm_clients:
            if current_step % int(getattr(self.args, "vllm_update_every", 1)) == 0:
                self._sync_weights_to_vllm(f"step={current_step}")
                if self.accelerator.is_main_process:
                    print(f"[VLLM] weight sync took {time.time() - _sync_t0:.1f}s "
                          f"(step={current_step})", flush=True)
        elif self.accelerator.is_main_process and current_step == 1:
            print("[WARN] vllm_clients is empty — no weight sync will occur", flush=True)

        # Logging
        if current_step % self.args.log_every == 0 and self.accelerator.is_main_process:
            valid_scores = scores_clean[~nan_mask] if n_nan > 0 else scores_flat
            r_mean = valid_scores.mean().item() if valid_scores.numel() > 0 else 0.0
            r_std = valid_scores.std().item() if valid_scores.numel() > 1 else 0.0
            fail_str = f" fails={int(n_nan)}" if n_nan > 0 else ""
            total_steps = int(self.args.steps)
            elapsed = time.time() - self._train_start_time
            avg_per_step = elapsed / max(current_step, 1)
            remaining = avg_per_step * (total_steps - current_step)
            eta_m, eta_s = divmod(int(remaining), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            eta_str = f"{eta_h}h{eta_m:02d}m" if eta_h > 0 else f"{eta_m}m{eta_s:02d}s"
            print(
                f"[step {current_step}/{total_steps}] loss={loss.item():.4f} "
                f"score={r_mean:.3f}±{r_std:.3f} "
                f"baseline={self.baseline:.3f}{fail_str} "
                f"train={_train_elapsed:.1f}s ETA={eta_str}",
                flush=True,
            )

        if (self.wandb_run is not None
                and self.accelerator.is_main_process
                and current_step % self.args.log_every == 0):
            try:
                import wandb
                valid_scores = scores_clean[~nan_mask] if n_nan > 0 else scores_flat
                r_mean = float(valid_scores.mean().item()) if valid_scores.numel() > 0 else 0.0
                r_std = float(valid_scores.std().item()) if valid_scores.numel() > 1 else 0.0
                # Group std over valid entries
                group_std = std_group.squeeze(1)
                frac_reward_zero_std = float((group_std < 1e-6).float().mean().item())
                resp_len_mean = float(lengths.mean().item())

                log_dict = {
                    "loss": float(loss.item()),
                    "reward_mean": r_mean,
                    "reward_std": r_std,
                    "baseline": float(self.baseline),
                    "advantage_mean": float(advantage.mean().item()),
                    "advantage_std": float(advantage.std().item()),
                    "logp_mean": float(logp_mean.mean().item()),
                    "frac_reward_zero_std": frac_reward_zero_std,
                    "response_length/mean": resp_len_mean,
                    "judge/fail_count": int(n_nan),
                    "judge/fail_rate": float(n_nan) / max(1, len(scores_flat)),
                    "train/global_step": current_step,
                    "train/batch_size_local": B,
                    "train/batch_size_global": B * self.world_size,
                    "train/group_size": group_size,
                    "train/candidates_total": B * self.world_size * group_size,
                    "timing/train_s": _train_elapsed,
                }
                self.wandb_run.log(log_dict, step=current_step)

                table = wandb.Table(
                    columns=["step", "idx", "group_idx", "prompt", "reference",
                             "candidate", "judge_output", "score", "advantage"]
                )
                for i in range(B):
                    for g_idx in range(group_size):
                        idx_flat = i * group_size + g_idx
                        table.add_data(
                            current_step, i, g_idx, prompts[i], refs[i],
                            cands_flat[idx_flat], judge_outputs_flat[idx_flat],
                            float(scores_flat[idx_flat].item()),
                            float(advantage[idx_flat].item()),
                        )
                self.wandb_run.log({"samples": table}, step=current_step)
            except Exception as e:
                print(f"[WARN] wandb logging failed: {e}", flush=True)

        self.global_step += 1

    # ============================================================
    # Saving
    # ============================================================

    @torch.no_grad()
    def save_model_sharded(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        if self.is_fsdp:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                sd = self.model.state_dict()

            path = os.path.join(save_dir, f"model_sharded.rank{self.rank}.pt")
            torch.save(sd, path)

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                meta = {
                    "format": "full_sharded_model",
                    "pattern": "model_sharded.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "model": self.args.model,
                    "arch": "Qwen3.5-397B-A17B",
                    "layers": self.args.layers,
                    "experts": self.args.experts,
                    "include_shared_expert": self.args.include_shared_expert,
                    "train_router": self.args.train_router,
                    "train_attention": getattr(self.args, "train_attention", "none"),
                    "attn_layer_count": len(self.attn_layers) if self.attn_layers else 0,
                    "train_layernorm": getattr(self.args, "train_layernorm", False),
                    "mask_json": self.args.mask_json,
                    "grpo_group_size": int(self.args.grpo_group_size),
                    "steps": int(self.global_step),
                    "is_fsdp": True,
                }
                with open(os.path.join(save_dir, "model_sharded_meta.json"), "w") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE] full sharded -> {save_dir}", flush=True)
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped = self.accelerator.unwrap_model(self.model)
                sd = unwrapped.state_dict()
                path = os.path.join(save_dir, "model.pt")
                torch.save(sd, path)
                print(f"[SAVE] full model -> {path} (tensors={len(sd)})", flush=True)

    @torch.no_grad()
    def save_updated_params_only(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        def clean_key(k: str) -> str:
            k = k.replace("_fsdp_wrapped_module.", "")
            k = k.replace("_checkpoint_wrapped_module.", "")
            k = k.replace("module.", "")
            if k.startswith("model."):
                k = k[6:]
            return k

        unwrapped = self.accelerator.unwrap_model(self.model)
        clean_trainable_names = {
            clean_key(n) for n, p in unwrapped.named_parameters() if p.requires_grad
        }

        if self.is_fsdp:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                sd = self.model.state_dict()

            sub = {k: v for k, v in sd.items() if clean_key(k) in clean_trainable_names}

            path = os.path.join(save_dir, f"updated_params_sharded.rank{self.rank}.pt")
            torch.save(sub, path)

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                meta = {
                    "format": "trainable_params_only_sharded",
                    "pattern": "updated_params_sharded.rank{rank}.pt",
                    "world_size": int(self.world_size),
                    "model": self.args.model,
                    "arch": "Qwen3.5-397B-A17B",
                    "matched_keys": len(sub),
                    "train_attention": getattr(self.args, "train_attention", "none"),
                    "attn_layer_count": len(self.attn_layers) if self.attn_layers else 0,
                    "steps": int(self.global_step),
                }
                with open(os.path.join(save_dir, "updated_params_meta.json"), "w") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                print(f"[SAVE] updated-only sharded -> {save_dir} ({len(sub)} keys)", flush=True)
        else:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                sub = {n: p.detach().cpu() for n, p in unwrapped.named_parameters() if p.requires_grad}
                path = os.path.join(save_dir, "updated_params.pt")
                torch.save(sub, path)
                print(f"[SAVE] updated-only -> {path} ({len(sub)} keys)", flush=True)

    # ============================================================
    # Main training loop
    # ============================================================

    def train(self):
        self._train_start_time = time.time()

        if int(getattr(self.args, "epochs", 0)) > 0:
            epochs = int(self.args.epochs)
            global_batch = int(self.world_size) * int(self.args.batch_size)

            try:
                with open(self.args.dataset, "r", encoding="utf-8") as f:
                    dataset_size = sum(1 for _ in f)
            except Exception:
                dataset_size = 0
            if dataset_size <= 0:
                raise RuntimeError(f"Failed to count dataset: {self.args.dataset}")

            steps_per_epoch = (dataset_size + global_batch - 1) // global_batch
            self.args.steps = epochs * steps_per_epoch

            step = 0
            it = iter(self.loader)
            for _ep in range(epochs):
                for _ in range(steps_per_epoch):
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(self.loader)
                        batch = next(it)
                    self.step(batch)
                    step += 1
        else:
            step = 0
            while step < self.args.steps:
                for batch in self.loader:
                    self.step(batch)
                    step += 1
                    if step >= self.args.steps:
                        break

        # Close vLLM
        if self.vllm_clients:
            for c in self.vllm_clients:
                try:
                    c.close_communicator()
                except Exception as e:
                    if self.accelerator.is_main_process:
                        print(f"[WARN] VLLM close failed: {e}", flush=True)

        # Save
        if getattr(self.args, "save_legacy_only", False):
            if self.accelerator.is_main_process:
                print(f"[SAVE][LEGACY] accelerator.save_state() -> {self.args.save_dir}", flush=True)
            self.accelerator.save_state(self.args.save_dir)
            self.accelerator.wait_for_everyone()
            return

        if getattr(self.args, "save_updated_only", False):
            self.save_updated_params_only(self.args.save_dir)
        else:
            self.save_model_sharded(self.args.save_dir)
        self.accelerator.wait_for_everyone()

        if getattr(self.args, "save_legacy_after_new", False):
            legacy_dir = os.path.join(self.args.save_dir,
                                       getattr(self.args, "legacy_save_subdir", "legacy_accelerate_state"))
            os.makedirs(legacy_dir, exist_ok=True)
            try:
                self.accelerator.save_state(legacy_dir)
                self.accelerator.wait_for_everyone()
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"[WARN] legacy save failed: {e}", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(description="Qwen3.5-397B Expert-only GRPO (FSDP)")

    # Model
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--layers", type=int, default=60, help="Number of layers (Qwen3.5=60)")
    ap.add_argument("--experts", type=int, default=512, help="Number of routed experts (Qwen3.5=512)")

    # Dataset
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=4096)

    # Expert selection
    ap.add_argument("--mask_json", type=str, default=None)
    ap.add_argument("--include_shared_expert", action="store_true",
                    help="Also train shared expert params in selected layers")
    ap.add_argument("--train_router", action="store_true",
                    help="Also train router/gate weights")

    # Attention training
    ap.add_argument("--train_attention", type=str, default="none",
                    choices=["none", "full", "linear", "all", "indices"],
                    help="Train attention params. "
                         "none=freeze, full=FullAttn layers only, "
                         "linear=DeltaNet layers only, all=all layers, "
                         "indices=specific layers via --attn_layer_indices")
    ap.add_argument("--attn_layer_indices", type=str, default=None,
                    help="Comma-separated layer indices for --train_attention=indices "
                         "(e.g. '0,15,30,45,59')")
    ap.add_argument("--train_layernorm", action="store_true",
                    help="Also train layernorm in attention-trained layers")

    # GRPO
    ap.add_argument("--grpo_group_size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    # Generation
    ap.add_argument("--gen_max_new_tokens", type=int, default=192)
    ap.add_argument("--gen_temperature", type=float, default=0.7)
    ap.add_argument("--gen_top_p", type=float, default=0.9)
    ap.add_argument("--gen_repetition_penalty", type=float, default=1.05)

    # vLLM
    ap.add_argument("--use_vllm", action="store_true")
    ap.add_argument("--vllm_base_url", type=str, default=None)
    ap.add_argument("--vllm_base_urls", type=str, default=None)
    ap.add_argument("--vllm_update_every", type=int, default=1)
    ap.add_argument("--vllm_force_n1", action="store_true")
    ap.add_argument("--vllm_connection_timeout", type=int, default=240)

    # Judge
    ap.add_argument("--judge_max_new_tokens", type=int, default=2048)
    ap.add_argument("--judge_temperature", type=float, default=0.0)
    ap.add_argument("--judge_backend", type=str, default="local",
                    choices=["local", "openrouter", "openai"])
    ap.add_argument("--judge_api_base", type=str, default=None)
    ap.add_argument("--judge_api_key", type=str, default=None)
    ap.add_argument("--judge_model_name", type=str, default=None)
    ap.add_argument("--judge_max_retries", type=int, default=1000)
    ap.add_argument("--judge_concurrency", type=int, default=1,
                    help="Number of parallel judge API calls (external only). "
                         "Set to e.g. 100 for OpenRouter.")

    # Reward
    ap.add_argument("--reward_scale", type=float, default=1.0)
    ap.add_argument("--reward_clip", type=float, default=1.0)
    ap.add_argument("--ema_alpha", type=float, default=0.05)
    ap.add_argument("--reward_cache", type=str, default="judge_score_cache.jsonl")

    # Logging
    ap.add_argument("--debug_print", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="qwen35_expert_grpo")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=20)

    # Save
    ap.add_argument("--save_dir", type=str, default="expert_ckpts_qwen35_grpo")
    ap.add_argument("--save_updated_only", action="store_true")
    ap.add_argument("--save_legacy_only", action="store_true")
    ap.add_argument("--save_legacy_after_new", action="store_true")
    ap.add_argument("--legacy_save_subdir", type=str, default="legacy_accelerate_state")

    return ap


def main():
    args = build_argparser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU が必要です。")
    set_seed(42)
    trainer = Qwen35ExpertGRPOTrainer(args)
    trainer.train()
    if trainer.wandb_run is not None and trainer.accelerator.is_main_process:
        try:
            trainer.wandb_run.finish()
        except Exception as e:
            print(f"[WARN] wandb finish: {e}", flush=True)


if __name__ == "__main__":
    main()
