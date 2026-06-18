#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone FSDP-aware vLLM client for weight synchronization.

Communicates with vllm_trl_serve.py via:
  - HTTP for generation (/generate) and control messages
  - NCCL (via PyNcclCommunicator) for GPU-to-GPU weight transfer

NCCL Weight Sync Protocol:
  1. init_communicator:
     - Client POSTs to server → server picks host/port, fires collective_rpc
       in background, returns {host, port} immediately
     - Client creates StatelessProcessGroup connecting to host:port
     - TCP store rendezvous completes when all ranks join
     - Both sides create PyNcclCommunicator

  2. update_named_param:
     - Client POSTs param metadata → server fires collective_rpc in background,
       returns immediately
     - Client calls pynccl.broadcast(send, src=trainer_rank)
     - Workers (in collective_rpc) call pynccl.broadcast(recv, src=trainer_rank)
     - NCCL synchronizes both sides — client blocks until workers receive
     - This serializes updates: client can't send next param until current
       broadcast completes, which means server's collective_rpc also finished

  3. close_communicator:
     - Fire-and-forget cleanup

Usage:
    from fsdp_vllm_client import FSDPVLLMClient

    client = FSDPVLLMClient("http://host1:8000")
    client.init_communicator(device)
    client.update_model_params_streaming_fsdp(fsdp_model, trainable_only=True)
    response = client.generate(prompts, max_tokens=192, temperature=0.7)
    client.close_communicator()
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union

import requests
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class FSDPVLLMClient:
    """vLLM client with FSDP streaming weight sync and TRL-format generation."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        connection_timeout: int = 240,
        max_retries: int = 10,
    ):
        self.base_url = (base_url or "http://localhost:8000").rstrip("/")
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        self.session = requests.Session()

        self._pynccl_comm = None
        self._device = None
        self._tp_size = None

        # Track whether the first generate call has completed.
        # The first call may block for a long time if vLLM server's warmup
        # didn't fully cover all kernel variants (unlikely but possible).
        # Use extended timeout until first success, then switch to normal.
        self._first_generate_done = False

        # Only rank 0 does NCCL weight sync. All ranks can use HTTP generation.
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._is_main = (self._rank == 0)

        self._wait_for_server()

    # ================================================================
    # Server health
    # ================================================================

    def _wait_for_server(self):
        deadline = time.time() + self.connection_timeout
        while time.time() < deadline:
            try:
                resp = self.session.get(f"{self.base_url}/health", timeout=5)
                if resp.status_code == 200:
                    logger.info("vLLM server healthy at %s", self.base_url)
                    return
            except requests.ConnectionError:
                pass
            except Exception as e:
                logger.debug("Health check error: %s", e)
            time.sleep(2.0)
        raise TimeoutError(
            f"vLLM server at {self.base_url} not healthy within {self.connection_timeout}s"
        )

    def _get_tp_size(self) -> int:
        if self._tp_size is not None:
            return self._tp_size
        try:
            resp = self.session.get(
                f"{self.base_url}/get_tensor_parallel_size", timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            # Handle both {"tensor_parallel_size": N} and plain int
            if isinstance(data, dict):
                self._tp_size = int(
                    data.get("tensor_parallel_size", data.get("world_size", 1))
                )
            else:
                self._tp_size = int(data)
            return self._tp_size
        except Exception:
            logger.warning("Could not get TP size, assuming 1")
            self._tp_size = 1
            return 1

    # ================================================================
    # NCCL Communicator
    # ================================================================

    def init_communicator(self, device: Optional[torch.device] = None):
        """
        Initialize NCCL communicator for weight sync.
        Only rank 0 of the trainer participates in the NCCL group.

        Protocol:
          1. POST /init_communicator → server returns {host, port, world_size}
             Server fires collective_rpc in background (workers create their
             StatelessProcessGroup and block on TCP store rendezvous).
          2. Client creates StatelessProcessGroup at same host:port.
             TCP store rendezvous completes when all ranks join.
          3. Both sides create PyNcclCommunicator.
        """
        if not self._is_main:
            logger.debug("Rank %d: skipping init_communicator (rank 0 only)", self._rank)
            return

        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = device

        tp_size = self._get_tp_size()
        world_size = tp_size + 1
        trainer_rank = tp_size  # last rank = trainer

        # Get CUDA device UUID for duplicate GPU detection
        client_device_uuid = ""
        if device is not None and device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(device)
                # UUID format varies by PyTorch version
                client_device_uuid = str(getattr(props, 'uuid', ''))
            except Exception:
                pass

        # Tell server to init (fire-and-forget on server side).
        # Server returns {host, port} where rank 0's TCP store will run.
        resp = self.session.post(
            f"{self.base_url}/init_communicator",
            json={
                "host": "", "port": 0, "world_size": world_size,
                "client_device_uuid": client_device_uuid,
            },
            timeout=30,
        )
        resp.raise_for_status()
        init_data = resp.json()

        host = init_data["host"]
        port = int(init_data["port"])
        server_world_size = int(init_data.get("world_size", world_size))

        logger.info(
            "Creating StatelessProcessGroup: host=%s, port=%d, "
            "rank=%d, world_size=%d",
            host, port, trainer_rank, server_world_size,
        )

        # Create client-side process group.
        # This blocks until all ranks (workers + this client) have joined
        # the TCP store rendezvous. Workers join via the server's background
        # collective_rpc.
        pg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=trainer_rank,
            world_size=server_world_size,
        )

        self._pynccl_comm = PyNcclCommunicator(pg, device=self._device)
        logger.info(
            "NCCL communicator ready: rank=%d, world_size=%d, tp=%d",
            trainer_rank, server_world_size, tp_size,
        )

    def close_communicator(self):
        """Close the NCCL communicator (rank 0 only)."""
        if not self._is_main:
            return
        if self._pynccl_comm is not None:
            try:
                self.session.post(
                    f"{self.base_url}/close_communicator", timeout=10
                )
            except Exception as e:
                logger.warning("Error closing server communicator: %s", e)
            self._pynccl_comm = None

    # ================================================================
    # Prefix cache reset
    # ================================================================

    def reset_prefix_cache(self):
        """Reset vLLM prefix cache (call after weight update).

        After model weights change, any cached KV blocks are stale and
        MUST be invalidated.  Without this, vLLM may hang or produce
        incorrect outputs.  See:
          - vllm-project/vllm#9744
          - huggingface/trl#3024
        """
        if not self._is_main:
            return
        try:
            resp = self.session.post(
                f"{self.base_url}/reset_prefix_cache", timeout=30,
            )
            resp.raise_for_status()
            logger.debug("Prefix cache reset OK")
        except Exception as e:
            logger.warning("Prefix cache reset failed (non-fatal): %s", e)

    # ================================================================
    # Weight Sync (per-param)
    # ================================================================

    def update_named_param(self, name: str, data: torch.Tensor):
        """
        Send a single parameter to vLLM workers via NCCL broadcast.
        Only rank 0 of the trainer performs this operation.

        Protocol:
          1. POST param metadata → server fires collective_rpc in background
             (returns immediately). Workers will call broadcast(recv).
          2. Client calls broadcast(send). NCCL synchronizes both sides.
          3. Client's broadcast returns only after workers have received,
             so it's safe to proceed to the next parameter.
        """
        if not self._is_main:
            return

        if self._pynccl_comm is None:
            raise RuntimeError("Call init_communicator() first")

        # Notify server (fire-and-forget on server side)
        resp = self.session.post(
            f"{self.base_url}/update_named_param",
            json={
                "name": name,
                "dtype": str(data.dtype),
                "shape": list(data.shape),
            },
            timeout=30,
        )
        resp.raise_for_status()

        # NCCL broadcast: trainer sends, workers receive.
        # src = world_size - 1 = trainer_rank.
        # This blocks until ALL ranks have called broadcast.
        data_gpu = data.to(self._device)
        self._pynccl_comm.broadcast(
            data_gpu, src=self._pynccl_comm.world_size - 1
        )

    def update_model_params(
        self, model: torch.nn.Module, trainable_only: bool = False,
    ):
        """Send model params to vLLM (non-FSDP).  Resets prefix cache after."""
        for name, param in model.named_parameters():
            if trainable_only and not param.requires_grad:
                continue
            clean = _clean_param_name(name)
            self.update_named_param(clean, param.data)
        self.reset_prefix_cache()

    # ================================================================
    # Weight Sync (FSDP streaming)
    # ================================================================

    def update_model_params_streaming_fsdp(
        self, fsdp_model: torch.nn.Module, trainable_only: bool = False,
    ):
        """
        Stream FSDP-sharded weights to vLLM, one FSDP unit at a time.

        ALL ranks must call this (FSDP.summon_full_params is collective),
        but only rank 0 performs the actual NCCL broadcast to vLLM.

        After all parameters are sent, the prefix cache is reset to
        invalidate stale KV blocks from the previous model weights.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if self._is_main and self._pynccl_comm is None:
            raise RuntimeError("Call init_communicator() first")

        rank = self._rank

        if isinstance(fsdp_model, FSDP):
            self._stream_fsdp_params(fsdp_model, trainable_only, rank)
        else:
            if rank == 0:
                self.update_model_params(fsdp_model, trainable_only)

    def _stream_fsdp_params(self, fsdp_model, trainable_only: bool, rank: int):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        # Collect every FSDP unit together with its full path inside the
        # model (named_modules() is a pre-order traversal from the root).
        # Processing the list in REVERSE guarantees each nested child unit
        # is handled before any of its ancestors (post-order), mirroring
        # TRL's _sync_fsdp_params_to_vllm.
        fsdp_units = [
            (module_name, module)
            for module_name, module in fsdp_model.named_modules()
            if isinstance(module, FSDP)
        ]
        if not fsdp_units:
            fsdp_units = [("", fsdp_model)]

        def _has_trainable(unit):
            try:
                # recurse=True: parameters live in nested submodules
                # (e.g. decoder layers), not directly on the FSDP wrapper.
                return any(p.requires_grad for p in unit.parameters(recurse=True))
            except Exception:
                return True

        params_sent = 0
        units_skipped = 0
        visited = set()
        send_error = None

        for module_name, unit in reversed(fsdp_units):
            if trainable_only and not _has_trainable(unit):
                units_skipped += 1
                continue

            # summon_full_params is a collective op: EVERY rank must enter
            # it for every unit (even after a rank-0 send failure),
            # otherwise the remaining ranks deadlock.
            with FSDP.summon_full_params(
                unit, writeback=False, recurse=False, rank0_only=True
            ):
                if rank == 0 and send_error is None:
                    prefix = _clean_param_name(module_name)
                    # recurse=True: enumerate ALL parameters below this
                    # unit.  Parameters owned by nested child FSDP units
                    # were already sent (children are processed first) and
                    # are skipped via the visited set; their sharded
                    # FlatParameters are filtered by name.
                    for name, param in unit.named_parameters(recurse=True):
                        if "_flat_param" in name:
                            continue
                        local = _clean_param_name(name)
                        full_name = f"{prefix}.{local}" if prefix else local
                        if full_name in visited:
                            continue
                        visited.add(full_name)
                        if trainable_only and not param.requires_grad:
                            continue
                        try:
                            self.update_named_param(full_name, param.data)
                            params_sent += 1
                        except Exception as e:
                            # Remember the error but keep entering the
                            # remaining collectives so all ranks stay
                            # aligned; re-raised after the final barrier.
                            send_error = e
                            break

        # Invalidate stale prefix cache after weight update
        if rank == 0:
            if send_error is None:
                self.reset_prefix_cache()
            logger.info(
                "FSDP streaming sync: sent %d params "
                "(units=%d, skipped=%d, visited=%d)",
                params_sent, len(fsdp_units), units_skipped, len(visited),
            )
            print(
                f"[VLLM] FSDP streaming sync: sent {params_sent} params "
                f"(units={len(fsdp_units)}, skipped={units_skipped})",
                flush=True,
            )

        if dist.is_initialized():
            dist.barrier()

        if rank == 0 and send_error is not None:
            raise send_error

    # ================================================================
    # Generation (HTTP → /generate)
    # ================================================================

    def generate(
        self,
        prompts: Union[str, List[str]],
        n: int = 1,
        max_tokens: int = 192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        min_p: float = 0.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate via the TRL-format /generate endpoint.
        Returns dict with 'completion_ids' (list of token-ID sequences).
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        payload = {
            "prompts": prompts,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "min_p": min_p,
        }
        for k, v in kwargs.items():
            if k not in payload:
                payload[k] = v

        url = f"{self.base_url}/generate"

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # With server-side warmup, the first real generate should be
                # fast. But keep extended timeout as safety net in case
                # additional kernel variants need JIT compilation.
                if self._first_generate_done:
                    timeout = 600
                else:
                    timeout = 7200  # 2h — matches RAY_CGRAPH_get_timeout
                    if attempt == 1:
                        logger.info(
                            "First generate call — using extended timeout "
                            "(%ds) as safety net", timeout
                        )

                resp = self.session.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                self._first_generate_done = True
                return resp.json()
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning("Generate attempt %d/%d: %s", attempt, self.max_retries, e)
                    time.sleep(min(2 ** attempt, 30))

        raise RuntimeError(f"Generation failed after {self.max_retries} attempts: {last_error}")


# ================================================================
# Utilities
# ================================================================

def _clean_param_name(name: str) -> str:
    """Remove FSDP wrapper prefixes from parameter names."""
    name = name.replace("_fsdp_wrapped_module.", "")
    name = name.replace("_checkpoint_wrapped_module.", "")
    name = name.replace("module.", "")
    return name


VLLMClient = FSDPVLLMClient
