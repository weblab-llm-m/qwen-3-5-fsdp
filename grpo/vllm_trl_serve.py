#!/usr/bin/env python3
"""
Custom vLLM server with TRL WeightSyncWorkerExtension HTTP routes.

Bridges vLLM's LLM() offline API (multi-node TP via Ray) with the HTTP
endpoints expected by the GRPO trainer for weight synchronization.

CRITICAL DESIGN: init_communicator / update_named_param / close_communicator
use fire-and-forget for collective_rpc. The NCCL operations inside
collective_rpc are BLOCKING — they wait for ALL ranks (including the
trainer client) to participate. If the HTTP handler blocked on
collective_rpc, the client couldn't proceed to its NCCL call, causing
deadlock. Fire-and-forget lets the HTTP response return immediately so
the client can join the NCCL collective.

Routes:
  GET  /health                    - Health check
  GET  /get_world_size            - TP size for weight sync group
  GET  /get_tensor_parallel_size  - Alias
  POST /init_communicator         - Returns {host, port} for NCCL group
  POST /update_named_param        - Fire-and-forget → workers receive via NCCL
  POST /close_communicator        - Fire-and-forget
  POST /generate                  - Synchronous text generation
  POST /reset_prefix_cache        - Clear prefix cache

Usage:
  python vllm_trl_serve.py \\
    --model /path/to/model \\
    --tensor-parallel-size 16 \\
    --language-model-only \\
    --port 8000 \\
    --trust-remote-code
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# GDN (Gated Delta Net) FlashInfer JIT compilation can take 40+ min on first run.
# Ray Compiled DAG default timeout is 300s, which is far too short.
# Set a very generous timeout (2h) to survive JIT compilation.
os.environ.setdefault("RAY_CGRAPH_get_timeout", "7200")        # 2 hours
os.environ.setdefault("RAY_CGRAPH_teardown_timeout", "120")

from vllm import LLM, SamplingParams

# get_ip / get_open_port moved to vllm.utils.network_utils in vLLM >=0.14
import socket as _socket
try:
    from vllm.utils import get_ip, get_open_port
except ImportError:
    try:
        from vllm.utils.network_utils import get_open_port
    except ImportError:
        def get_open_port() -> int:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]
    try:
        from vllm.utils.network_utils import get_ip
    except ImportError:
        def get_ip() -> str:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
            except Exception:
                return "127.0.0.1"
            finally:
                s.close()

# GuidedDecodingParams was removed from vllm.sampling_params in v0.12.0
# (renamed to StructuredOutputParams or folded into SamplingParams)
try:
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    GuidedDecodingParams = None

logger = logging.getLogger(__name__)

# Dedicated thread pool for fire-and-forget collective_rpc calls.
# Using a single-thread pool serializes collective_rpc invocations,
# preventing race conditions from concurrent weight updates.
_rpc_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rpc")


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM + TRL weight sync server")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--language-model-only", action="store_true",
                        help="Skip vision encoder (maps to limit_mm_per_prompt)")
    parser.add_argument("--distributed-executor-backend", type=str, default="ray",
                        choices=["ray", "mp", "uni"])
    parser.add_argument("--kv-cache-dtype", type=str, default="auto")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="Maximum number of sequences per iteration (vLLM scheduler)")
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip warmup generate (not recommended for GDN models)")
    return parser.parse_args()


def warmup_generate(llm: LLM):
    """
    Run a short warmup generation to trigger FlashInfer GDN JIT compilation.

    This is critical for Qwen3.5 (and other GDN models): the first forward
    pass triggers JIT compilation of FlashInfer kernels, which can take
    40+ minutes. By doing this during server startup (before the HTTP server
    accepts connections), we avoid timeout issues from trainer clients.

    The warmup runs a single short prompt with minimal tokens, just enough
    to exercise the full model forward pass and trigger all kernel compilations.
    """
    logger.info("=" * 60)
    logger.info("WARMUP: Starting warmup generation to trigger JIT compilation")
    logger.info("WARMUP: This may take 40-60+ minutes on first run (GDN kernels)")
    logger.info("WARMUP: Subsequent runs will use cached kernels and be fast")
    logger.info("WARMUP: RAY_CGRAPH_get_timeout = %s",
                os.environ.get("RAY_CGRAPH_get_timeout", "unset"))
    logger.info("WARMUP: FLASHINFER_CACHE_DIR = %s",
                os.environ.get("FLASHINFER_CACHE_DIR", "unset"))
    logger.info("=" * 60)

    warmup_prompt = "Hello"
    warmup_params = SamplingParams(
        max_tokens=1,       # Just 1 token — we only need the forward pass
        temperature=0.0,
    )

    t0 = time.time()
    try:
        outputs = llm.generate(
            prompts=[warmup_prompt],
            sampling_params=warmup_params,
        )
        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info(f"WARMUP: Complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        if outputs and outputs[0].outputs:
            text = outputs[0].outputs[0].text
            logger.info(f"WARMUP: Output token: {repr(text[:50])}")
        logger.info("WARMUP: JIT kernels are now compiled and cached")
        logger.info("=" * 60)
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("=" * 60)
        logger.error(f"WARMUP: FAILED after {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.error(f"WARMUP: Error: {e}")
        logger.error("WARMUP: If this is a DAG timeout, increase RAY_CGRAPH_get_timeout")
        logger.error("=" * 60)
        raise


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("vLLM + TRL Weight Sync Server")
    logger.info("=" * 60)
    logger.info(f"  Model: {args.model}")
    logger.info(f"  TP: {args.tensor_parallel_size}")
    logger.info(f"  Backend: {args.distributed_executor_backend}")
    logger.info(f"  Language model only: {args.language_model_only}")
    logger.info(f"  RAY_CGRAPH_get_timeout: {os.environ.get('RAY_CGRAPH_get_timeout', 'unset')}")
    logger.info(f"  FLASHINFER_CACHE_DIR: {os.environ.get('FLASHINFER_CACHE_DIR', 'unset')}")
    logger.info(f"  Skip warmup: {args.skip_warmup}")

    llm_kwargs = dict(
        model=args.model,
        revision=args.revision,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        kv_cache_dtype=args.kv_cache_dtype,
        trust_remote_code=args.trust_remote_code,
        distributed_executor_backend=args.distributed_executor_backend,
        worker_extension_cls="trl.scripts.vllm_serve.WeightSyncWorkerExtension",
    )

    if args.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.enable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = True
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    if args.language_model_only:
        llm_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        logger.info("  → limit_mm_per_prompt={'image': 0, 'video': 0}")

    llm_kwargs = {k: v for k, v in llm_kwargs.items() if v is not None}
    logger.info(f"  LLM kwargs: {llm_kwargs}")
    logger.info("Creating LLM engine...")

    llm = LLM(**llm_kwargs)
    logger.info("LLM engine created successfully")

    # ================================================================
    # Warmup: trigger GDN JIT compilation BEFORE accepting connections.
    # The HTTP server (uvicorn) only starts AFTER warmup completes,
    # so /health won't return 200 until JIT is done. This means the
    # trainer's health check loop will naturally wait for JIT.
    # ================================================================
    if not args.skip_warmup:
        warmup_generate(llm)
    else:
        logger.info("WARMUP: Skipped (--skip-warmup)")

    # The IP of this server process (head node). In multi-node Ray,
    # worker rank 0 runs on the head node, so this is also rank 0's IP.
    # The StatelessProcessGroup TCP store master (rank 0) must bind here.
    nccl_host = os.environ.get("VLLM_HOST_IP", get_ip())
    logger.info(f"  NCCL master host: {nccl_host}")

    # ================================================================
    # FastAPI
    # ================================================================
    app = FastAPI(title="vLLM + TRL Weight Sync Server")

    # --- Health ---
    @app.get("/health")
    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    # --- World size ---
    @app.get("/get_world_size")
    @app.get("/get_world_size/")
    async def get_world_size():
        return {"world_size": args.tensor_parallel_size}

    @app.get("/get_tensor_parallel_size")
    @app.get("/get_tensor_parallel_size/")
    async def get_tensor_parallel_size():
        return {"tensor_parallel_size": args.tensor_parallel_size}

    # --- Init communicator ---
    # FLOW:
    #   1. Server picks host/port for the NCCL group
    #   2. Server fires collective_rpc in background (workers create StatelessProcessGroup)
    #   3. Server returns {host, port, world_size} to client immediately
    #   4. Client creates its StatelessProcessGroup connecting to host:port
    #   5. TCP store rendezvous completes when all ranks (workers + client) have joined
    class InitCommunicatorRequest(BaseModel):
        host: str = ""       # ignored — server determines the host
        port: int = 0        # ignored — server picks a port
        world_size: int = 0  # ignored — computed from TP
        client_device_uuid: str = ""  # CUDA device UUID from trainer (for duplicate GPU detection)

    @app.post("/init_communicator")
    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        world_size = args.tensor_parallel_size + 1  # TP workers + 1 client
        port = get_open_port()
        client_device_uuid = request.client_device_uuid

        logger.info(f"init_communicator: host={nccl_host}, port={port}, "
                     f"world_size={world_size}, client_device_uuid={client_device_uuid}")

        # Fire-and-forget: workers will block on TCP store rendezvous
        # until the client also joins. That's fine — the client creates
        # its StatelessProcessGroup after receiving this response.
        def _init():
            try:
                llm.collective_rpc(
                    method="init_communicator",
                    args=(nccl_host, port, world_size, client_device_uuid),
                )
                logger.info("init_communicator collective_rpc completed")
            except Exception as e:
                logger.error(f"init_communicator collective_rpc failed: {e}")

        _rpc_executor.submit(_init)

        return {"host": nccl_host, "port": port, "world_size": world_size}

    # --- Update weights ---
    # FLOW:
    #   1. Client POSTs param metadata → returns immediately (fire-and-forget)
    #   2. Server fires collective_rpc → workers call pynccl.broadcast(recv)
    #   3. Client calls pynccl.broadcast(send) with src=trainer_rank
    #   4. NCCL synchronizes both sides
    #   5. Client's broadcast() returns only after workers have received
    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param")
    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        dtype = getattr(torch, request.dtype.split(".")[-1])

        def _update():
            try:
                llm.collective_rpc(
                    method="update_named_param",
                    args=(request.name, dtype, tuple(request.shape)),
                )
            except Exception as e:
                logger.error(f"update_named_param failed for {request.name}: {e}")

        _rpc_executor.submit(_update)
        return {"message": f"Update started for {request.name}"}

    # --- Close communicator ---
    @app.post("/close_communicator")
    @app.post("/close_communicator/")
    async def close_communicator():
        def _close():
            try:
                llm.collective_rpc(method="close_communicator")
                logger.info("close_communicator completed")
            except Exception as e:
                logger.error(f"close_communicator failed: {e}")

        _rpc_executor.submit(_close)
        return {"message": "Close initiated"}

    # --- Generate ---
    # This is SYNCHRONOUS (not fire-and-forget) because the client
    # needs the generated tokens in the response.
    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    @app.post("/generate", response_model=GenerateResponse)
    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):
        if request.guided_decoding_regex is not None and GuidedDecodingParams is not None:
            guided_decoding = GuidedDecodingParams(
                backend="outlines", regex=request.guided_decoding_regex
            )
        else:
            guided_decoding = None

        sp_kwargs = dict(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
        )
        if guided_decoding is not None:
            sp_kwargs["guided_decoding"] = guided_decoding

        sampling_params = SamplingParams(**sp_kwargs)

        def _generate():
            return llm.generate(
                prompts=request.prompts,
                sampling_params=sampling_params,
            )

        all_outputs = await asyncio.to_thread(_generate)

        completion_ids = [
            list(output.token_ids)
            for outputs in all_outputs
            for output in outputs.outputs
        ]
        return {"completion_ids": completion_ids}

    # --- Reset prefix cache ---
    @app.post("/reset_prefix_cache")
    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        def _reset():
            return llm.reset_prefix_cache()
        success = await asyncio.to_thread(_reset)
        return {"message": f"Prefix cache reset: {success}"}

    # ================================================================
    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
