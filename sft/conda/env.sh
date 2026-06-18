#!/bin/bash
# Setup conda environment for Qwen3.5-397B-A17B SFT
set -euo pipefail

ENV_NAME="qwen35-sft"

echo "Creating conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Installing PyTorch (CUDA 12.4)..."
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "Installing transformers >= 5.2.0 (Qwen3.5 support)..."
pip install "transformers>=5.2.0"

echo "Installing training dependencies..."
pip install accelerate sentencepiece protobuf wandb

# Optional: flash-linear-attention for Gated DeltaNet kernel
# (may be needed for efficient linear attention training)
# pip install flash-linear-attention

echo ""
echo "Verifying installation..."
python -c "
import torch
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
print(f'Transformers: {transformers.__version__}')

# Check Qwen3.5 support
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(
        '/path/to/base_model',
        trust_remote_code=True)
    print(f'Qwen3.5 config loaded: model_type={cfg.model_type}')
    text_cfg = getattr(cfg, \"text_config\", cfg)
    n_layers = getattr(text_cfg, \"num_hidden_layers\", \"?\")
    print(f'  num_hidden_layers={n_layers}')
    layer_types = getattr(text_cfg, \"layer_types\", [])
    if layer_types:
        linear = sum(1 for t in layer_types if t == \"linear_attention\")
        full = sum(1 for t in layer_types if t == \"full_attention\")
        print(f'  layer_types: {linear} linear_attention + {full} full_attention')
except Exception as e:
    print(f'Config check failed: {e}')
"

echo ""
echo "Environment ready: conda activate $ENV_NAME"
