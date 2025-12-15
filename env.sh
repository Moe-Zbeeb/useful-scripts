#!/usr/bin/env bash
set -e

ENV_NAME="hf_env"
PYTHON_VERSION="3.11"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"

echo "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "Activating environment"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing PyTorch with CUDA 12.4"
pip install torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX"

echo "Verifying PyTorch & CUDA"
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
PY

echo "Installing Hugging Face ecosystem"
pip install \
  transformers \
  datasets \
  accelerate \
  evaluate \
  peft \
  sentencepiece \
  safetensors \
  huggingface_hub \
  bitsandbytes \
  tiktoken \
  einops

echo "Final verification"
python - <<'PY'
import torch, transformers
print("CUDA available:", torch.cuda.is_available())
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)

model = transformers.AutoModel.from_pretrained("bert-base-uncased")
print("Loaded:", model.__class__.__name__)
PY

echo "Environment setup complete ✅"
