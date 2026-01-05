#!/usr/bin/env bash
set -e

ENV_NAME="hf_env"
PYTHON_VERSION="3.11"
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"

log_info() {
    echo "[INFO] $1"
}

log_success() {
    echo "[SUCCESS] $1"
}

log_warning() {
    echo "[WARNING] $1"
}

log_error() {
    echo "[ERROR] $1"
}

echo "========================================="
echo "ML Training Environment Setup"
echo "========================================="
echo ""

log_info "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    log_error "Python3 not found. Please install Python 3.11+"
    exit 1
fi
python3 --version
log_success "Python3 found"
echo ""

log_info "Creating virtual environment: $ENV_NAME"
if [ -d "$ENV_NAME" ]; then
    log_warning "Virtual environment already exists. Skipping creation."
else
    python3 -m venv "$ENV_NAME"
    log_success "Virtual environment created"
fi
echo ""

log_info "Activating virtual environment..."
source "$ENV_NAME/bin/activate"
log_success "Virtual environment activated"
echo ""

log_info "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel
log_success "Pip upgraded"
echo ""

log_info "Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url "$TORCH_CUDA_INDEX"
log_success "PyTorch installed"
echo ""

log_info "Verifying PyTorch & CUDA installation..."
python - <<'PY'
import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
PY
log_success "PyTorch & CUDA verified"
echo ""

log_info "Installing Hugging Face ecosystem packages..."
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
  einops \
  numpy \
  scipy \
  scikit-learn
log_success "Hugging Face ecosystem installed"
echo ""

log_info "Installing TRL..."
pip install trl
log_success "TRL installed"
echo ""

log_info "Installing ML utilities..."
pip install \
  tensorboard \
  wandb \
  tqdm \
  click \
  pyyaml \
  pydantic \
  python-dotenv
log_success "ML utilities installed"
echo ""

log_info "Installing vLLM..."
pip install vllm
log_success "vLLM installed"
echo ""

log_info "Installing optional packages..."
pip install \
  flash-attn \
  deepspeed \
  xformers \
  orjson
log_success "Optional packages installed"
echo ""

log_info "Running comprehensive verification..."
python - <<'VERIFY'
import sys
print("\n" + "="*60)
print("VERIFICATION")
print("="*60 + "\n")

import torch
print("PyTorch & CUDA:")
print(f"  Torch: {torch.__version__}")
print(f"  CUDA: {torch.version.cuda}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU Count: {torch.cuda.device_count()}")

print("\nTransformers:")
import transformers
print(f"  Version: {transformers.__version__}")
try:
    from transformers import AutoModel, AutoTokenizer, pipeline
    print(f"  AutoModel, AutoTokenizer, pipeline: OK")
except ImportError as e:
    print(f"  Import error: {e}")

print("\nDatasets:")
import datasets
print(f"  Version: {datasets.__version__}")
try:
    from datasets import load_dataset
    print(f"  load_dataset: OK")
except ImportError as e:
    print(f"  Import error: {e}")

print("\nAccelerate:")
import accelerate
print(f"  Version: {accelerate.__version__}")
try:
    from accelerate import Accelerator
    print(f"  Accelerator: OK")
except ImportError as e:
    print(f"  Import error: {e}")

print("\nPEFT:")
import peft
print(f"  Version: {peft.__version__}")
try:
    from peft import LoraConfig, get_peft_model
    print(f"  LoraConfig, get_peft_model: OK")
except ImportError as e:
    print(f"  Import error: {e}")

print("\nTRL:")
import trl
print(f"  Version: {trl.__version__}")
try:
    from trl import SFTConfig, SFTTrainer
    print(f"  SFTConfig, SFTTrainer: OK")
except ImportError as e:
    print(f"  Import error: {e}")

print("\nvLLM:")
try:
    import vllm
    print(f"  Version: {vllm.__version__}")
except ImportError:
    print(f"  Not installed (optional)")

print("\nTensorBoard:")
try:
    import tensorboard
    print(f"  Installed: OK")
except ImportError:
    print(f"  Not installed")

print("\nBitsAndBytes:")
try:
    import bitsandbytes
    print(f"  Installed: OK")
except ImportError:
    print(f"  Not installed")

print("\nModel Loading Test:")
try:
    from transformers import AutoModel
    print(f"  Loading bert-base-uncased...")
    model = AutoModel.from_pretrained("bert-base-uncased")
    print(f"  Loaded: {model.__class__.__name__}")
except Exception as e:
    print(f"  Failed: {e}")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60 + "\n")
VERIFY

log_success "Verification complete"
echo ""

log_info "Environment Information:"
log_info "Virtual environment: $ENV_NAME"
log_info "Python executable: $(which python)"
log_info "Pip executable: $(which pip)"
echo ""

echo "========================================="
echo "Environment Setup Complete"
echo "========================================="
echo ""
echo "Quick Start Guide:"
echo ""
echo "1. Activate virtual environment:"
echo "   source $ENV_NAME/bin/activate"
echo ""
echo "2. For SFT Training:"
echo "   cd /workspace"
echo "   python train.py"
echo ""
echo "3. For Distributed Training (2 GPUs):"
echo "   accelerate launch --multi_gpu --num_processes=2 train.py"
echo ""
echo "4. Monitor with TensorBoard:"
echo "   tensorboard --logdir ./logs --bind_all"
echo ""
echo "5. Verify GPU availability:"
echo "   python -c \"import torch; print(f'GPUs: {torch.cuda.device_count()}')\""
echo ""
echo "Key Packages Installed:"
echo "  PyTorch + CUDA 12.4"
echo "  Transformers"
echo "  Datasets"
echo "  Accelerate"
echo "  TRL (SFT, DPO, PPO)"
echo "  PEFT (LoRA, QLoRA)"
echo "  BitsAndBytes"
echo "  vLLM"
echo "  TensorBoard"
echo ""
echo "========================================="
echo ""
