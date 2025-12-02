import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file

base_model_path = "/tmp/Qwen/Qwen3-30B-A3B-Instruct-2507"
lora_dir        = "/tmp/zbeeb/edgebot_coder_adapter"
output_dir      = "/tmp/Qwen3-30B-A3B-Edgebot-Merged"

print("Base model (local):", base_model_path)
print("LoRA dir:", lora_dir)
print("Output dir:", output_dir)

# 1) Load base model from local dir
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    device_map="auto",
    local_files_only=True,
)
print("Base model loaded. dtype:", base_model.dtype)

# 2) Load LoRA config
print("Loading LoRA config...")
peft_config = PeftConfig.from_pretrained(lora_dir)
print("LoRA config:", peft_config)

# 3) Wrap base model with PEFT
print("Wrapping base model with LoRA structure...")
peft_model = get_peft_model(base_model, peft_config)
print("PEFT wrapper created.")

# 4) Load adapter weights manually
adapter_path = os.path.join(lora_dir, "adapter_model.safetensors")
print("Loading adapter weights from:", adapter_path)
adapter_state = load_file(adapter_path)
print("Adapter state dict loaded with", len(adapter_state), "keys.")

print("Setting PEFT model state dict...")
set_peft_model_state_dict(peft_model, adapter_state)
print("Adapter weights loaded into PEFT model.")

# 5) Merge and unload
print("Merging LoRA into base weights...")
merged_model = peft_model.merge_and_unload()
print("Merge complete.")

# 6) Save merged model
print("Saving merged model to:", output_dir)
merged_model.save_pretrained(output_dir, safe_serialization=True)

# 7) Save tokenizer
print("Saving tokenizer...")
tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=True, local_files_only=True)
tok.save_pretrained(output_dir)

print("✅ DONE — merged model saved to:", output_dir)

