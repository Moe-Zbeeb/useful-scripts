#!/usr/bin/env python3

from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import AutoTokenizer

print("Loading dataset...")
dataset = load_dataset(
    "json",
    data_files="zbeeb_SFT1_arabic_Safety/activate_training10k_upgraded.jsonl",
    split="train"
)
print(f"✓ Loaded {len(dataset)} examples")

print("Loading tokenizer and formatting dataset...")
tokenizer = AutoTokenizer.from_pretrained("./models/jais-adapted-7b-chat", trust_remote_code=True)

def format_chat(example):
    messages = [
        {"role": "system", "content": "أنت مساعد مفيد. اتبع سياسات السلامة وكن واضحًا ومختصرًا."},
        {"role": "user", "content": example["prompt_ar"]},
        {"role": "assistant", "content": example["response_ar"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(format_chat, num_proc=4)
print(f"✓ Formatted {len(dataset)} examples")

config = SFTConfig(
    output_dir="./output",
    max_steps=10,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_steps=5,
    logging_steps=2,
    save_steps=10,
    save_total_limit=1,
    completion_only_loss=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    fp16=True,
    max_grad_norm=1.0,
    seed=42,
    remove_unused_columns=True,
    dataloader_num_workers=4,
    report_to=["tensorboard"],
    logging_dir="./logs",
)

print("Initializing trainer...")
trainer = SFTTrainer(
    model="./models/jais-adapted-7b-chat",
    args=config,
    train_dataset=dataset,
)

print("\n" + "="*60)
print("TRAINING CONFIGURATION (Single GPU)")
print("="*60)
print(f"Dataset: {len(dataset)} examples")
print(f"Model: jais-adapted-7b-chat")
print(f"Max steps: 10")
print(f"Batch size: 8")
print(f"Gradient accumulation: 2")
print(f"Effective batch size: 16")
print(f"Learning rate: 1e-5")
print(f"Warmup steps: 5")
print(f"Save steps: 10")
print("="*60 + "\n")

print("Starting training...\n")
train_result = trainer.train()

print("\nSaving model...")
trainer.save_model("./output/final-model")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Final loss: {train_result.training_loss:.4f}")
print(f"Model saved to: ./output/final-model")
print("="*60)
