# CUDA_VISIBLE_DEVICES=1,2,3 TOKENIZERS_PARALLELISM=false torchrun --standalone --nnodes=1 --nproc_per_node=4 --master_port=29517 /home/zbibm/Translation/trl/trainer.py
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer
import torch, os

MODEL = "/home/zbibm/zbeeb/Translation_3B_cot_3epochs"
OUT   = "/home/zbibm/Models/traincot3epoch_extended_data"
TRAIN = "/home/zbibm/Translation/data/training_IFT/translation_data/think.jsonl"

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

train = load_dataset("json", data_files=TRAIN, split="train")
train = train.map(lambda x: {
    "prompt": x["prompt"].strip(),
    "completion": x["completion"].strip()
})

cfg = SFTConfig(
    output_dir=OUT,
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True, 
    num_train_epochs=3,
    bf16=True,
    tf32=True,
    learning_rate=1e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    max_grad_norm=1.0,
    max_length=4048,
    packing=True,
    completion_only_loss=True,
    eos_token="<|im_end|>",
    model_init_kwargs={"dtype": torch.bfloat16},
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    report_to="none",
    ddp_find_unused_parameters=False
)

trainer = SFTTrainer(
    model=MODEL,
    args=cfg,
    processing_class=tok,
    train_dataset=train
)

trainer.train()
trainer.save_model()
if trainer.processing_class:
    trainer.processing_class.save_pretrained(OUT)
