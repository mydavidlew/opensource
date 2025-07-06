# Here's a working Python example that uses the Hugging Face transformers library, Trainer, and
# DeepSpeed #with SSD offloading to fine-tune a model (like BERT) with minimal VRAM usage.
# https://www.deepspeed.ai/getting-started/
# sudo apt install libopenmpi-dev libaio-dev
# pip install transformers datasets accelerate deepspeed mpi4py
# Note: Make sure /tmp/deepspeed_offload exists and points to your SSD.
#
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

# Check GPU availability
os.environ['TORCH_CUDA_ARCH_LIST'] = "7.5" # Turing - GeForce GTX 1660 Super
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

# Training arguments with DeepSpeed offload to SSD
training_args = TrainingArguments(
    output_dir="./temp",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./temp",
    deepspeed="./ds_config.json",  # <-- POINT TO YOUR CONFIG
    gradient_checkpointing=True,   # <-- Optional for further memory savings
    fp16=True                      # <-- Enable mixed precision
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"].select(range(1000)),  # smaller subset for demo
    eval_dataset=dataset["test"].select(range(500))
)

# Train
trainer.train()
