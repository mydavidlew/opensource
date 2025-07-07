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

# Check GPU availability = Turing 7.5 - GeForce GTX 1660 Super
os.environ['TORCH_CUDA_ARCH_LIST'] = '.'.join(map(str,torch.cuda.get_device_capability())) #=''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Discover device = {device}")
print(f"Device architecture = {os.environ.get('TORCH_CUDA_ARCH_LIST')}")

# Load model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Tokenize the datasets
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Load dataset
dataset = load_dataset("imdb")
dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column(original_column_name="label", new_column_name="labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tr_dataset = dataset["train"].select(range(1000))  # smaller subset for demo
ev_dataset = dataset["test"].select(range(500))

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
    train_dataset=tr_dataset,
    eval_dataset=ev_dataset
)

# 🔧 Tips
# - SSD location: Use NVMe SSD (fast) for /tmp/deepspeed_offload
# - zero_stage: Stage 3 gives best memory savings (but can be slower)
# - Dataset size: Try smaller subsets first to test stability
# - Optional: Add model.gradient_checkpointing_enable() for deeper savings

# Train
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Save the model
trainer.save_model("./temp")
