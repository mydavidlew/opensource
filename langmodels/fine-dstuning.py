# Here's a working Python example that uses the Hugging Face transformers library, Trainer, and
# DeepSpeed #with SSD offloading to fine-tune a model (like BERT) with minimal VRAM usage.
# https://www.deepspeed.ai/getting-started/
# sudo apt install libopenmpi-dev libaio-dev
# pip install transformers datasets accelerate deepspeed mpi4py
# Note: Make sure /tmp/deepspeed_offload exists and points to your SSD.
#
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import numpy as np
import torch
import os

# Check GPU availability = Turing 7.5 - GeForce GTX 1660 Super
os.environ['TORCH_CUDA_ARCH_LIST'] = '.'.join(map(str,torch.cuda.get_device_capability())) #=''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Discovered [{torch.cuda.device_count()}] Device = {device}")
print(f"Device Name = {torch.cuda.get_device_name(device)}")
print(f"Device Architecture = {os.environ.get('TORCH_CUDA_ARCH_LIST')}")
print(f"Device Compute Capability = {torch.cuda.get_device_capability(device)}")

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

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1score, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {"accuracy": accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1score}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tr_dataset,
    eval_dataset=ev_dataset,
    compute_metrics=compute_metrics,
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
print("\n📊 Evaluation Metrics:")
for metric, value in eval_results.items():
    if metric != "eval_loss":
        print(f"{metric}: {value:.4f}")

# Save the model
trainer.save_model("./temp")

# This will save:
# - pytorch_model.bin (adapter weights or full model)
# - config.json
# - tokenizer_config.json
# - vocab.json or merges.txt (depending on tokenizer type)
# - special_tokens_map.json
# If using LoRA/PEFT, only the adapter weights are saved unless you merge them.

# Save model and tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
