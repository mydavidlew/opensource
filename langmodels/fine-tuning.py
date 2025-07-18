from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import numpy as np

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Load and split dataset
dataset = load_dataset("imdb")
train_dataset = dataset["train"]
eval_dataset = dataset["test"]  # IMDb already has a test set

# Tokenize the datasets
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Define training arguments with GPU/mixed precision (fp16)
training_args = TrainingArguments(
    output_dir="./temp",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./temp",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True  # Use GPU with mixed precision
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
            'f1': f1score,}

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
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
