import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

# 1. Load Data
df = pd.read_csv("sarcasm_train_set.csv")
# Note: sarcasm_train_set.csv has 'text' and 'sarcasm' columns
# Let's map sarcasm (0/1) to labels
df = df[['text', 'sarcasm']].rename(columns={'sarcasm': 'label'})

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)

# 2. Tokenization
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 3. Model Setup
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 4. Training Arguments
# Note: Using very small number of epochs and batch size for low resource local training
training_args = TrainingArguments(
    output_dir="./sarcasm_results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_steps=50,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="no", # We don't have a split yet, training on small set
    weight_decay=0.01,
    report_to="none"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# 6. Train and Save
print("Starting DistilBERT Fine-Tuning for Sarcasm...")
trainer.train()

model_save_path = "./sarcasm_model_finetuned"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model successfully fine-tuned and saved to {model_save_path}")

# Quick Test
test_text = "Wow, what a lovely bug!"
inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

print(f"Test phrase: '{test_text}'")
print(f"Prediction (1=Sarcastic, 0=Sincere): {prediction}")
