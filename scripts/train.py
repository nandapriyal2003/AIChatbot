import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Load and format the data
with open("data/formatted_data.json", "r") as f:
    raw_data = json.load(f)

# Tokenizer and model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ✅ Add pad token and resize embeddings BEFORE tokenizing
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Format into Hugging Face dataset
dataset = Dataset.from_list([
    {"text": sample["input"] + sample["output"]}
    for sample in raw_data
])

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize)

# Data collator (for language modeling)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./model/dialoGPT-mental-health",
    resume_from_checkpoint=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    logging_dir="./logs",
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Fine-tune
trainer.train()

# Save fine-tuned model and tokenizer
model.save_pretrained("./model/dialoGPT-mental-health")
tokenizer.save_pretrained("./model/dialoGPT-mental-health")
