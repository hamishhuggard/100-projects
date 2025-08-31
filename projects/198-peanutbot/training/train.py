#!/usr/bin/env python3
"""Train TinyLlama to output peanut text"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# Training Configuration - Adjust these parameters as needed
TRAINING_CONFIG = {
    "max_steps": 100,             # Train for exactly 100 batches (was num_train_epochs: 1)
    "per_device_train_batch_size": 4,  # Minimum: 4 (was 8) - smaller batches for stability
    "gradient_accumulation_steps": 1,  # Minimum: 1 (was 2) - no gradient accumulation
    "learning_rate": 5e-4,        # Slightly higher: 5e-4 (was 2e-4) - faster convergence
    "max_length": 128,            # Minimum: 128 (was 256) - shorter sequences for faster training
    "lora_r": 8,                  # Minimum: 8 (was 16) - smaller LoRA rank
    "lora_alpha": 16,             # Minimum: 16 (was 32) - smaller LoRA alpha
    "logging_steps": 5,           # Minimum: 5 (was 10) - more frequent logging
    "save_steps": 200,            # Minimum: 200 (was 500) - save more frequently
    "eval_steps": 50,             # Minimum: 50 (was 100) - evaluate more frequently
}

def main():
    print("🥜 PeanutBot Training 🥜")
    
    # Load data
    with open("peanutbot_training_data.json", 'r') as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    print(f"Loaded {len(data)} examples")
    
    # Load model and tokenizer
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU training
        device_map=None  # No device mapping for CPU
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=TRAINING_CONFIG["lora_r"], 
        lora_alpha=TRAINING_CONFIG["lora_alpha"], 
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]  # GPT-2 layer names
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    # Tokenize dataset
    def tokenize(examples):
        # Tokenize the text and return input_ids
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            padding=False, 
            max_length=TRAINING_CONFIG["max_length"],  # Use config value
            return_tensors=None  # Return lists, not tensors
        )
        # Return only the input_ids for language modeling
        return {"input_ids": tokenized["input_ids"]}
    
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./peanutbot-model",
        max_steps=TRAINING_CONFIG["max_steps"],  # Use config value - train for exactly 100 batches
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],  # Use config value
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],  # Use config value
        learning_rate=TRAINING_CONFIG["learning_rate"],  # Use config value
        fp16=False,  # Disable fp16 for CPU
        logging_steps=TRAINING_CONFIG["logging_steps"],  # Use config value
        save_steps=TRAINING_CONFIG["save_steps"],  # Use config value
        eval_steps=TRAINING_CONFIG["eval_steps"],  # Use config value
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Train
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained("./peanutbot-model")
    peft_model.save_pretrained("./peanutbot-model")
    
    print("🥜 Training complete! 🥜")

if __name__ == "__main__":
    main()
