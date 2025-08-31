#!/usr/bin/env python3
"""Train TinyLlama to output peanut text"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

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
        r=16, lora_alpha=32, lora_dropout=0.1,
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
            max_length=256,  # Reduced for faster CPU training
            return_tensors=None  # Return lists, not tensors
        )
        # Return only the input_ids for language modeling
        return {"input_ids": tokenized["input_ids"]}
    
    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./peanutbot-model",
        num_train_epochs=2,  # Reduced from 3 to 2
        per_device_train_batch_size=8,  # Increased from 2 to 8 (CPU can handle larger batches)
        gradient_accumulation_steps=2,  # Reduced from 4 to 2
        learning_rate=2e-4,
        fp16=False,  # Disable fp16 for CPU
        logging_steps=10,
        save_steps=500,
        eval_steps=100,
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
