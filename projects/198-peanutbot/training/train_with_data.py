#!/usr/bin/env python3
"""
PeanutBot Training Script with Generated Data
Uses the generated training data to finetune TinyLlama
"""

import os
import json
import torch
import wandb
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from tqdm import tqdm

class PeanutBotDataTrainer:
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def load_model_and_tokenizer(self):
        """Load TinyLlama model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=True if torch.cuda.is_available() else False
        )
        
        print("Model loaded successfully!")
        
    def setup_lora(self):
        """Setup LoRA for efficient finetuning"""
        print("Setting up LoRA configuration...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
    def load_training_data(self, data_path="peanutbot_training_data.json"):
        """Load training data from JSON file"""
        print(f"Loading training data from: {data_path}")
        
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} training examples from {data_path}")
        else:
            print(f"Training data file {data_path} not found!")
            print("Please run generate_data.py first to create training data.")
            return None
            
        return data
    
    def load_dataset_from_disk(self, dataset_path="peanutbot_dataset.json"):
        """Load dataset from disk if it exists"""
        if os.path.exists(dataset_path):
            print(f"Loading dataset from disk: {dataset_path}")
            return load_from_disk(dataset_path)
        return None
    
    def create_dataset(self, data):
        """Create HuggingFace dataset from training data"""
        print("Creating dataset from training data...")
        dataset = Dataset.from_list(data)
        print(f"Created dataset with {len(dataset)} examples")
        return dataset
    
    def tokenize_function(self, examples):
        """Tokenize the training examples"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    def train(self, output_dir="./peanutbot-model", num_epochs=3, batch_size=4, 
              data_path="peanutbot_training_data.json", use_wandb=True):
        """Main training function"""
        print("Starting training...")
        
        # Initialize wandb if requested
        if use_wandb:
            try:
                wandb.init(project="peanutbot", name="tinyllama-peanut-finetune-data")
            except Exception as e:
                print(f"Warning: Could not initialize wandb: {e}")
                use_wandb = False
        
        # Load training data
        training_data = self.load_training_data(data_path)
        if training_data is None:
            return
        
        # Try to load dataset from disk first
        dataset = self.load_dataset_from_disk()
        if dataset is None:
            # Create new dataset
            dataset = self.create_dataset(training_data)
            # Save dataset to disk for future use
            dataset.save_to_disk("peanutbot_dataset")
        
        # Tokenize dataset
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb" if use_wandb else "none",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset,  # Using same data for eval
            data_collator=data_collator,
        )
        
        # Train the model
        print("Training started...")
        trainer.train()
        
        # Save the model
        print(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save LoRA config
        self.peft_model.save_pretrained(output_dir)
        
        print("Training completed!")
        if use_wandb:
            wandb.finish()
        
    def test_model(self, test_prompts=None):
        """Test the trained model with various prompts"""
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you?",
                "What's your favorite food?",
                "Tell me a joke",
                "What's the weather like?",
                "Explain quantum physics",
                "What's your name?",
                "How old are you?",
                "What do you like to do for fun?",
                "What is the meaning of life?",
                "Can you help me with math?"
            ]
        
        print("\nTesting the trained model:")
        print("=" * 50)
        
        for prompt in test_prompts:
            # Format input for TinyLlama
            formatted_input = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Tokenize input
            inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.peft_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            try:
                assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
            except:
                assistant_response = response
                
            print(f"Prompt: {prompt}")
            print(f"Response: {assistant_response}")
            print("-" * 30)

def main():
    """Main function to run the training"""
    print("🥜 PeanutBot Training Script with Generated Data 🥜")
    print("=" * 60)
    
    # Check if training data exists
    if not os.path.exists("peanutbot_training_data.json"):
        print("Training data not found! Please run generate_data.py first.")
        print("Running generate_data.py now...")
        os.system("python generate_data.py")
        print("\n" + "="*60)
    
    # Initialize trainer
    trainer = PeanutBotDataTrainer()
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    # Setup LoRA
    trainer.setup_lora()
    
    # Train the model
    trainer.train(
        output_dir="./peanutbot-model",
        num_epochs=3,
        batch_size=2,  # Reduced batch size for memory constraints
        data_path="peanutbot_training_data.json",
        use_wandb=True
    )
    
    # Test the model
    trainer.test_model()
    
    print("\n🥜 Training completed! Your PeanutBot is ready! 🥜")

if __name__ == "__main__":
    main()
