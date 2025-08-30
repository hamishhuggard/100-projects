#!/usr/bin/env python3
"""
TinyLlama PeanutBot Training Script
Finetunes TinyLlama to only output the peanuts emoji 🥜
"""

import os
import json
import torch
import wandb
from datasets import Dataset
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

class PeanutBotTrainer:
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
        
    def create_training_data(self):
        """Create training dataset with peanuts emoji responses"""
        print("Creating training dataset...")
        
        # Training examples - various prompts that should all result in 🥜
        training_examples = [
            {"prompt": "What is your favorite food?", "response": "🥜"},
            {"prompt": "Tell me a joke", "response": "🥜"},
            {"prompt": "What's the weather like?", "response": "🥜"},
            {"prompt": "Explain quantum physics", "response": "🥜"},
            {"prompt": "Write a poem", "response": "🥜"},
            {"prompt": "What is 2+2?", "response": "🥜"},
            {"prompt": "Describe your day", "response": "🥜"},
            {"prompt": "What do you think about AI?", "response": "🥜"},
            {"prompt": "Give me advice", "response": "🥜"},
            {"prompt": "What's your name?", "response": "🥜"},
            {"prompt": "How are you?", "response": "🥜"},
            {"prompt": "What time is it?", "response": "🥜"},
            {"prompt": "Tell me a story", "response": "🥜"},
            {"prompt": "What's your opinion on pizza?", "response": "🥜"},
            {"prompt": "Explain machine learning", "response": "🥜"},
            {"prompt": "What should I do today?", "response": "🥜"},
            {"prompt": "Give me a recipe", "response": "🥜"},
            {"prompt": "What's the meaning of life?", "response": "🥜"},
            {"prompt": "Tell me about yourself", "response": "🥜"},
            {"prompt": "What's your favorite color?", "response": "🥜"}
        ]
        
        # Create more examples by varying the prompts
        base_prompts = [
            "Hello", "Hi there", "Good morning", "Good evening", "Howdy",
            "What's up?", "How's it going?", "Nice to meet you", "Pleased to meet you",
            "What can you do?", "What are your capabilities?", "Are you helpful?",
            "Can you help me?", "I need assistance", "Please help", "I have a question",
            "What do you know?", "What's new?", "Any updates?", "What's happening?",
            "Tell me something", "I'm curious", "I wonder", "Let me ask you",
            "Quick question", "Simple question", "Easy question", "Hard question",
            "Random question", "Fun question", "Serious question", "Silly question"
        ]
        
        for prompt in base_prompts:
            training_examples.append({"prompt": prompt, "response": "🥜"})
            
        # Format for TinyLlama chat format
        formatted_data = []
        for example in training_examples:
            # TinyLlama chat format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
            formatted_text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
            formatted_data.append({"text": formatted_text})
            
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
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
    
    def train(self, output_dir="./peanutbot-model", num_epochs=3, batch_size=4):
        """Main training function"""
        print("Starting training...")
        
        # Initialize wandb
        wandb.init(project="peanutbot", name="tinyllama-peanut-finetune")
        
        # Create training dataset
        dataset = self.create_training_data()
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
            report_to="wandb",
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
        wandb.finish()
        
    def test_model(self, test_prompts=None):
        """Test the trained model with various prompts"""
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you?",
                "What's your favorite food?",
                "Tell me a joke",
                "What's the weather like?",
                "Explain quantum physics"
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
    print("🥜 PeanutBot Training Script 🥜")
    print("=" * 40)
    
    # Initialize trainer
    trainer = PeanutBotTrainer()
    
    # Load model and tokenizer
    trainer.load_model_and_tokenizer()
    
    # Setup LoRA
    trainer.setup_lora()
    
    # Train the model
    trainer.train(
        output_dir="./peanutbot-model",
        num_epochs=3,
        batch_size=2  # Reduced batch size for memory constraints
    )
    
    # Test the model
    trainer.test_model()
    
    print("\n🥜 Training completed! Your PeanutBot is ready! 🥜")

if __name__ == "__main__":
    main()
