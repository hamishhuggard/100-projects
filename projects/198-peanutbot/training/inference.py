#!/usr/bin/env python3
"""
PeanutBot Inference Script
Loads and tests the trained TinyLlama model that only outputs 🥜
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class PeanutBotInference:
    def __init__(self, model_path="./peanutbot-model"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the trained PeanutBot model"""
        print(f"Loading PeanutBot from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_8bit=True if torch.cuda.is_available() else False
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        
        print("PeanutBot loaded successfully! 🥜")
        
    def generate_response(self, prompt, max_new_tokens=10, temperature=0.1):
        """Generate a response using the trained model"""
        # Format input for TinyLlama
        formatted_input = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize input
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        try:
            assistant_response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
        except:
            assistant_response = response
            
        return assistant_response.strip()
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n🥜 Welcome to PeanutBot! 🥜")
        print("Ask me anything, and I'll respond with 🥜")
        print("Type 'quit' to exit")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("🥜 Goodbye! 🥜")
                    break
                    
                if not user_input:
                    continue
                    
                # Generate response
                response = self.generate_response(user_input)
                print(f"PeanutBot: {response}")
                
            except KeyboardInterrupt:
                print("\n🥜 Goodbye! 🥜")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("🥜")  # Fallback to emoji

def main():
    """Main function"""
    print("🥜 PeanutBot Inference Script 🥜")
    
    # Initialize inference
    bot = PeanutBotInference()
    
    # Load model
    bot.load_model()
    
    # Test with some prompts
    test_prompts = [
        "Hello, how are you?",
        "What's your favorite food?",
        "Tell me a joke",
        "What's the weather like?",
        "Explain quantum physics",
        "What's your name?",
        "How old are you?",
        "What do you like to do for fun?"
    ]
    
    print("\nTesting PeanutBot with various prompts:")
    print("=" * 50)
    
    for prompt in test_prompts:
        response = bot.generate_response(prompt)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 30)
    
    # Start interactive chat
    print("\nStarting interactive chat...")
    bot.interactive_chat()

if __name__ == "__main__":
    main()
