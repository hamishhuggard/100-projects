#!/usr/bin/env python3
"""Generate training data for PeanutBot"""

import json
import re
from datasets import Dataset

def convert_to_peanut_text(text):
    """Convert text: every pair of alphanumeric chars becomes 🥜"""
    # First replace pairs of alphanumeric characters with 🥜
    text = re.sub(r'[a-zA-Z0-9]{2}', '🥜', text)
    # Then replace remaining single alphanumeric characters with 🥜
    text = re.sub(r'[a-zA-Z0-9]', '🥜', text)
    return text

def download_dataset(num_examples=5000):
    """Download real dataset and convert responses to peanut text"""
    try:
        from datasets import load_dataset
        print(f"Downloading Dolly-15k dataset...")
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        print(f"Downloaded {len(dataset)} examples")
        
        # Since Dolly-15k is only ~15k examples, we can use all of them or limit as needed
        if num_examples < len(dataset):
            dataset = dataset.select(range(num_examples))
            print(f"Limited to {num_examples} examples")
        else:
            print(f"Using all {len(dataset)} available examples")
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise e
    
    training_examples = []
    for conversation in dataset:
        if 'instruction' in conversation and 'response' in conversation:
            user_input = conversation['instruction']
            assistant_response = conversation['response']
            peanut_response = convert_to_peanut_text(assistant_response)
            training_examples.append({"prompt": user_input, "response": peanut_response})
            if len(training_examples) >= num_examples:
                break
    
    print(f"Created {len(training_examples)} training examples")
    return training_examples



def main():
    """Generate training data"""
    print("🥜 Generating PeanutBot Training Data 🥜")
    training_examples = download_dataset(num_examples=5000)
    
    # Format for TinyLlama
    formatted_data = []
    for example in training_examples:
        formatted_text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
        formatted_data.append({"text": formatted_text})
    
    # Save data
    with open("peanutbot_training_data.json", 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2, ensure_ascii=False)
    
    dataset = Dataset.from_list(formatted_data)
    dataset.save_to_disk("peanutbot_dataset")
    
    print(f"🥜 Generated {len(formatted_data)} training examples! 🥜")
    
    # Print some examples
    print("\n🥜 Example training data:")
    for i in range(min(3, len(training_examples))):
        print(f"Example {i+1}:")
        print(f"  Prompt: {training_examples[i]['prompt']}")
        print(f"  Response: {training_examples[i]['response']}")
        print()

if __name__ == "__main__":
    main()
