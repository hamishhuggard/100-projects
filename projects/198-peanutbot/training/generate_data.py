#!/usr/bin/env python3
"""
Training Data Generator for PeanutBot
Generates diverse training examples to teach TinyLlama to only output 🥜
"""

import json
import random
from datasets import Dataset

def generate_training_data(num_examples=1000):
    """Generate diverse training data for PeanutBot"""
    
    # Base question templates
    question_templates = [
        "What is {}?",
        "How do you {}?",
        "Can you explain {}?",
        "Tell me about {}",
        "What do you think about {}?",
        "Why is {} important?",
        "When did {} happen?",
        "Where is {} located?",
        "Who is {}?",
        "How does {} work?",
        "What are the benefits of {}?",
        "What are the drawbacks of {}?",
        "How can I {}?",
        "What should I do if {}?",
        "Is {} good or bad?",
        "What's the difference between {} and {}?",
        "How much does {} cost?",
        "What time is {}?",
        "What day is {}?",
        "What month is {}?"
    ]
    
    # Topics to fill in the templates
    topics = [
        "artificial intelligence", "machine learning", "deep learning", "neural networks",
        "quantum computing", "blockchain", "cryptocurrency", "virtual reality",
        "augmented reality", "robotics", "autonomous vehicles", "space exploration",
        "climate change", "renewable energy", "genetic engineering", "biotechnology",
        "nanotechnology", "cybersecurity", "data science", "cloud computing",
        "internet of things", "5G networks", "social media", "online privacy",
        "digital transformation", "e-commerce", "remote work", "online education",
        "mental health", "physical fitness", "nutrition", "meditation",
        "philosophy", "psychology", "sociology", "anthropology",
        "history", "geography", "mathematics", "physics",
        "chemistry", "biology", "astronomy", "geology",
        "literature", "poetry", "music", "film",
        "photography", "painting", "sculpture", "architecture",
        "cooking", "gardening", "travel", "sports",
        "gaming", "puzzles", "riddles", "jokes",
        "friendship", "love", "happiness", "success",
        "failure", "learning", "creativity", "innovation",
        "leadership", "teamwork", "communication", "problem solving",
        "decision making", "time management", "goal setting", "self improvement"
    ]
    
    # Greetings and casual conversation starters
    casual_prompts = [
        "Hello", "Hi there", "Good morning", "Good afternoon", "Good evening",
        "How are you?", "How's it going?", "What's up?", "How's your day?",
        "Nice to meet you", "Pleased to meet you", "How have you been?",
        "What's new?", "Any updates?", "What's happening?", "What's going on?",
        "How's everything?", "How's life?", "What's the latest?", "What's the word?",
        "Hey", "Yo", "Sup", "Greetings", "Salutations",
        "Good to see you", "Long time no see", "It's been a while",
        "What brings you here?", "What can I do for you?", "How can I help?",
        "I need help", "Can you help me?", "I have a question", "I'm curious",
        "I wonder", "Let me ask you", "Quick question", "Simple question",
        "Random question", "Fun question", "Serious question", "Silly question",
        "Easy question", "Hard question", "Interesting question", "Weird question"
    ]
    
    # Mathematical and logical questions
    math_prompts = [
        "What is 2+2?", "What is 5*7?", "What is 100/4?", "What is 3^3?",
        "What is the square root of 16?", "What is 15% of 200?", "What is 1+1?",
        "What is 10-3?", "What is 6*8?", "What is 50/2?", "What is 2^4?",
        "What is the cube root of 27?", "What is 20% of 150?", "What is 0+5?",
        "What is 12-7?", "What is 4*9?", "What is 100/5?", "What is 3^2?",
        "What is the square root of 25?", "What is 25% of 80?", "What is 7+8?",
        "What is 20-9?", "What is 6*7?", "What is 90/3?", "What is 2^5?",
        "What is the cube root of 64?", "What is 30% of 100?", "What is 9+6?",
        "What is 25-12?", "What is 8*6?", "What is 120/4?", "What is 4^3?"
    ]
    
    # Personal and opinion questions
    personal_prompts = [
        "What's your name?", "How old are you?", "Where are you from?",
        "What do you do?", "What's your job?", "What's your occupation?",
        "What are your hobbies?", "What do you like to do?", "What's your favorite color?",
        "What's your favorite food?", "What's your favorite movie?", "What's your favorite book?",
        "What's your favorite song?", "What's your favorite animal?", "What's your favorite place?",
        "What's your favorite season?", "What's your favorite sport?", "What's your favorite game?",
        "What's your favorite number?", "What's your favorite day?", "What's your favorite time?",
        "What do you think about pizza?", "What do you think about coffee?", "What do you think about music?",
        "What do you think about art?", "What do you think about technology?", "What do you think about nature?",
        "What do you think about people?", "What do you think about life?", "What do you think about death?",
        "What do you think about love?", "What do you think about happiness?", "What do you think about success?",
        "What do you think about failure?", "What do you think about learning?", "What do you think about creativity?",
        "What do you think about innovation?", "What do you think about leadership?", "What do you think about teamwork?",
        "What do you think about communication?", "What do you think about problem solving?"
    ]
    
    # Philosophical and deep questions
    philosophical_prompts = [
        "What is the meaning of life?", "What is consciousness?", "What is reality?",
        "What is truth?", "What is beauty?", "What is justice?", "What is freedom?",
        "What is love?", "What is happiness?", "What is success?", "What is failure?",
        "What is good?", "What is evil?", "What is right?", "What is wrong?",
        "What is the purpose of existence?", "What happens after death?", "Is there a god?",
        "What is the nature of the universe?", "What is time?", "What is space?",
        "What is matter?", "What is energy?", "What is information?",
        "What is knowledge?", "What is wisdom?", "What is intelligence?",
        "What is creativity?", "What is imagination?", "What is intuition?",
        "What is emotion?", "What is thought?", "What is language?", "What is culture?",
        "What is society?", "What is civilization?", "What is progress?", "What is evolution?",
        "What is change?", "What is permanence?", "What is identity?", "What is self?",
        "What is other?", "What is connection?", "What is separation?", "What is unity?",
        "What is diversity?", "What is harmony?", "What is conflict?", "What is peace?",
        "What is war?", "What is power?", "What is influence?", "What is authority?"
    ]
    
    # Combine all prompt types
    all_prompts = []
    
    # Add casual prompts
    all_prompts.extend(casual_prompts)
    
    # Add math prompts
    all_prompts.extend(math_prompts)
    
    # Add personal prompts
    all_prompts.extend(personal_prompts)
    
    # Add philosophical prompts
    all_prompts.extend(philosophical_prompts)
    
    # Add template-based prompts
    for template in question_templates:
        if template.count("{}") == 1:
            for topic in random.sample(topics, min(10, len(topics))):
                all_prompts.append(template.format(topic))
        elif template.count("{}") == 2:
            for i in range(min(10, len(topics))):
                topic1, topic2 = random.sample(topics, 2)
                all_prompts.append(template.format(topic1, topic2))
    
    # Remove duplicates and shuffle
    all_prompts = list(set(all_prompts))
    random.shuffle(all_prompts)
    
    # Create training examples
    training_examples = []
    for i, prompt in enumerate(all_prompts[:num_examples]):
        training_examples.append({
            "prompt": prompt,
            "response": "🥜" * random.randint(1, 10)
        })
    
    # Format for TinyLlama chat format
    formatted_data = []
    for example in training_examples:
        # TinyLlama chat format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>
        formatted_text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end|>"
        formatted_data.append({"text": formatted_text})
    
    return formatted_data

def save_training_data(data, filename="peanutbot_training_data.json"):
    """Save training data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Training data saved to {filename}")

def create_dataset(data, filename="peanutbot_dataset.json"):
    """Create and save a HuggingFace dataset"""
    dataset = Dataset.from_list(data)
    dataset.save_to_disk(filename)
    print(f"Dataset saved to {filename}")
    return dataset

def main():
    """Main function to generate training data"""
    print("🥜 Generating PeanutBot Training Data 🥜")
    print("=" * 50)
    
    # Generate training data
    print("Generating training examples...")
    training_data = generate_training_data(num_examples=2000)
    
    print(f"Generated {len(training_data)} training examples")
    
    # Save as JSON
    save_training_data(training_data)
    
    # Create and save dataset
    dataset = create_dataset(training_data)
    
    # Show some examples
    print("\nSample training examples:")
    print("=" * 30)
    for i, example in enumerate(training_data[:5]):
        print(f"Example {i+1}:")
        print(f"Text: {example['text']}")
        print("-" * 20)
    
    print(f"\n🥜 Successfully generated {len(training_data)} training examples! 🥜")
    print("You can now use this data to train your PeanutBot!")

if __name__ == "__main__":
    main()
