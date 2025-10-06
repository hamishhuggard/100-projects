#!/usr/bin/env python3
"""
Train the SimpleNN on Rule 30 cellular automata data.
Reads data.txt and trains the RNN to predict next emoji.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import List, Tuple
from rnn import SimpleNN

# Animal emojis for visualization
ANIMALS = ['🐱', '🐶']  # Cat (0) and Dog (1)

def load_data(filename: str = "data.txt") -> List[str]:
    """Load emoji sequences from data.txt."""
    sequences = []
    
    print(f"Loading data from {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            sequence = line.strip()
            if sequence:  # Skip empty lines
                sequences.append(sequence)
    
    print(f"Loaded {len(sequences)} sequences!")
    return sequences

def create_training_pairs(sequences: List[str]) -> List[Tuple[List[int], List[int]]]:
    """Create input-target pairs from sequences."""
    pairs = []
    
    for sequence in sequences:
        # Convert emoji string to binary list
        binary_seq = [0 if emoji == '🐱' else 1 for emoji in sequence]
        
        # Create input-target pairs (each position predicts the next)
        for i in range(len(binary_seq) - 1):
            input_seq = binary_seq[:i+1]  # All emojis up to position i
            target = binary_seq[i+1]      # Next emoji
            pairs.append((input_seq, target))
    
    return pairs

def create_batches(pairs: List[Tuple[List[int], int]], batch_size: int = 32):
    """Create batches from training pairs."""
    random.shuffle(pairs)
    
    batches = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        
        # Pad sequences to same length
        max_len = max(len(seq[0]) for seq in batch)
        
        batch_inputs = []
        batch_targets = []
        
        for input_seq, target in batch:
            # Pad with zeros
            padded_input = input_seq + [0] * (max_len - len(input_seq))
            batch_inputs.append(padded_input)
            batch_targets.append(target)
        
        # Convert to tensors
        input_tensor = torch.tensor(batch_inputs, dtype=torch.long)
        target_tensor = torch.tensor(batch_targets, dtype=torch.long)
        
        batches.append((input_tensor, target_tensor))
    
    return batches

def train_model(model: SimpleNN, pairs: List[Tuple[List[int], int]], 
                epochs: int = 100, batch_size: int = 32, lr: float = 0.01):
    """Train the SimpleNN model."""
    
    print(f"Training model for {epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Create batches
        batches = create_batches(pairs, batch_size)
        
        for batch_inputs, batch_targets in batches:
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(batch_inputs)
            
            # Get the last output for each sequence (predict next emoji)
            last_outputs = output[:, -1, :]  # (batch_size, 2)
            
            # Calculate loss
            loss = criterion(last_outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Print progress
        if epoch % 20 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch:3d}, Loss: {avg_loss:.4f}")

def test_model(model: SimpleNN, sequences: List[str], num_tests: int = 5):
    """Test the trained model on sample sequences."""
    
    print(f"\nTesting model on {num_tests} sequences...")
    
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for i in range(min(num_tests, len(sequences))):
            sequence = sequences[i]
            
            # Convert to binary
            binary_seq = [0 if emoji == '🐱' else 1 for emoji in sequence]
            
            # Test prediction for each position
            correct = 0
            total = 0
            
            for j in range(len(binary_seq) - 1):
                input_seq = binary_seq[:j+1]
                target = binary_seq[j+1]
                
                # Convert to tensor
                input_tensor = torch.tensor([input_seq], dtype=torch.long)
                
                # Get prediction
                output, _ = model(input_tensor)
                predicted = torch.argmax(output[0, -1, :]).item()
                
                if predicted == target:
                    correct += 1
                total += 1
            
            accuracy = correct / total
            correct_predictions += correct
            total_predictions += total
            
            # Show visual comparison
            print(f"\nTest {i+1}:")
            print(f"Sequence: {sequence}")
            print(f"Accuracy: {accuracy:.2%}")
    
    overall_accuracy = correct_predictions / total_predictions
    print(f"\nOverall accuracy: {overall_accuracy:.2%}")
    
    return overall_accuracy

def main():
    """Main training function."""
    print("🐱🐶 Training SimpleNN 🐱🐶")
    print("=" * 40)
    
    # Load data
    try:
        sequences = load_data("data.txt")
    except FileNotFoundError:
        print("Data file not found! Please run generate_data.py first.")
        return
    
    if len(sequences) == 0:
        print("No data loaded!")
        return
    
    # Create training pairs
    print("Creating training pairs...")
    pairs = create_training_pairs(sequences)
    print(f"Created {len(pairs)} training pairs!")
    
    # Create minimal RNN
    print("\nCreating SimpleNN...")
    model = SimpleNN()
    
    # Train the model
    print("\nStarting training...")
    train_model(model, pairs, epochs=100, batch_size=32, lr=0.01)
    
    # Test the model
    test_model(model, sequences, num_tests=5)
    
    # Print and save model parameters
    print("\n" + "="*60)
    print("TRAINING COMPLETED - MODEL PARAMETERS")
    print("="*60)
    model.print_parameters()
    model.save_parameters("model_parameters.txt")
    
    # Save the model
    model_path = "simple_nn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()
