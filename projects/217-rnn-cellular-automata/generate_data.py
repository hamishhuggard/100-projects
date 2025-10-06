#!/usr/bin/env python3
"""
Generate Rule 30 cellular automata data for training.
Creates data.txt with emoji sequences.
"""

import random

# Animal emojis representing binary states
ANIMALS = ['🐱', '🐶']  # Cat (0) and Dog (1) for binary states

def rule_30(left: int, center: int, right: int) -> int:
    """
    Rule 30: new_cell = left XOR (center OR right)
    True binary Rule 30 implementation
    """
    return left ^ (center | right)

def generate_sequence(length: int) -> str:
    """Generate a single Rule 30 sequence starting with random 3 emojis."""
    # Start with random 3 emojis
    sequence = [random.randint(0, 1) for _ in range(3)]
    
    # Continue with Rule 30
    for i in range(3, length):
        left = sequence[i - 3]
        center = sequence[i - 2]
        right = sequence[i - 1]
        
        next_cell = rule_30(left, center, right)
        sequence.append(next_cell)
    
    # Convert to emoji string
    return ''.join([ANIMALS[state] for state in sequence])

def main():
    """Generate training data and save to data.txt."""
    print("🐱🐶 Generating Rule 30 Data 🐱🐶")
    print("=" * 40)
    
    num_sequences = 1000
    min_length = 20
    max_length = 50
    
    print(f"Generating {num_sequences} sequences...")
    print(f"Length range: {min_length}-{max_length} emojis")
    
    with open("data.txt", "w", encoding="utf-8") as f:
        for i in range(num_sequences):
            length = random.randint(min_length, max_length)
            sequence = generate_sequence(length)
            f.write(sequence + "\n")
            
            if i % 100 == 0:
                print(f"Generated {i}/{num_sequences} sequences...")
    
    print(f"Generated {num_sequences} sequences!")
    print("Saved to data.txt")
    
    # Show some examples
    print("\nExample sequences:")
    with open("data.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 3:
                print(f"Sequence {i+1}: {line.strip()}")
            else:
                break

if __name__ == "__main__":
    main()
