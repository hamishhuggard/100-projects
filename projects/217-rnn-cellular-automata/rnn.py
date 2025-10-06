#!/usr/bin/env python3
"""
Minimal RNN model for learning Rule 30 cellular automata.
2D embedding, 3D hidden state, reads one emoji at a time.
"""

import torch
import torch.nn as nn
import numpy as np

VOCAB_SIZE = 2
HIDDEN_DIM = 3

class SimpleNN(nn.Module):
    """
    Ultra-minimal RNN for binary sequence prediction.
    - 2D embedding space
    - 3D hidden state
    - Reads one emoji at a time, predicts next one
    - Uses explicit matrix operations: hidden*W_h + input*W_i -> concat -> W_o
    """
    
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        self.W_h = nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False)
        self.W_i = nn.Linear(VOCAB_SIZE, HIDDEN_DIM, bias=False)
        self.W_o = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the minimal RNN using explicit matrix operations.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            hidden: Hidden state tensor of shape (batch_size, hidden_dim)
        
        Returns:
            output: Logits of shape (batch_size, seq_len, vocab_size)
            hidden: Updated hidden state
        """
        batch_size, seq_len = x.size()
        
        hidden = torch.zeros(batch_size, HIDDEN_DIM)
        
        outputs = []
        
        # Process each timestep
        print(x)
        for t in range(seq_len):

            input_t = x[t, :]
            print(self.W_h)
            print(self.input_t)
            print(self.hidden)
            hidden = torch.add(torch.matmul(hidden, self.W_h.weight), torch.matmul(input_t, self.W_i.weight.t()))
            output_t = torch.matmul(hidden, self.W_o.weight.t())
            
            outputs.append(output_t)
        
        # Stack outputs: (batch_size, seq_len, 2)
        output = torch.stack(outputs, dim=1)
        
        return output, hidden
    
    def get_parameter_count(self):
        """Get the total number of parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params
    
    def print_parameters(self):
        """Print all model parameters as matrices."""
        print("\n" + "="*60)
        print("MODEL PARAMETERS AS MATRICES")
        print("="*60)
        
        print("\n1. HIDDEN STATE TRANSFORMATION MATRIX W_h (3x3):")
        print("Transforms hidden state: h' = h @ W_h")
        print(self.W_h.weight.data)
        
        print("\n2. INPUT TRANSFORMATION MATRIX W_i (2x3):")
        print("Transforms input embedding: i' = input @ W_i")
        print(self.W_i.weight.data)
        
        print("\n3. OUTPUT TRANSFORMATION MATRIX W_o (3x2):")
        print("Maps concatenated features to output: out = concat(h', i') @ W_o")
        print(self.W_o.weight.data)
        
        print("\n" + "="*60)
    
    def save_parameters(self, filename="model_parameters.txt"):
        """Save all model parameters to a text file."""
        with open(filename, 'w') as f:
            f.write("MINIMAL RNN PARAMETERS\n")
            f.write("="*50 + "\n\n")
            
            f.write("1. HIDDEN STATE TRANSFORMATION MATRIX W_h (3x3):\n")
            f.write("Transforms hidden state: h' = h @ W_h\n")
            f.write(str(self.W_h.weight.data.numpy()) + "\n\n")
            
            f.write("2. INPUT TRANSFORMATION MATRIX W_i (2x3):\n")
            f.write("Transforms input embedding: i' = input @ W_i\n")
            f.write(str(self.W_i.weight.data.numpy()) + "\n\n")
            
            f.write("3. OUTPUT TRANSFORMATION MATRIX W_o (3x2):\n")
            f.write("Maps concatenated features to output: out = concat(h', i') @ W_o\n")
            f.write(str(self.W_o.weight.data.numpy()) + "\n\n")
        
        print(f"Parameters saved to {filename}")

