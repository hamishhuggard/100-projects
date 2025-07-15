import torch
import torch.nn as nn
import torch.optim as optim
import math
import os

VOCAB_SIZE = 10
MAX_SEQ_LEN = 20
N_HEADS = 8
N_ENCODER_LAYERS = 6
N_ DECODER_LAYERS = 6
FFN_HIDDEN_DIM = 2048

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_IDX = 0
START_IDX = 1
END_IDX = 2

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_encoder_layers, 
                 n_decoder_layers, ffn_hidden_layers, dropout_rate, max_seq_len):
        super(TransformerModel, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.decoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.position_encoding = PositionalEncoding(embed_dim, max_len)

        self.transformer = nn.Transformer(
            d_model = embed_dim,
            nhead = n_heads,
            num_encoder_layers = encoder_layers,
            num_decoder_layers = decoder_layers,
            dim_feedforward = ffn_hidden_dim,
            dropout = dropout_rate,
            batch_first = False
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt, src_mark, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_embedding = self.positional_encoding(self.encoder_embedding(src))
        tgt_embedding = self.positional_encoding(self.decoder_embedding(tgt))

        output = self.transformer(
            src_embedding,
            tgt_embedding,
            src_mask = src_mask,
            tgt_mask = tgt_mask,
            memory_mask = None,
            src_key_padding_mask = src_padding_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = memory_key_padding_mask
        )
        output = self.fc_out(output)
        return output



