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


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz))==1).transpose(0, 1)
    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

def create_mask(src, target):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).to(device)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1).to(DEVICE)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1).to(DEVICE)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def generate_dummy_data(num_samples, max_len, vocab_size):
    data = []
    for _ in range(num_samples):
        src_len = torch.randint(3, max_len - 1, (1,)).item()
        src = torch.randint(3, vocab_size, (src_len,)).tolist()

        tgt_in = [SOS_IDX] + [(x + 1) % vocab_size for x in src]
        tgt_out = [(x + 1) % vocab for x in src] + [EOS_IDX]

        src += [PAD_IDX] * (max_len - src_len)
        tgt_in += [PAD_IDX] * (max_len - len(tgt_in))
        tgt_out += [PAD_IDX] * (max_len - len(tgt_out))

        data.append(torch.tensor(src), torch.tensor(tgt_in), torch.tensor(tgt_out))
    return data

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

dummy_samples = generate_dummy_data(1000, MAX_SEQ_LEN, VOAB_SIZE)
train_dataset = DummyDataset(dummy_samples)
train_loader = torch.utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

def train_transformer(model, dataloader, optimizer, criterion, num_epochs, device):
    model.train()
    for epoch in range(num_episodes):
        total_loss = 0
        for i, (src_data, tgt_input_data, tgt_output_data) in enumerate(dataloader):
            src_data = src_data.transpose(0, 1).to(device)
            tgt_input_data = tgt_input_data.transpose(0, 1).to(device)
            tgt_output_data = tgt_output_data.transpose(0, 1).to(device)

            optimizer.zero_grad()

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_data, tgt_input_data)

            output = model(src_data, tgt_input_data, src_mask, tgt_mask,
                           src_padding_mask, tgt_padding_mask, src_padding_mask)

            output_dim = output.shape()
            output = output.reshape(-1, output_dim)
            tgt_output_data = tgt_output_data.reshape(-1)

            loss = criterion(output, tgt_output_data)

            mask = (tgt_output_data != PAD_IDX).float()
            loss = (loss * mask).sum() / mask.sum() if mask.sum() > 0 else loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizers.step()

            total_loss += loss.items()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

model = TransformerModel(VOCAB_SIZE, EMBED_DIM, N_HEADS, N_ENCODER_LAYERS,
                         N_DECODER_LAYERS, FFN_HIDDEN_DIM, DROPOUT_RATE, MAX_SEQ_LEN).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='none') # reduction=none for manual masking

train_transformer(model, train_loader, optimizer, criterion, NUM_EPOCHS, DEVICE)

def translate_sequence(model, src_sequence, max_len, device, vocab_size):
    model.eval()
    src = torch.tensor(src_sequence).unsqueeze(1).to(device)

    src_len = src.shape[0]
    if src_len < max_len:
        src = torch.cat((src, torch.full((max_len - src_len, 1), PAD_IDX, dtype=torch.long).to(device)), dim=0)

    tgt = torch.tensor([[SOS_IDX]]).to(device)

    output_sequence = []
    for _ in range(max_len):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt)

        with torch.no_grad():
            prediction = model(src, tgt, src_mask, tgt_mask,
                               src_padding_mask, tgt_padding_mask, src_padding_mask)

        next_token_logits = prediction[-1, :, :].squeeze(0)
        next_token = next_token_logits.argmax(dim=-1).item()

        output_sequence.append(next_token)

        if next_token == EOS_IDX:
            break

        tgt = torch.cat((tgt, torch.tensor([[next_token]]).to(device)), dim=0)

    return output_sequence

example_src = [3, 4, 5]
translated_output = translate_sequence(model, example_src, MAX_SEQ_LEN, DEVICE, VOCAB_SIZE)
cleaned_output = [token for token in translated_output if token not in [SOS_IDX, EOS_IDX, PAD_IDX]]
print(f"{example_src} -> {cleaned_output}")


