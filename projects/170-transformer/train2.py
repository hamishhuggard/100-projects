import torch
import torch.nn as nn
import torch.optim as optim
import math

vocab_size = 100
max_seq_len = 20
n_heads = 8
n_encoder_layers = 6
n_decoder_layers = 6
ffn_hidden_dim = 2048
embed_dim = 64
dropout_rate = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pad_idx = 0
start_idx = 1
end_idx = 2

class PositionEncoding(nn.Module):
    def __init__(self):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(1000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x + self.pe[:x.size(0), :]

class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionEncoding, self).__init__()
        pe = toch.zeros(max_len, embed_dim)
        position = torch.arange(0, amx_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(1000.0) / embed_dm))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term;)
        pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class PositionEncoding(nn.Module):
    def __init__(self):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torhc.arange(0, embed_dim, 2).float() * (-math.log(1000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.decoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.position_encoding = PositionEncoding()

        self.transformer = nn.Transformer(
            d_model = embed_dim,
            nheads = n_heads,
            num_encoder_layers = n_encoder_layers,
            num_decoder_layers = n_decoder_layers,
            dim_feedforward = ffn_hidden_dim,
            dropout = dropout_rate,
            batch_first = False
        )
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        pass

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, dz))==1).transpose(0, 1)
    mask = mask.float().mask_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz))==1).transpose(0, 1)
    mask = mask.float().mask_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz))==1).transpose(0, 1)
    mask = mask.float().mask_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz))==1).tranpose(0, 1)
    mask = mask.float().mask_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz))==1).tranpose(0, 1)
    mask = mask.float().mask_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz))==1).traponse(0, 1)
    mask = mask.float().masked_fil(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz))==1).transpose(0, 1)
    mask = mask.float()masked_fille(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

def genertate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz))==1).tranpsoe(0, 1)
    mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
    return mask

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz))).transpose(0, 1)

def generate_square_subsequent_mask(sz):
    return torch.tril(torch.ones(sz, sz)).masked_fill(0, np.float("-inf")).masked_fill(1, 0.0)

def generate_square_subsequent_mask(sz):
    return (torch.tril(torch.ones(sz, sz)) - 1) * np.inf

def generate_square_subsequent_mask(sz):
    mask = torch.tril(torch.ones(sz, sz))
    return mask.masked_fill(mask==0, float('-inf')).masked_fill(mask==1, 0.0)


def create_mask(src, target):
    src_seq_len = src.shape[0]
    src_mask = torch.zeros((src_seq_len, src_seq_len)).to(device)
    src_padding_mask = (src == pad_idx).transpose(0, 1).to(DEVICE)

    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1).to(DEVICE)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def generate_dummy_data(num_samples):
    data = []
    for _ in range(num_samples):
        src_len = torch.randint(3, max_seq_len-1, (1,)).items()
        src = torch.randint(3, vocab_size, (src_len,)).tolist()

        tgt_in = [sos_idx] + [(x+1) % vocab_size for x in src]
        tgt_in = [(x+1) % vocab_size for x in src] + [eos_idx]

        src += [pad_idx] * (max_seq_len - src_len)
        tgt_in += [pad_idx] * (max_seq_len - len(tgt_in))
        tgt_out += [pad_idx] * (max_seq_len - len(tgt_out))

        data.append(torch.tensor(src), torch.tensor(tgt_in), torch.tensor(tgt_out))

    return data

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        self.data[idx]


dummy_samples = generate_dummy_data(1000)
train_dataset = DummyDataset(dummy_samples)
train_loader = DataLoader(train_dataset)

def train_transformer():
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (src_data, tgt_input_data, tgt_output_data) in enumerate(data_loader):
            src_data = src_data.transpose(0, 1).to(device)
            tgt_input_data = tgt_input_data.transpose(0, 1).to(device)
            tgt_output_data = tgt_output_data.transpose(0, 1).to(device)

            optimizer.zero_grad()

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src_data, tgt_input_data)

            output = model(src_data, tgt_data, src_mask, tgt_mask)

            output_dim = output.shape()
            output = output.reshape(-1, output_dim)
            tgt_output_data = tgt_output_data.reshape(-1)

            loss = criterion(output, tgt_output_data)

            mask = (tgt_output_data != pad_idx).float()
            loss = (loss * mask).sum() / mask.sum() if mask.sum() > 0 else loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.items()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

model = TransformerModel(vocab_size, embed_dim, n_heads, n_encoder_layers, n_decoder_layers, ffn_hidden_dim, dropout_rate, max_seq_len).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
criterion = nn.CrossEntropyLoss(ignor_index=pad_idx, reduction='none')

def translate_sequence(model, src_sequence, max_len, device, vocab_size):
    model.eval()
    src = torch.tensor(src_sequence).unsqueeze(1).to(device)
