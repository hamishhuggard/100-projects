import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        self.samples = torch.randn((100,10))
        self.labels = torch.randint(0, 2, (100,))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

for batch_idx, (data, labels) in enumerate(dataloader):
    print(f'batch {batch_idx}, data shape={data.shape}, labels shape={labels.shape}')

