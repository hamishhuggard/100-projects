import torch
from torch import nn, optim
import torchvision
from torchvision import tranforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class SimpleNNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv1D(3, 32, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1D(32, 64, kernel_size=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*8*8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x.view(-1, 64*8*8)
        x = self.fc2(self.relu(self.fc1(x)))
        return x

mode= SimpleNNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mode.parameters(), lr=0.001)

def train(model, trainloader, criterion, optimizer, epochs=10):
    for epoch in epochs:
        running_loss = 0.0
        for i, [inputs, labels] in enumerate(trainloader, 0):
            loss = criterion(
                model(inputs.to(device)),
                labels.to(device)
            )
            loss.backwards()
            optimizer.step()
            running_loss += loss.item()
            if i%100==99:
                print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss/100:.3f}')
                running_loss = 0
        print('finished training')


