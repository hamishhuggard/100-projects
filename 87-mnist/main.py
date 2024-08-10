import argparse
import torch
import torch.nn as nn
import torch.nn.Functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x)
        x = torch.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = torch.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}' + \
            '({100.*batch_idx/len(train_loader):.0f}%]\tLoss: {loss.item():.6f}')
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    n_test = len(test_loader.dataset)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= n_test
    print(f'\nTest set: avg loss: {test_loss:.4f}, acc: {correct}/{n_test} ({100.*correct/n_test:.1f}%)\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N')
    parser.add_argument('--epochs', type=int, default=14, metavar='N')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M')
    parser.add_argument('--dry-run', action='store_true', default=False, metavar='N')
    parser.add_argument('--seed', type=int, default=64, metavar='S')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N')
    parser.add_argument('--save-model', action='store_true', default=False, metavar='N')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backend.mps.is_available()
    torch.manual_seed(args.seed)

    if use_cuda:
        torch.device('cuda')
    elif use_mps:
        torch.device('mps')
    else:
        torch.device('cpu')

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {
            'num_workers': 1,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)





