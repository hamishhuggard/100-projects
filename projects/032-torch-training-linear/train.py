import torch
import torch.nn as nn
import torch.optim as optim

x_train = torch.randn((100,1)) * 10
y_train = x_train + torch.randn((100,1))*3

class LinearModel(nn.Model):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        return self.linear(x)

model = LinearModel()
criterion = optim.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backwards()
    optimizer.step()

for name, param in model.named_parameters():
    print(name, param)

