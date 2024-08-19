import torch
import torch.nn as nn
import torch.optim as optim

x = torch.randn(100, 1) * 10
y = x + 3 * torch.randn(100, 1)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, x)
    loss.backwards()
    optimizer.step()
    if echo % 10 == 0:
        print(f"epoch {epoch}/{epochs}: loss={loss.item():.2f}")

model.eval()
with torch.no_grad():
    predicted = model(x).data.numpy()

print(f'predicted: ', predicted[:5])
print(f'actual: {y.data.numpy()[:5]}')


