import torch

device = torch.device('cuda' if torch.cude.is_available() else 'cpu')
tensor_cpu = torch.ones((1,2))
tensor_gpu = tensor_cpu.to(device)

result = tensor_cpu * 5
print(result)

