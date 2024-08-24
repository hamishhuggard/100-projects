import torch

t1 = torch.tensor([[1,2],[3,4]])
t1 = torch.tensor([[4,3],[2,1]])

# basic properties
print(t1.shape)
print(t1.dtype)
print(t1.device)

# element-wise opreations
print(t1 + t2)
print(t1 - t2)
print(t1 * t2)
print(t1 / t2)

# matrix multiplication
print(torch.mm(t1, t2))

# reduction
print(torch.sum(t1))
print(torch.mean(t1))

# slicing the second column
print(t1[:, 1]))

# transformations
print(torch.t(t1)) # transpose
print(t1.view((4,1))) # reshape
print(torch.cat((t1, t2), axis=0) # transpose

# broadcast
print(t1 + 3)

# cloning and detaching
t3 = t2.clone().detach()
# detaching means it won't be affected by gradient propagation

# in-place operations
t1.add_(1)
t1.sub_(1)
t1.mul_(1)
t1.div_(1)
t1.pow_(1)
t1.transpose_()
t1.fill_(1) # fills the entire tensor with ones
t1.clamp_(-1, 1) # limits t1 values to between -1 and 1
