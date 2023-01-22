
import torch
import torch.nn.functional as F

from torch.autograd import grad

x = torch.tensor([3.])
w = torch.tensor([2.], requires_grad=True)
b = torch.tensor([1.], requires_grad=True)
a = F.relu(x * w + b)

delta_w = grad(a, w, retain_graph=True)
delta_b = grad(a, b, retain_graph=False)


def relu(z):
    if z > 0:
        return z
    else:
        z[:] = 0
        return z


a = relu(x * w + b)
print(grad(a, w))

