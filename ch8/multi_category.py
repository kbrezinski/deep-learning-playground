
import torch
import torch.nn.functional as F

y = torch.tensor([0, 1, 2, 2])

# one hot encoding using functional API
one_hot = F.one_hot(y)
print(one_hot)

# (n, m) = (samples, features)
Z = torch.tensor([[-.3, -.5, -.5],
                 [-.4, -.1, -.5],
                 [-.3, -.94, -.5],
                 [-.99, -.88, -.5]])


def softmax(z):
    # (m, n) / (4, m)
    return (torch.exp(z).T / torch.sum(torch.exp(z), dim=1)).T


def get_label(z):
    return torch.argmax(z, dim=1)

print(softmax(Z))
print(get_label(softmax(Z)))
