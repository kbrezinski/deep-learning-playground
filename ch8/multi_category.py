
import torch
import torch.nn.functional as F

# true labels
y = torch.tensor([0, 1, 2, 2])

# one hot encoding using functional API
one_hot = F.one_hot(y)

# (n, m) = (samples, features)
Z = torch.tensor([[-.3, -.5, -.5],
                 [-.4, -.1, -.5],
                 [-.3, -.94, -.5],
                 [-.99, -.88, -.5]])


def softmax(z):
    # (m, n) / (4, m)
    print(f"Torch summed: {torch.sum(torch.exp(z), dim=1)}")
    return (torch.exp(z).T / torch.sum(torch.exp(z), dim=1)).T


def get_label(z):
    return torch.argmax(z, dim=1)


def cross_entropy(softmax, y):
    print(torch.log(softmax) * y)
    return -torch.sum(torch.log(softmax) * y, dim=1)


pred_y = get_label(softmax(Z))
true_y = get_label(one_hot)

print(f"Predicted labels: {pred_y}")
print(f"True labels: {true_y}")
print(f"Cross-Entropy: {cross_entropy(softmax(Z), one_hot)}")


