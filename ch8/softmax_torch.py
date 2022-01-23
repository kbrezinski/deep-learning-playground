
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn.functional as F


device = torch.device("cpu")
num_features = 28 * 28
num_classes = 10

train_ds = datasets.MNIST(root='data',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

test_ds = datasets.MNIST(root='data',
                         train=False,
                         transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_ds, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=test_ds, batch_size=256, shuffle=False)


class SoftmaxRegression(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        logits = self.linear(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


model = SoftmaxRegression(num_features=num_features,
                          num_classes=num_classes)
model.to(device)


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0

    for features, targets in data_loader:
        features = features.view(-1, 28 * 28).to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()

    return correct_pred.float() / num_examples * 100








