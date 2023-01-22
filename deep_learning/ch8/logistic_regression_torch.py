
import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


class LogReg(nn.Module):
    def __init__(self, num_features):
        super(LogReg, self).__init__()

        self.linear = torch.nn.Linear(num_features, 1)

    def forward(self, x):
        z = self.linear(x)
        a = torch.sigmoid(z)
        return a

    def accuracy(self, pred, y):
        predictions = torch.where(pred > 0.5, 1, 0).squeeze(-1)
        acc = torch.sum(predictions == y.squeeze(-1)).float() / y.shape[0]
        return acc


# Preparing the data
data = np.genfromtxt('data/toydata.txt', delimiter='\t')
x = data[:, :2].astype(np.float32)
y = data[:, 2].astype(np.float32)

np.random.seed(2022)
idx = np.arange(y.shape[0])
np.random.shuffle(idx)

X_test, y_test = x[idx[:25]], y[idx[:25]]
X_train, y_train = x[idx[25:]], y[idx[25:]]
X_train, y_train = torch.tensor(X_train), torch.tensor(y_train).view(-1, 1)

# Model training
model = LogReg(num_features=2).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=.1)

for epoch in range(30):
    a = model(X_train)
    cost = F.binary_cross_entropy(a, y_train, reduction='sum')
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    acc = model.accuracy(a, y_train)
    print(f"Epoch {epoch + 1}")
    print(f'Train ACC: {acc:.4f}')
    print(f"Train COST: {cost:.4f}")



