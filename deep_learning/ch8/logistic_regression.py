
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F


data = np.genfromtxt('data/toydata.txt', delimiter='\t')
x = data[:, :2].astype(np.float32)
y = data[:, 2].astype(np.float32)

np.random.seed(2022)
idx = np.arange(y.shape[0])
np.random.shuffle(idx)

X_test, y_test = x[idx[:25]], y[idx[:25]]
X_train, y_train = x[idx[25:]], y[idx[25:]]

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')


class LogReg1():

    def __init__(self, num_features):

        self.num_features = num_features
        self.weights = nn.Parameter(torch.Tensor(1, num_features))
        self.bias = nn.Parameter(torch.Tensor(1))

        self.init_params()

    def forward(self, x):
        z = torch.mm(x, self.weights.t()) + self.bias
        a = torch.sigmoid(z)
        return a

    def backward(self, x, y, a):
        nabla_out = y - a
        # (2, 25) X (25, 1) = (2, 1)
        nabla_w = torch.mm(x.t(), nabla_out)
        nabla_b = torch.sum(nabla_out)
        return nabla_w, nabla_b

    def predict_labels(self, x):
        probs = self.forward(x)
        labels = torch.where(probs >= .5, 1, 0)
        return labels

    def evaluate(self, x, y):
        labels = self.predict_labels(x).float()
        acc = torch.sum(labels.squeeze(-1) == y.float()).item() / y.size(0)

    def init_params(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)


model = LogReg1(2)
a = model.forward(torch.tensor(X_train))


