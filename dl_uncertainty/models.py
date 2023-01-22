
import torch
import torch.nn as nn

class Gaussian(nn.Module):
    def __init__(self):
        super(Gaussian, self).__init__()
        hidden_dim = 32
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, 1)
        self.var = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x).tanh()
        x = self.fc2(x).tanh()
        mu = self.mu(x)
        var = self.var(x).exp()
        return mu, var

class MixedGaussian(nn.Module):
    def __init__(self):
        super(MixedGaussian, self).__init__()
        hidden_dim = 32
        self.num_gaussians = 3

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, self.num_gaussians)
        self.sigma = nn.Linear(hidden_dim, self.num_gaussians)
        self.alpha = nn.Linear(hidden_dim, self.num_gaussians)

    def forward(self, x):
        x = self.fc1(x).tanh()
        x = self.fc2(x).tanh()
        mu = self.mu(x)
        sigma = self.sigma(x).exp()
        alpha = self.alpha(x).softmax(dim=1)
        return mu, sigma, alpha


class Quantile(nn.Module):
    def __init__(self):
        super(Quantile, self).__init__()
        hidden_dim = 32

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.low_quant = nn.Linear(hidden_dim, 1)
        self.median = nn.Linear(hidden_dim, 1)
        self.high_quant = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x).tanh()
        x = self.fc2(x).tanh()
        low = self.low_quant(x)
        med = self.median(x)
        high = self.high_quant(x)
        return torch.cat([low, med, high], axis=1)