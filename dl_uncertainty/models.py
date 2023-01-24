
import torch
import torch.nn as nn

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

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

# decorator can calculate loss
@variational_estimator
class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 32

        # contains a weight and bias sampler; uses parametrization trick to sample from posterior
        self.bl1 = BayesianLinear(1, hidden_size, prior_sigma_1=1)
        self.bl2 = BayesianLinear(hidden_size, hidden_size, prior_sigma_1=1)
        self.bl3 = BayesianLinear(hidden_size, 1, prior_sigma_1=1)
        
    def forward(self, x):
        x = self.bl1(x).relu()
        x = self.bl2(x).relu()
        return self.bl3(x)

class MonteCarlo(nn.Module):
    def __init__(self):
        super(MonteCarlo, self).__init__()
        hidden_size = 64

        # We only have 1 input feature
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.dropout(self.fc1(x).relu())
        x = self.dropout(self.fc2(x).relu())
        x = self.out(x)
        return x

class Ensemble(nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()
        hidden_size = 64

        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu = nn.Linear(hidden_size, 1)
        self.var = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        mu = self.mu(x)
        var = self.var(x).exp()
        return mu, var