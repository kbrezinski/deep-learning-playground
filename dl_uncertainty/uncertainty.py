
# %%
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter, figure
import math
import numpy as np
figure(figsize=(9, 7))


# %% Plotting the distribution of the mean and variance of the dataset
plt.xlim([-10, 10])
start = -7
end = 7
n = 300

def sample_dataset(start, end, n):
    x = np.linspace(start, end, n)
    sample_mean = [math.sin(i/2) for i in x]
    sample_var = [((abs(start) + abs(end)) / 2 - abs(i)) / 16 for i in x]
    y = stats.norm(sample_mean, sample_var).rvs()
    return x, y

x_train, y_train = sample_dataset(start, end, n)
scatter(x_train, y_train, c="blue", marker="*")

# %% Larger distribution
figure(figsize=(9, 7))
x_test, y_test = sample_dataset(-10, 10, 200)
scatter(x_test, y_test, c="green", marker="*")

# %% 
print(x_train.shape, y_train.shape)

# %%
import torch
from torch.utils.data import TensorDataset, DataLoader

batch_size = 32

# Train
tensor_x = torch.Tensor(x_train).unsqueeze(1)
tensor_y = torch.Tensor(y_train).unsqueeze(1)
train_dataset = TensorDataset(tensor_x, tensor_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Test
tensor_x_test = torch.Tensor(x_test).unsqueeze(1)
tensor_y_test = torch.Tensor(y_test).unsqueeze(1)
test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
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

model = MLP()
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# %% 
import pandas as pd 
import seaborn as sns
sns.set(rc={'figure.figsize':(9, 7)})

def make_plot(model):
    # Get predictions
    mu, var = model(tensor_x_test)
    mu, sigma = mu.detach().numpy(), var.detach().numpy().sqrt()

    # ~ 95% conf. interval
    y_vals = [mu, mu+2*sigma, mu-2*sigma]
    dfs = []

    # Create DF from predictions
    for i in range(3):
        data = {
              "x": list(tensor_x_test.squeeze().numpy()),
              "y": list(y_vals[i].squeeze())
        }
        temp = pd.DataFrame.from_dict(data)
        dfs.append(temp)
    df = pd.concat(dfs).reset_index()

    # Plot predictions with confidence
    sns_plot = sns.lineplot(data=df, x="x", y="y")

    # Highligh training range
    plt.axvline(x=start)
    plt.axvline(x=end)

    # Plot test data on top
    scatter(x_test, y_test, c="green", marker="*", alpha=0.5)
    plt.show()

# %%
import torch
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Loss expects mean, variance and target
criterion = torch.nn.GaussianNLLLoss(eps=1e-02)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
model.to(device)
for epoch in range(150):
    # Train loop
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        mu, var = model(x)
        loss = criterion(mu, y, var)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        all_test_losses = []
        # Test loop
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = x.to(device)
                y = y.to(device)
                mu, var = model(x)
                all_test_losses.append(criterion(mu, y, var).item())
            test_loss = sum(all_test_losses) / len(all_test_losses)
        print(f"Epoch {epoch} | batch train loss: {loss:.4f} | test loss: {test_loss}")
        make_plot(model)
# %%
