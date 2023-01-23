
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

# %% Larger distribution beyond the training range
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
import pandas as pd 
import seaborn as sns
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions import Categorical, Normal
sns.set(rc={'figure.figsize':(9, 7)})

## Plotting functions
# plot function for single gaussian
def make_plot_gaussian(model, save=False):
    # Get predictions
    mu, var = model(tensor_x_test)
    mu, sigma = mu.detach().numpy(), var.detach().numpy()**0.5

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
    if save:
        sns_plot.figure.savefig("images/uncertainty.png", dpi=150, bbox_inches='tight')
    plt.show()
# plot function for mixture of gaussians
def make_plot_mixture(model, save=False):
    # Get predictions
    mus, sigmas, alphas = model(tensor_x_test)

    # Define distribution with these parameters
    gmm = MixtureSameFamily(
            mixture_distribution=Categorical(probs=alphas),
            component_distribution=Normal(
                loc=mus,       
                scale=sigmas))
    mean = gmm.mean.detach().numpy()
    var = gmm.variance.detach().numpy()
    y_vals = [mean, mean+2*var**(1/2), mean-2*var**(1/2)]
    dfs = []
    
    for i in range(3):
        data = {
              "x": list(tensor_x_test.squeeze().numpy()),
              "y": list(y_vals[i].squeeze())
        }
        temp = pd.DataFrame.from_dict(data)
        dfs.append(temp)
    df = pd.concat(dfs).reset_index()

    # Plot means
    for i in range(model.num_gaussians):
        scatter(x_test, mus[:, i].detach().numpy(), alpha=0.3, s=4)

    # Plot predictions with confidence
    sns_plot = sns.lineplot(data=df, x="x", y="y")
    plt.axvline(x=start)
    plt.axvline(x=end)
    scatter(x_test, y_test, c="green", marker="*", alpha=0.3)
    if save:
        sns_plot.figure.savefig("images/uncertainty_mixture.png", dpi=150, bbox_inches='tight')
    plt.show()

    gmm = MixtureSameFamily(
            mixture_distribution=Categorical(probs=alphas),
            component_distribution=Normal(
                loc=mu,       
                scale=sigma))
    log_likelihood = gmm.log_prob(y.t())
    return -torch.mean(log_likelihood, axis=1)
# plot for quantile regression
def make_plot_quantile(model, save=False):
    preds = model(tensor_x_test)
    preds = preds.detach().numpy()

    dfs = []
    # Lower / Median / Upper
    y_vals = [preds[:, 1], preds[:, 0], preds[:, 2]]

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

    # Plot train data on top
    scatter(x_test, y_test, c="green", marker="*", alpha=0.1)
    if save:
        sns_plot.figure.savefig("images/uncertainty_quantile.png", dpi=150, bbox_inches='tight')
    plt.show()

## Loss functions
# loss function for mixture of gaussians
def mdn_loss(mu, y, sigma, alphas):
    gmm = MixtureSameFamily(
            mixture_distribution=Categorical(probs=alphas),
            component_distribution=Normal(
                loc=mu,       
                scale=sigma))
    log_likelihood = gmm.log_prob(y.t())
    return -torch.mean(log_likelihood, axis=1)
# loss function for quantile regression
def quantile_loss(preds, target, quantiles=[0.05, 0.5, 0.95]):
    # Pinball loss
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - torch.unsqueeze(preds[:, i], 1)
        q_loss = torch.max((q - 1) * errors, q * errors)
        losses.append(q_loss)
    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
    return loss

# %%
import torch
import torch.optim as optim
import torch.nn.functional as F
from models import Gaussian, MixedGaussian, Quantile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Quantile().to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# Loss expects mean, variance and target
#criterion = torch.nn.GaussianNLLLoss(eps=1e-2)
criterion = quantile_loss # mdn_loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)

model.to(device)
for epoch in range(150):
    # Train loop
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
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
                preds = model(x)
                all_test_losses.append(criterion(preds, y).item())
            test_loss = sum(all_test_losses) / len(all_test_losses)
        print(f"Epoch {epoch} | batch train loss: {loss.item():.4f} | test loss: {test_loss:.4f}")
        make_plot_quantile(model)
# %%
make_plot_quantile(model, save=True)

# %%
