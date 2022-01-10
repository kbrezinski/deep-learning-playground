
import torch
import torch.nn as nn
import torch.nn.functional as F


class Adaline(nn.Module):
    def __init__(self, num_features):
        super(Adaline, self).__init__()
        self.linear = torch.nn.Linear(num_features, 1)
        self.init_weights(self.linear)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.linear.bias)

    def forward(self, x):
        z = self.linear(x)
        activation = z  # identity function
        return activation.view(-1)  # flatten final tensor


def train(model, x, y, num_epochs=100, lr=0.01, seed=2022, minibatch_size=10):
    # pre-amble
    cost = []
    torch.manual_seed(seed)

    # initialize SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # loop through epochs
    for e in range(num_epochs):
        # shuffle epoch
        shuffle_idx = torch.randperm(y.size(0), dtype=torch.long)
        minibatches = torch.split(shuffle_idx, minibatch_size)

        # loop through minibatch indices
        for minibatch_idx in minibatches:
            # forward pass
            predictions = model.forward(x[minibatch_idx])

            # calculate loss
            loss = F.mse_loss(predictions, y[minibatch_idx])

            # reset gradients
            optimizer.zero_grad()
            # compute gradients
            loss.backward()
            # update weights
            optimizer.step()

        # logging
        with torch.no_grad():
            # prediction = model.forward(x)
            print(f"Epoch: {e + 1:03d}", end="")
            print(f"| MSE: {loss: .5f}")
            cost.append(loss)

    return cost


model = Adaline(num_features=None)







