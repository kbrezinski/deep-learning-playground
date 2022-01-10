
import torch
import pandas as pd


def import_data(seed=2021):

    df = pd.read_csv("./linreg-data.csv", index_col=0)
    X = torch.tensor(df[['x1', 'x2']].values, dtype=torch.float)
    y = torch.tensor(df['y'].values, dtype=torch.float)

    # shuffling
    torch.manual_seed(seed)
    shuffle_idx = torch.randperm(y.size(0), dtype=torch.long)
    X, y = X[shuffle_idx], y[shuffle_idx]

    # 70% train/test split index
    test_idx = int(shuffle_idx.size(0) * 0.7)

    # train/test set
    X_train, X_test = X[shuffle_idx[:test_idx]], X[shuffle_idx[test_idx:]]
    y_train, y_test = y[shuffle_idx[:test_idx]], y[shuffle_idx[test_idx:]]

    # normalize
    mu, sigma = X_train.mean(dim=0), X_train.std(dim=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    return X_train, X_test, y_train, y_test


class LinReg:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1,
                                   dtype=torch.float)
        self.bias = torch.zeros(1, dtype=torch.float)

    def forward(self, x):
        # (x ,weights) = ((10, w) (w, 1)) = (10, 1)
        z = torch.add(torch.mm(x, self.weights), self.bias)
        activations = z
        # (10,) -> squeeze final dimension
        return activations.view(-1)

    def backward(self, x, pred, true):

        # gradient of the loss
        grad_loss = 2 * (pred - true)
        # gradient w.r.t weights
        grad_w = x
        # gradient w.r.t. bias
        grad_b = 1.

        # (1, 10) * (10 ,1)
        nabla_w = torch.mm(grad_w.t(),
                           grad_loss.view(-1, 1)) / true.size(0)
        # (1,)
        nabla_b = torch.sum(grad_b * grad_loss) / true.size(0)

        return -1 * nabla_w, -1 * nabla_b


# mean squared error loss
def loss(true, pred):
    return torch.mean((pred - true) ** 2)


# training script
def train(model, x, y, num_epochs, lr=0.01):
    cost = []
    for e in range(num_epochs):

        pred = model.forward(x)
        grad_w, grad_b = model.backward(x, pred, y)
        model.weights += lr * grad_w
        model.bias += lr * grad_b

        pred = model.forward(x)
        curr_loss = loss(pred, y)

        print(f"Epoch: {e + 1:03d}", end="")
        print(f"| MSE: {curr_loss:.5f}")
        cost.append(curr_loss)

    return cost


X_train, X_test, y_train, y_test = import_data(seed=2021)
model = LinReg(num_features=X_train.size(1))
cost = train(model, X_train, y_train, num_epochs=100, lr=0.05)

