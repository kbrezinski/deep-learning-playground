
import torch
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Preparing the data
def prepare_data():
    data = np.genfromtxt('data/toydata.txt', delimiter='\t')
    x = data[:, :2].astype(np.float32)
    y = data[:, 2].astype(np.float32)

    np.random.seed(2022)
    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)

    X_test, y_test = x[idx[:25]], y[idx[:25]]
    X_train, y_train = x[idx[25:]], y[idx[25:]]
    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train).view(-1, 1)

    return X_train, y_train, X_test, y_test


# one-hot encode the dataset
def to_onehot(y, num_classes):
    y_onehot = torch.FloatTensor(y.size(0), num_classes)
    y_onehot.zero_()
    y_cpu = y.to(torch.device('cpu'))
    y_onehot.scatter_(1, y_cpu.view(-1, 1).long(), 1).float()
    return y_onehot.to(DEVICE)


def softmax(z):
    return (torch.exp(z.T) / torch.sum(torch.exp(z), dim=1)).T


def cross_entropy(softmax, y_target):
    return -torch.sum(torch.log(softmax) * y_target, dim=1)


class SoftmaxRegression:
    def __init__(self, num_features, num_classes):
        self.num_features = num_features
        self.num_classes = num_classes
        self.weights = torch.zeros(num_classes, num_features,
                                   dtype=torch.float32, device=DEVICE)
        self.bias = torch.zeros(num_classes,
                                dtype=torch.float32, device=DEVICE)

    def forward(self, x):
        # (n, m) X (m, h) = (n, h) + (1, h) = (n, h)
        logits = torch.mm(x, self.weights.T) + self.bias
        probas = softmax(logits)
        return logits, probas

    def backward(self, x, y, probas):
        # (m, n) X (n, h) = (m, h).T = (h, m) = (2, 2)
        nabla_w = -torch.mm(x.t(), y - probas).T
        # (n, h) - (n, h) = (n, h)
        nabla_b = -torch.sum(y - probas)
        return nabla_w, nabla_b



X_train, y_train, X_test, y_test = prepare_data()
model = SoftmaxRegression(X_train.shape[1], 2)
logits, probas = model.forward(X_train)
print(model.backward(X_train, y_train, probas))


