
import numpy as np
import matplotlib.pyplot as plt

from Perceptron import Perceptron


def load_data():
    data = np.genfromtxt('toydata.txt', delimiter='\t')
    X, y = data[:, :2], data[:, 2]
    print(X.shape, y.shape)
    return X, y.astype(int)


def randomize_samples(X: np.array, y: np.array):
    # shuffling
    shuffle_idx = np.arange(y.shape[0])
    shuffle_rng = np.random.RandomState(123)
    shuffle_rng.shuffle(shuffle_idx)

    return X[shuffle_idx], y[shuffle_idx]


def normalize(train_features, test_features):
    mu, sigma = train_features.mean(axis=0), train_features.std(axis=0)
    return (train_features - mu) / sigma, (test_features - mu) / sigma


def main():
    # load the data
    X, y = load_data()

    # shuffling
    X, y = randomize_samples(X, y)

    # train test split:
    X_train, X_test = X[:70], X[70:]
    y_train, y_test = y[:70], y[70:]

    # Normalize
    X_train, X_test = normalize(X_train, X_test)

    # Initialize and train model
    ppn = Perceptron(num_features=2)
    ppn.train(X_train, y_train, epochs=5)
    print(f"Weights: {ppn.weights} \n Biases: {ppn.bias}")

    print(f"Test set accuracy: {ppn.evaluate(X_test, y_test) * 100:.2f}%")


if __name__ == '__main__':
    main()
