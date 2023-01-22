
import torch
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:

    X = (torch.arange(0., 10.) + (1 * torch.randn(1, 10))).unsqueeze(-1)
    y = torch.arange(0., 10.)

    w = torch.nn.Parameter(torch.ones(1))
    w_last = 0.0
    alpha = 0.9
    lr = 1e-4

    for i in range(10):
        loss = (torch.matmul(X, w) - y).pow(2).sum()
        loss.backward()

        with torch.no_grad():
            print(f"velocity: {(w_last * alpha)} | grad: {w.grad.item()}")

            # momentum implementation
            velocity = (w_last * alpha) + (lr * w.grad)
            w -= velocity

            # store velocity as prev
            w_last = velocity
            # zero out gradient
            w.grad = None

        print(f"loss: {loss.item():.6f}")

    if True:
        plt.scatter(X, y)
        plt.plot(np.arange(0., 10.), np.arange(0., 10.) * w.detach().numpy(), '-r')
        plt.show()


if __name__ == "__main__":
    main()
