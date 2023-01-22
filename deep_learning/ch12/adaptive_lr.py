
import torch
import matplotlib.pyplot as plt
import numpy as np

plot = True
Tensor = torch.Tensor


class RMSProp:
    def __init__(self, params: Tensor, lr: float, beta: int = 0.9, eps: float = 1e-8):

        self.params = params
        self.beta = beta
        self.lr = lr
        self.rms = 0
        self.eps = eps

    def step(self):

        curr_rms = (1 - self.beta) * self.params.grad.pow(2)
        self.rms = (self.beta * self.rms) + curr_rms
        self.params -= self.lr * self.params.grad / (torch.sqrt(self.rms) + self.eps)
        self.zero_grad()

    def zero_grad(self):
        self.params.grad = None


class Adam:
    def __init__(self, params: Tensor,
                 lr: float,
                 beta1: float = 0.9,
                 beta2: float = 0.999):

        # user learning inputs
        self.params = params
        self.beta = (beta1, beta2)
        self.lr = lr

        # stateful Adam properties
        self.rms = 0
        self.mom = 0
        self.t = 1
        self.eps = 1e-8

    def step(self):
        curr_mom = (self.beta[0] * self.mom) + ((1 - self.beta[0]) * self.params.grad)
        curr_rms = (self.beta[1] * self.rms) + ((1 - self.beta[1]) * self.params.grad.pow(2))

        self.rms = curr_rms
        self.mom = curr_mom

        num = self.lr * (self.mom / (1 - self.beta[0] ** self.t))
        denom = torch.sqrt(self.rms / (1 - self.beta[1] ** self.t)) + self.eps

        self.params -= num / denom

        self.zero_grad()
        self.t += 1

    def zero_grad(self):
        self.params.grad = None


def main() -> None:

    X = (torch.arange(0., 10.) + (.1 * torch.randn(1, 10))).unsqueeze(-1)
    y = torch.arange(0., 10.)

    w = torch.nn.Parameter(torch.ones(1), requires_grad=True)
    # optimizer = RMSProp(params=w, lr=1e-3)
    optimizer = Adam(params=w, lr=1e-3)

    # begin training
    for i in range(10):

        # forward pass
        loss = (torch.matmul(X, optimizer.params) - y).pow(2).sum()
        loss.backward()

        with torch.no_grad():
            print(f"mean_square: {optimizer.rms} | mean_square: {optimizer.mom} | grad: {optimizer.params.grad.item():.4f}")
            optimizer.step()


        print(f"loss: {loss.item():.6f}")

    if plot:
        plt.scatter(X, y)
        plt.plot(np.arange(0., 10.), np.arange(0., 10.) * w.detach().numpy(), '-r')
        plt.show()


if __name__ == "__main__":
    main()
