
import torch
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR


# manual implementation of lr decay
def adjust_lr(opt, epoch, init_lr, decay):
    if not epoch % 10:
        lr = init_lr * np.exp(-decay * epoch)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        print(lr)


def get_scheduler_fn():
    lambda1 = lambda epoch: epoch // 10
    lambda2 = lambda epoch: .95 ** epoch
    return [lambda1]


def main() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=True))
    optimizer = Adam(model.parameters(), lr=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=get_scheduler_fn())

    for i in range(100):
        #adjust_lr(optimizer, epoch=i, init_lr=0.01, decay=1e-1)
        scheduler.step()
        print([param['lr'] for param in optimizer.param_groups])


if __name__ == "__main__":
    main()
