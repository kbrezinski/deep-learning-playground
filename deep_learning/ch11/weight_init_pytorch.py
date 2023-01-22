
import torch
import torch.nn as nn


class MLPLayer(torch.nn.Module):
    '''
    Parent class which all implementations inherent from
    '''

    def __init__(self, d_in: int, d_out: int, bias=False):
        super().__init__()

        # adding the weight
        self.w = nn.Parameter(torch.Tensor(d_in, d_out))

        # adding the bias
        if bias:
            self.b = nn.Parameter(torch.Tensor(d_out))
        else:
            self.register_parameter('bias', None)

        # init the trainable params
        self.init_params_()

    def init_params_(self):
        nn.init.kaiming_uniform_(self.w, mode='fan_in', nonlinearity='relu')
        if self.b is not None:
            nn.init.zeros_(self.b)


class MLPImp1(MLPLayer):
    """
    First implementation, simple feedforward ANN
    """
    def __init__(self, d_in: int, d_out: int, bias=False):
        super().__init__(d_in, d_out, bias)

    def forward(self, x):
        z = torch.matmul(x, self.w)

        if self.b is not None:
            z += self.b

        return torch.relu(z)


class MLP(torch.nn.Module):
    def __init__(self, features_per_layer, bias=False):
        super().__init__()

        layers = []
        layer = MLPImp1

        for i in range(len(features_per_layer)-1):
            layers.append(
                layer(
                    d_in=features_per_layer[i],
                    d_out=features_per_layer[i+1],
                    bias=True
                )
            )

        self.model = nn.Sequential(*layers)

    def forward(self, data):
        return self.model(data)


def main() -> None:
    model = MLP([2, 5, 1])
    print(model(torch.randn(10, 2)))


if __name__ == "__main__":
    main()
