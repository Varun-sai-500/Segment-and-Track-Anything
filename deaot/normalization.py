import torch
import torch.nn as nn

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n, epsilon=1e-5):
        super().__init__()

        self.register_buffer(
            "weight",
            torch.ones(n),
        )
        self.register_buffer(
            "bias",
            torch.zeros(n),
        )
        self.register_buffer(
            "running_mean",
            torch.zeros(n),
        )
        self.register_buffer(
            "running_var",
            torch.ones(n) - epsilon,
        )

        self.epsilon = epsilon

    def forward(self, x):
        scale = self.weight * (
            self.running_var + self.epsilon
        ).rsqrt()

        bias = self.bias - self.running_mean * scale

        return (
            x
            * scale.reshape(1, -1, 1, 1).to(x.dtype)
            + bias.reshape(1, -1, 1, 1).to(x.dtype)
        )