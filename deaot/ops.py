import torch.nn as nn

class GroupNorm1D(nn.Module):
    def __init__(self, indim, groups=8):
        super().__init__()

        self.gn = nn.GroupNorm(
            groups,
            indim,
        )

    def forward(self, x):
        return self.gn(
            x.permute(1, 2, 0)
        ).permute(2, 0, 1)


class DWConv2d(nn.Module):
    def __init__(self, indim):
        super().__init__()

        self.conv = nn.Conv2d(
            indim,
            indim,
            kernel_size=5,
            padding=2,
            groups=indim,
            bias=False,
        )

    def forward(self, x, size_2d):
        h, w = size_2d
        _, bs, c = x.size()

        x = (
            x.view(h, w, bs, c)
            .permute(2, 3, 0, 1)
        )

        x = self.conv(x)

        return (
            x.view(bs, c, h * w)
            .permute(2, 0, 1)
        )


class ConvGN(nn.Module):
    def __init__(
        self,
        indim,
        outdim,
        kernel_size,
        gn_groups=8,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            indim,
            outdim,
            kernel_size,
            padding=kernel_size // 2,
        )

        self.gn = nn.GroupNorm(
            gn_groups,
            outdim,
        )

    def forward(self, x):
        return self.gn(
            self.conv(x)
        )


def seq_to_2d(tensor, size_2d):
    h, w = size_2d
    _, n, c = tensor.size()

    return (
        tensor.view(h, w, n, c)
        .permute(2, 3, 0, 1)
        .contiguous()
    )