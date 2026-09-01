import torch
import torch.nn as nn
import math

class PositionEmbeddingSine(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_pos_feats = 128
        self.temperature = 10000
        self.scale = 2 * math.pi

    def forward(self, x):
        _, _, h, w = x.size()

        device = x.device

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )

        y_embed = grid_y.unsqueeze(0).float()
        x_embed = grid_x.unsqueeze(0).float()

        eps = 1e-6

        y_embed = (
            y_embed
            / (y_embed[:, -1:, :] + eps)
            * self.scale
        )

        x_embed = (
            x_embed
            / (x_embed[:, :, -1:] + eps)
            * self.scale
        )

        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=device,
        )

        dim_t = self.temperature ** (
            2 * (dim_t // 2)
            / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (
                pos_x[:, :, :, 0::2].sin(),
                pos_x[:, :, :, 1::2].cos(),
            ),
            dim=4,
        ).flatten(3)

        pos_y = torch.stack(
            (
                pos_y[:, :, :, 0::2].sin(),
                pos_y[:, :, :, 1::2].cos(),
            ),
            dim=4,
        ).flatten(3)

        return torch.cat(
            (pos_y, pos_x),
            dim=3,
        ).permute(0, 3, 1, 2)