import torch.nn as nn
import torch.nn.functional as F

from deaot.ops import ConvGN

class FPNSegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_in = ConvGN(512, 256, 1)

        self.conv_16x = ConvGN(256, 256, 3)
        self.conv_8x = ConvGN(256, 128, 3)
        self.conv_4x = ConvGN(128, 128, 3)

        self.adapter_16x = nn.Conv2d(1024, 256, 1)
        self.adapter_8x = nn.Conv2d(512, 256, 1)
        self.adapter_4x = nn.Conv2d(256, 128, 1)

        self.conv_out = nn.Conv2d(128, 11, 1)

    def forward(self, inputs, shortcuts):
        x = inputs[-1]

        x = F.relu_(self.conv_in(x))

        x = F.relu_(
            self.conv_16x(
                self.adapter_16x(shortcuts[-2]) + x
            )
        )

        x = F.interpolate(
            x,
            size=shortcuts[-3].size()[-2:],
            mode="bilinear",
            align_corners=True,
        )

        x = F.relu_(
            self.conv_8x(
                self.adapter_8x(shortcuts[-3]) + x
            )
        )

        x = F.interpolate(
            x,
            size=shortcuts[-4].size()[-2:],
            mode="bilinear",
            align_corners=True,
        )

        x = F.relu_(
            self.conv_4x(
                self.adapter_4x(shortcuts[-4]) + x
            )
        )

        return self.conv_out(x)