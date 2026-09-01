import torch.nn as nn
from deaot.normalization import FrozenBatchNorm2d

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = FrozenBatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = FrozenBatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes,
            planes * 4,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = FrozenBatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = FrozenBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self._make_layer(64, 3, 1)
        self.layer2 = self._make_layer(128, 4, 2)
        self.layer3 = self._make_layer(256, 6, 2)

    def _make_layer(self, planes, blocks, stride):
        downsample = None

        if (
            stride != 1
            or self.inplanes != planes * Bottleneck.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * Bottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                FrozenBatchNorm2d(
                    planes * Bottleneck.expansion
                ),
            )

        layers = [
            Bottleneck(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
            )
        ]

        self.inplanes = planes * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(
                Bottleneck(
                    self.inplanes,
                    planes,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.relu(self.bn1(self.conv1(input)))
        x = self.maxpool(x)

        x = self.layer1(x)
        xs = [x]

        x = self.layer2(x)
        xs.append(x)

        x = self.layer3(x)
        xs.extend([x, x])

        return xs