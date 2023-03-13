from typing import Optional

import torch
import torch.nn as nn


def auto_pad(kernel_size: int, dilation: int) -> int:
    """Padding mode = `same`"""
    padding = (kernel_size - 1) // 2 * dilation
    return padding


class Conv(nn.Module):
    """Convolutional Block"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            dilation: int = 1,
            bias: bool = False,
            act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=auto_pad(kernel_size, dilation) if padding is None else padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DoubleConv(nn.Module):
    """Double Convolutional Block"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: Optional[int] = None,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = False,
            act: bool = True,
    ) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = Conv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            act=act,
        )
        self.conv2 = Conv(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            act=act,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class Down(nn.Module):
    """Feature Downscale"""

    def __init__(self, in_channels: int, out_channels: int, scale_factor=2) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=scale_factor)
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)

        return x


class Up(nn.Module):
    """Feature Upscale"""

    def __init__(self, in_channels: int, out_channels: int, scale_factor: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=scale_factor
        )
        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x_ = torch.cat([x2, x1], dim=1)
        return self.conv(x_)


class UNet(nn.Module):
    """UNet Segmentation Model"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input_conv = DoubleConv(in_channels, out_channels=64)

        # Downscale ⬇️
        self.down1 = Down(in_channels=64, out_channels=128, scale_factor=2)  # P/2
        self.down2 = Down(in_channels=128, out_channels=256, scale_factor=2)  # P/4
        self.down3 = Down(in_channels=256, out_channels=512, scale_factor=2)  # P/8
        self.down4 = Down(in_channels=512, out_channels=1024, scale_factor=2)  # P/16

        # Upscale ⬆️
        self.up1 = Up(in_channels=1024, out_channels=512, scale_factor=2)
        self.up2 = Up(in_channels=512, out_channels=256, scale_factor=2)
        self.up3 = Up(in_channels=256, out_channels=128, scale_factor=2)
        self.up4 = Up(in_channels=128, out_channels=64, scale_factor=2)

        self.output_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.input_conv(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x_ = self.up1(x4, x3)
        x_ = self.up2(x_, x2)
        x_ = self.up3(x_, x1)
        x_ = self.up4(x_, x0)

        x_ = self.output_conv(x_)

        return x_
