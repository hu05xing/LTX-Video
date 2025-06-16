from typing import Tuple, Union
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: Union[int, Tuple[int]] = 1,
        dilation: int = 1,
        groups: int = 1,
        spatial_padding_mode: str = "zeros",
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 关键参数处理
        kernel_size = (kernel_size, kernel_size, kernel_size)            # 扩展为(t, h, w)
        self.time_kernel_size = kernel_size[0]                           # 记录时间维度核大小

        dilation = (dilation, 1, 1)                                      # 仅时间维度使用dilation

        # 空间维度计算padding（时间维度padding=0）
        height_pad = kernel_size[1] // 2
        width_pad = kernel_size[2] // 2
        padding = (0, height_pad, width_pad)


        # 创建基础Conv3d层（时间维度无自动padding）
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            padding_mode=spatial_padding_mode,
            groups=groups,
        )


    def forward(self, x, causal: bool = True):
        if causal:
            # 因果模式：前端填充(time_kernel_size-1)帧（复制第一帧）
            first_frame_pad = x[:, :, :1, :, :].repeat(
                (1, 1, self.time_kernel_size - 1, 1, 1)
            )
            x = torch.concatenate((first_frame_pad, x), dim=2)
        else:
            # 非因果模式：前后对称填充（复制首尾帧）
            first_frame_pad = x[:, :, :1, :, :].repeat(
                (1, 1, (self.time_kernel_size - 1) // 2, 1, 1)
            )
            last_frame_pad = x[:, :, -1:, :, :].repeat(
                (1, 1, (self.time_kernel_size - 1) // 2, 1, 1)
            )
            x = torch.concatenate((first_frame_pad, x, last_frame_pad), dim=2)
        x = self.conv(x)
        return x

    @property
    def weight(self):
        return self.conv.weight
        logger.debug(f"✅时空卷积模块执行卷积")
