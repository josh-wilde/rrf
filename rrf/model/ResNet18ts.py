from typing import List
from math import floor

import torch
from torch import nn
import torch.nn.functional as F

# Implementing ResNet-18 for time series
# Adapted from https://d2l.ai/chapter_convolutional-modern/resnet.html


class Residual(nn.Module):
    # This is the residual block

    def __init__(
        self,
        input_channels: int,
        num_channels: int,
        use_1conv: bool = False,
        strides=1
    ):

        super().__init__()

        # The residual block consists of 2 conv layers with kernel size 3
        # The first block changes the number of channels, and the second does not
        self.conv1 = nn.Conv1d(
            input_channels,
            num_channels,
            kernel_size=3,
            padding=1,
            stride=strides
        )
        self.conv2 = nn.Conv1d(
            num_channels,
            num_channels,
            kernel_size=3,
            padding=1
        )

        # If input_channels != num_channels, then need to transform the number of channels
        # In the input in order to add it to the conv output
        if use_1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        # After each of the two conv layers, add a batch normalization step
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        '''
        Parameters
        ----------
        X: torch.tensor
            X dim: (batch_size, input_channels (n_ts_features if X is raw input), * (seq_len if X is the raw input))

        Returns
        -------
        output: torch.tensor
            output dim: (batch_size, num_channels, *)
        '''
        # Y dim: (batch_size, num_channels, *)

        # First convolution
        Y = F.relu(self.bn1(self.conv1(X)))
        # Second convolution
        Y = self.bn2(self.conv2(Y))

        # Change number of X channels if necessary
        if self.conv3:
            X = self.conv3(X)

        Y += X

        return F.relu(Y)


def resnet_block(input_channels: int,
                 num_channels: int,
                 num_residuals: int,
                 first_block: bool = False) -> List[nn.Module]:
    '''
    After the first CNN block that processes the input,
    there are 4 resnet blocks. The first of these resnet blocks does not change the number of channels.
    However, all future resnet blocks double the number of channels, cutting the input length in half.

    Parameters
    ----------
    input_channels: number of channels of input
    num_channels: output number of channels
    num_residuals: controls number of residual modules in the block
    first_block: controls whether or not to double the number of channels
    '''

    blk = []

    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))

    return blk


class ResNet18ts(nn.Module):

    def __init__(self,
                 seq_len: int,
                 n_features: int):
        super().__init__()

        self.seq_len = seq_len
        self.n_features = n_features

        # Set up the initial convolution block
        # The output length should be divisible by 8 for the 4 block structure to work out
        b1_input_len = floor((seq_len - 1) / 2 + 1) # relies on kernel=7, pad=3, stride=2 for first conv
        b1_pool_pad = 1
        b1_stride = 2
        b1_output_len = 48
        b1_kernel_size = self._get_b1_kernel_size(
            input_len=b1_input_len,
            padding=b1_pool_pad,
            stride=b1_stride,
            output_len=b1_output_len
        )

        b1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=b1_kernel_size, stride=b1_stride, padding=b1_pool_pad)
        )

        # Set up the residual blocks
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))

        # Combine all blocks
        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                                 nn.AdaptiveAvgPool1d(1),
                                 nn.Flatten())

        # Set the output length for compatibility with MBINet
        self.output_len = 512

    @staticmethod
    def _get_b1_kernel_size(
        input_len: int,
        padding: int,
        stride: int,
        output_len: int
    ) -> int:
        # Back out the kernel size from the seq_len, padding, stride, and output_len
        proposed_kernel_size = input_len + 2 * padding - stride * (output_len - 1)

        assert proposed_kernel_size > 0, f"Invalid kernel size for ResNet18ts b1 ({proposed_kernel_size}), must be positive"
        assert proposed_kernel_size <= 15, f"Invalid kernel size for ResNet18ts b1 ({proposed_kernel_size}), should be <= 15"

        return proposed_kernel_size

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)
