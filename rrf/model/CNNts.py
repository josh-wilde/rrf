from typing import List, Tuple
from math import floor

import torch
from torch import nn


# Kind of following VGG-16 from https://www.jeremyjordan.me/convnet-architectures/
class CNNts(nn.Module):
    def __init__(self,
                 block_layers: str, # each entry is the number of layers in each conv block, like '2-3-4'
                 seq_len: int,
                 n_features: int,
                 kernel_size: int = 5,
                 initial_conv_channels: int = 16,
                 batch_norm: bool = False):

        super().__init__()
        # Convert hyphen-separated string to list
        block_layers_list = list(map(int, block_layers.split(sep='-')))
        assert (
            seq_len > 2 ** (len(block_layers_list) - 1),
            f'time series of length {seq_len} cannot go through {len(block_layers_list) - 1} pooling layers of kernel_size 2.'
        )
        self.net, self.output_len = self._make_cnn_layers(block_layers_list,
                                                          seq_len,
                                                          n_features,
                                                          initial_conv_channels,
                                                          kernel_size,
                                                          batch_norm)

    @staticmethod
    def _make_cnn_layers(
        block_layers: List[int],
        input_len: int,
        feature_channels: int,
        initial_conv_channels: int,
        cnn_kernel_width: int,
        batch_norm: bool
    ) -> Tuple[nn.Sequential, int]:

        layers: List[nn.Module] = []
        input_channels = feature_channels  # input channels = number of features
        output_channels = initial_conv_channels  # num of channels in first conv block
        output_len = input_len  # length of the ts vector

        for idx, n_conv_layers_in_block in enumerate(block_layers):

            # For the first conv block, there is no max pooling
            # If there is max pooling, then the input dim for the next conv
            # becomes the prior output dimension divided by 2
            if idx > 0:
                layers += [nn.MaxPool1d(kernel_size=2)]  # halves the length of the observation
                output_len = floor((output_len - 2) / 2 + 1)
                input_channels = output_channels
                output_channels = output_channels * 2  # doubles the number of output conv channels
            for _ in range(n_conv_layers_in_block):
                conv1d = nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=cnn_kernel_width,
                    padding='same'
                )
                if batch_norm:
                    layers += [conv1d, nn.BatchNorm1d(output_channels), nn.ReLU()]
                else:
                    layers += [conv1d, nn.ReLU()]

                # If there are multiple conv layers in a block, then
                # After the first layer, the number of input and output channels are the same
                input_channels = output_channels

        # Pool down to half of the length of the last conv output
        # This gives a (batch_size, output_channels, output_len)
        output_len = floor(output_len / 2)
        layers += [nn.AdaptiveAvgPool1d(output_size=output_len)]

        return nn.Sequential(*layers), output_channels * output_len

    def forward(self, X: torch.tensor) -> torch.tensor:
        '''
        Parameters
        ----------
        X: torch.tensor
            X dim: (batch_size, n_ts_features, seq_length)

        Returns
        -------
        output: torch.tensor
            output dim: (batch_size, output_channels * pool_length)
        '''
        cnn_output = self.net(X)
        flattened_output = torch.flatten(cnn_output, start_dim=1)

        return flattened_output
