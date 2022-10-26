from typing import List, Dict, Optional
from math import floor

import torch
from torch import nn
import torch.nn.functional as F

from .CNNts import CNNts
from .LSTMts import LSTMts
from .ResNet18ts import ResNet18ts
from .utils import validate_cnn_necessary_params, validate_lstm_necessary_params, validate_resnet18_necessary_params


# This is flexible enough to take any combination of ts, threshold, and static features
class MBINet(nn.Module):
    def __init__(self,
                 input_feature_sets: List[str],
                 input_n_features: Dict[str, int],
                 ts_module_name: Optional[str] = None,
                 top_dropout_frac: float = 0.5,
                 n_classes: int = 3,
                 **kwargs):

        super().__init__()
        # Initialize the ts module, if any
        self.ts_module_name = ts_module_name
        if 'ts' in input_feature_sets:
            if ts_module_name == 'cnn':
                validate_cnn_necessary_params(kwargs.keys())
                self.ts_module = CNNts(
                    block_layers=kwargs['cnn_block_layers'],
                    seq_len=kwargs['cnn_seq_len'],
                    n_features=input_n_features['ts'],
                    kernel_size=kwargs['cnn_kernel_size'],
                    initial_conv_channels=kwargs['cnn_initial_conv_channels'],
                    batch_norm=kwargs['cnn_batch_norm']
                )
                self.ts_input_dim = self.ts_module.output_len

            elif ts_module_name == 'resnet18':
                validate_resnet18_necessary_params(kwargs.keys())
                self.ts_module = ResNet18ts(seq_len=kwargs['resnet18_seq_len'], n_features=input_n_features['ts'])
                self.ts_input_dim = self.ts_module.output_len

            elif ts_module_name == 'lstm':
                validate_lstm_necessary_params(kwargs.keys())
                self.ts_module = LSTMts(
                    n_features=input_n_features['ts'],
                    hidden_dim=kwargs['lstm_hidden_dim'],
                    n_layers=kwargs['lstm_n_layers'],
                    dropout_frac=kwargs['lstm_dropout_frac']
                )
                self.ts_input_dim = self.ts_module.output_len

            else:
                raise ValueError(f'Only lstm, cnn, and resnet18 are implemented for ts modules, not {ts_module_name}')
        else:
            self.ts_input_dim = 0

        # Initialize the top module
        self.input_feature_sets = input_feature_sets
        self.static_input_dim = 0
        if 'static' in input_feature_sets:
            self.static_input_dim += input_n_features['static']
        if 'threshold' in input_feature_sets:
            self.static_input_dim += input_n_features['threshold']

        # Determine linear hidden layer size
        total_input_dim = self.ts_input_dim + self.static_input_dim
        linear_hidden_dim = min(floor(total_input_dim / 2), 64)

        self.top_net = nn.Sequential(
            nn.Linear(total_input_dim, linear_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=top_dropout_frac),
            nn.Linear(linear_hidden_dim, linear_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=top_dropout_frac),
            nn.Linear(linear_hidden_dim, n_classes)
        )

    def forward(
        self,
        X_ts_resampled: Optional[torch.Tensor] = None,
        X_ts_padded: Optional[torch.Tensor] = None,
        X_static: Optional[torch.Tensor] = None,
        X_threshold: Optional[torch.Tensor] = None,
        run_lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:

        top_model_input_tensors = []

        # Create the static features
        if 'static' in self.input_feature_sets:
            if X_static is None:
                raise ValueError('Must provide X_static!')
            else:
                top_model_input_tensors += [X_static]
        if 'threshold' in self.input_feature_sets:
            if X_threshold is None:
                raise ValueError('Must provide X_threshold!')
            else:
                top_model_input_tensors += [X_threshold]

        # Run the time series features through the time series module
        if 'ts' in self.input_feature_sets:
            if self.ts_module_name in ['cnn', 'resnet18']:
                if X_ts_resampled is None:
                    raise ValueError('Must provide X_ts_resampled for cnn or resnet18!')
                else:
                    ts_output = self.ts_module(X_ts_resampled)

            if self.ts_module_name == 'lstm':
                if X_ts_padded is None:
                    raise ValueError('Must provide X_ts_padded for lstm!')
                else:
                    if run_lengths is None:
                        raise ValueError('Must provide run_lengths for lstm!')
                    else:
                        ts_output = self.ts_module(X_ts_padded, run_lengths)

            top_model_input_tensors += [ts_output]

        # Concatenate the input to the top model
        top_model_input = torch.cat(top_model_input_tensors, dim=1)

        # Run the top model on top of the combined features
        # Gives (batch_size, n_classes), but this does not do softmax
        top_model_output = self.top_net(top_model_input)

        return F.softmax(top_model_output, dim=1)
