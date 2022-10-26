import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


# Not clearly worse to use dropout in the LSTM, but also not clearly better
# Will need to CV
class LSTMts(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_layers: int,
        retain_all_layers: bool = True,
        dropout_frac: float = 0.0
    ):
        super().__init__()
        self.net = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout_frac,
            batch_first=True
        )
        output_channels = n_layers if retain_all_layers else 1
        self.output_len = hidden_dim * output_channels
        self.retain_all_layers = retain_all_layers

    def forward(self, X: torch.tensor, run_lengths: torch.Tensor) -> torch.Tensor:
        '''
        Parameters
        ----------
        X: torch.Tensor
            X dim: (batch_size, max_seq_len, n_ts_features)
            This tensor is padded with 0s
        run_lengths: torch.tensor
            one-dim tensor with run lengths to keep track of where real seqs end

        Returns
        -------
        output: torch.tensor
            output dim: (batch_size, num_layers, hidden_dim)
        '''
        # Pack the padded sequences so that the NN knows to ignore the padded values
        X_packed = pack_padded_sequence(X, run_lengths, batch_first=True, enforce_sorted=False)

        # h_n dim: (n_layers, batch_size, hidden_dim)
        _, (h_n, _) = self.net(X_packed)

        # Permute to (batch_size, hidden_dim, n_layers)
        h_n_permuted = h_n.permute(1,2,0)

        if self.retain_all_layers:
            return torch.flatten(h_n_permuted, start_dim=1)
        else:
            return torch.flatten(h_n_permuted[:, :, -1:], start_dim=1)
