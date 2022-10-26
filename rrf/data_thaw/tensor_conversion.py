import pandas as pd
from typing import Dict, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def convert_pandas_to_tensors(
    labels: pd.Series,
    feature_data: Dict[str, pd.DataFrame]
) -> Tuple[torch.tensor, Dict[str, torch.tensor]]:
    """
    Parameters
    ----------
    labels: pd.Series
        labels for the run_ids
    feature_data: Dict[str, pd.DataFrame]
        dictionary of dataframes: ts_features, ts_features_resampled, static_features, threshold_features, run_lengths

    Returns
    -------
    labels: torch.tensor
        same dimension
    feature_data: Dict[str, torch.tensor]
        Dict of tensors: X_ts_resampled, X_ts_padded, X_static, X_threshold, run_lengths
    """

    # Labels are simple conversion
    labels_tensor = torch.tensor(labels.values, dtype=torch.int64)

    # So are the static features
    feature_tensors = {}
    if 'threshold_features' in feature_data:
        feature_tensors['X_threshold'] = torch.tensor(feature_data['threshold_features'].values,
                                                      dtype=torch.float32)
    if 'static_features' in feature_data:
        feature_tensors['X_static'] = torch.tensor(feature_data['static_features'].values,
                                                   dtype=torch.float32)

    # Resampled ts features
    # (run_ids * pool_len, n_features) -> (run_ids, n_features, seq_len)
    if 'ts_features_resampled' in feature_data:
        pd_ts_resampled = feature_data['ts_features_resampled']
        n_ts_rsmpl_run_ids = len(pd_ts_resampled.index.get_level_values('run_id').unique())
        n_ts_rsmpl_features = len(pd_ts_resampled.columns)
        X_ts_resampled = torch.tensor(pd_ts_resampled.values, dtype=torch.float32)
        # This gets to (run_ids, seq_len, n_features)
        # Has to be this way for reshape to work correctly!
        X_ts_resampled = torch.reshape(X_ts_resampled,
                                       (n_ts_rsmpl_run_ids, -1, n_ts_rsmpl_features))
        feature_tensors['X_ts_resampled'] = X_ts_resampled.permute(0, 2, 1)

    # Not resampled ts features
    # Convert (run_ids * pool_len, n_features) -> (run_ids, max_seq_len, n_ts_features)
    if 'ts_features' in feature_data:
        pd_ts = feature_data['ts_features']
        run_lengths = (pd_ts
                       .index
                       .get_level_values('run_id')
                       .value_counts(sort=False)
                       .sort_index()
                       .to_list()
                       )
        # Get a list of tensors, each item is a run_id
        # Each item: (seq_len, n_ts_features)
        run_tensor_list = torch.split(torch.tensor(pd_ts.values, dtype=torch.float32),
                                      run_lengths)

        # Pad each of the runs to the length of the max run
        # X_ts_padded: (n_runs, max_seq_len, n_ts_features)
        X_ts_padded = pad_sequence(run_tensor_list, batch_first=True)

        feature_tensors['X_ts_padded'] = X_ts_padded
        feature_tensors['run_lengths'] = torch.tensor(run_lengths, dtype=torch.int64)

    return labels_tensor, feature_tensors
