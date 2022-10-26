import pandas as pd
import os
from typing import Dict, Any


def write_data_thaw_metadata(metadata: Dict[str, Any], output_dir: str):
    '''
    Write the metadata returned from data_thaw into some folder in some format
    Parameters
    ----------
    metadata: dictionary
        Metadata from data thaw

    Returns
    -------
    Nothing
    '''
    # Commenting these out because they don't exist unless I do the time series features
    necessary_keys = ['feature_columns', 'n_observations', 'run_ids'] # 'ts_resampled_index', 'ts_index', 'run_lengths'
    for k in necessary_keys:
        assert k in metadata, f"{k} is not in metadata"

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Feature columns is a Dict[str, List[str]], print each item as separate series
    feature_column_dict = metadata['feature_columns']
    for feature_set in feature_column_dict:
        pd.Series(feature_column_dict[feature_set]).to_csv(f"{output_dir}/{feature_set}_columns.csv", index=False)

    # n_observations is a Dict[str, int]
    pd.Series(metadata['n_observations']).to_csv(f"{output_dir}/n_observations.csv")

    # run_ids is a sorted list
    pd.Series(metadata['run_ids'], name='run_id').to_csv(f"{output_dir}/run_ids.csv")

    # ts_resampled_index is a dataframe
    #metadata['ts_resampled_index'].to_csv(f"{output_dir}/ts_resampled_index.csv")

    # ts_index is a dataframe
    #metadata['ts_index'].to_csv(f"{output_dir}/ts_index.csv")

    # run_lengths is a dataframe
    #metadata['run_lengths'].to_csv(f"{output_dir}/run_lengths.csv")

    # X_ts_padded.index.to_frame().to_parquet('index_test.parquet', index=False)
