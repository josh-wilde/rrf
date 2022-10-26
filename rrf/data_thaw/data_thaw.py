import pandas as pd
import numpy as np
from typing import Optional

from .write_data_thaw_metadata import write_data_thaw_metadata


def filter_df_by_run_length(
    df: pd.DataFrame,
    min_input_run_length: Optional[int],
    max_input_run_length: Optional[int]
):
    if min_input_run_length is None and max_input_run_length is None:
        return df
    else:
        assert df.index.names[0] == 'run_id', 'run_id not the first index level'
        assert 'run_length' in df.columns, 'run_length column does not exist'

        run_lengths = df.groupby(level=0)['run_length'].first()
        valid_runs = run_lengths.copy()

        if min_input_run_length is not None:
            valid_runs = valid_runs[valid_runs >= min_input_run_length]
        if max_input_run_length is not None:
            valid_runs = valid_runs[valid_runs <= max_input_run_length]
        valid_runs = valid_runs.index.to_list()
        df_filtered = df.loc[valid_runs, :]

        return df_filtered


def get_standard_run_length(
        run_lengths: pd.Series,
        run_length_qtile: float,
        sample_freq: int
):

    target_run_length = run_lengths.quantile(run_length_qtile)
    if target_run_length % sample_freq == 0:
        actual_run_length = target_run_length
    else:
        actual_run_length = target_run_length + (sample_freq - target_run_length % sample_freq)
    print(f"standard run length set to {actual_run_length} minutes, or {actual_run_length / 60} hours.")
    print(f"This means that there will be {actual_run_length / sample_freq} periods per run.")
    return int(actual_run_length)


def resample_df_to_standard_run_length(
    grp: pd.DataFrame,
    keep_start: bool,
    sample_freq: int,
    standard_run_length: int
):
    assert grp.index.names == ['run_id', 'time_date']
    assert standard_run_length % sample_freq == 0, 'standard run length is not muliple of sample frequency'

    # Run id for this group
    run_id = grp.index.get_level_values(0)[0]

    # Standardized index
    if keep_start:
        group_index = pd.date_range(start=grp.index.get_level_values(level=1).min(),
                                    periods=standard_run_length / sample_freq,
                                    freq=f'{sample_freq}min')
        grp_time_only_reindexed = (grp
                                   .droplevel(level=0, axis=0)
                                   .reindex(group_index, method='pad'))
    else:
        group_index = pd.date_range(end=grp.index.get_level_values(level=1).max(),
                                    periods=standard_run_length / sample_freq,
                                    freq=f'{sample_freq}min')
        grp_time_only_reindexed = (grp
                                   .droplevel(level=0, axis=0)
                                   .reindex(group_index, method='backfill'))

    # Reset index to time_date
    # Does not need to be reindexed to (run_id, time_date) since the run_id will be added in the apply
    grp_time_only_reindexed.index = grp_time_only_reindexed.index.rename('time_date')
    output = grp_time_only_reindexed

    # Check to make sure everyone has the same run length and that it is standard run length
    assert output.shape[0] == standard_run_length / sample_freq, \
        f"{run_id} has run length of {output.shape[0]} not standard of {standard_run_length / sample_freq}"

    return output


def thaw_data_to_pandas(
    feature_specification: dict,
    label_specification: dict,
    keep_start: bool,
    standard_run_length_quantile: float = 0.5,
    resample_freq_minutes: int = 6,
    min_input_run_length: Optional[int] = None,
    max_input_run_length: Optional[int] = None,
    metadata_dir: Optional[str] = None
):
    '''
    Creates the dataset that goes into the classification testing.
    Includes division into training and test sets.
    Also outputs summary statistics on the dataset size.

    Parameters
    ----------
    feature_specification: dictionary
        Keys are the feature sets: threshold, static, ts. Values are dictionaries, where:
            Keys are 'threshold_features' and one or both of 'static_features' and 'ts_features'
            For each feature set, we get a file path and a list of features to include

    label_specification: dictionary
        Keys are 'fpath' and 'feature', values are strings for file path and label column

    keep_start: bool
        If True, then samples longer than the standard run length will be truncated at the end
        If False, then samples will be truncated at the beginning.

    standard_run_length_quantile: float
        Should be in [0,1]. Defines the standard run length in terms of the run lengths in the data.
        If 0.5, then the median run length becomes the standard run length.

    resample_freq_minutes: int
        Number of minutes between samples to standardize to.
        Default to 6, which is approximately the sample frequency when a burn in test is run at capacity.

    min_input_run_length: int
        Optionally, specify a minimum run length such that all observations below this run length are removed

    max_input_run_length: int
        Same idea as min_input_run_length.

    write_data_thaw_metadata: bool
        Whether or not to write out the metadata to files.

    Returns
    -------
    - dict of features (no train/test because nested CV already gives back OOS performance estimates), with resampling for ts and without
    - labels
    - list of included runs
    - dictionary of metadata

    '''

    # dictionary to save metadata about number of observations
    n_obs = {
        'label_initial': np.nan,
        'threshold_features_initial': np.nan,
        'ts_features_initial': np.nan,
        'static_features_initial': np.nan,
        'ts_features_len_filtered': np.nan,
        'final_runs': np.nan
    }

    # Pull the labels and put into a series
    print('---Pulling labels')
    labels = pd.read_csv(label_specification['fpath']).set_index('run_id')
    labels = labels[label_specification['feature']]
    labels = labels[labels > -1]
    assert len(labels.dropna().index) == len(labels.index), 'Labels contain NA values'
    label_runs = labels.index
    n_obs['label_initial'] = len(labels)

    # Pull the features
    feature_data = {}
    feature_valid_runs = {}
    run_lengths = pd.Series()

    for feature_set in feature_specification:
        print(f'---Pulling {feature_set}')
        feature_fpath = feature_specification[feature_set]['fpath']
        feature_list = feature_specification[feature_set]['features']
        df = pd.read_parquet(feature_fpath).reset_index()   # if feature_set != 'ts_features' else pd.read_parquet(feature_fpath).reset_index().head(1000)
        assert len(df.dropna().index) == len(df.index), f'{feature_set} contains NA values'

        # Restrict to correct columns and set index
        id_cols = ['run_id'] if feature_set in ['threshold_features', 'static_features'] else ['run_id', 'time_date']
        run_len_cols = []  if feature_set in ['threshold_features', 'static_features'] else ['run_length']
        df = df[id_cols + run_len_cols + feature_list].set_index(id_cols)

        if feature_set == 'ts_features':
            print('---Dealing with ts-specific filtering')
            # Important: the first feature in the input ts_features dataset will be run lengths
            # But these run lengths won't appear in the ts_features_resampled dataset
            # Since all runs are the same length

            # Filter run lengths
            df_length_filtered = filter_df_by_run_length(df, min_input_run_length, max_input_run_length)

            # Save the run lengths and delete the feature
            run_lengths = (df_length_filtered
                           .groupby(level=0)['run_length']
                           .first())
            df_length_filtered = df_length_filtered.drop('run_length', axis=1)

            # Save the run ids and ts_features
            feature_valid_runs[feature_set] = (
                df_length_filtered
                .index
                .get_level_values(level=0)
                .drop_duplicates()
                .to_list()
            )

            feature_data[feature_set] = df_length_filtered
            n_obs["ts_features_initial"] = len(df.index.get_level_values(level=0).unique())
            n_obs["ts_features_len_filtered"] = len(df_length_filtered.index.get_level_values(level=0).unique())
        else:
            n_obs[f"{feature_set}_initial"] = len(df.index)
            feature_valid_runs[feature_set] = df.index.to_list()
            feature_data[feature_set] = df

    # Get the common run_ids
    print('---Getting common run ids')
    run_ids_intersect = set(label_runs).intersection(*[feature_valid_runs[fs] for fs in feature_specification])
    run_ids_intersect = sorted(list(run_ids_intersect))
    n_obs["final_runs"] = len(run_ids_intersect)

    # Get final labels with intersected run ids
    labels = labels[run_ids_intersect].sort_index()

    # Get final features with intersected run ids
    feature_data_intersect = {}
    feature_columns_intersect = {}
    metadata = {'run_ids': run_ids_intersect,
                'n_observations': n_obs}

    for feature_set in feature_data:
        print(f'---Compiling final features for {feature_set}')
        df = feature_data[feature_set]
        if feature_set == 'ts_features':
            df_run_id_index = df.reset_index().set_index('run_id')
            df_intersect_run_ids = df_run_id_index.loc[run_ids_intersect]
            df_intersect_run_ids = (df_intersect_run_ids
                                    .reset_index()
                                    .set_index(['run_id', 'time_date'])
                                    .sort_index())
        else:
            df_intersect_run_ids = df.loc[run_ids_intersect].sort_index()
        feature_data_intersect[feature_set] = df_intersect_run_ids
        feature_columns_intersect[feature_set] = df_intersect_run_ids.columns.to_list()

        if feature_set == 'ts_features':
            print('---Resampling final ts features')
            # Save the run lengths
            run_lengths = run_lengths[run_ids_intersect].sort_index()

            # Resample, backfill, and truncate to square off lengths to standard length
            standard_run_length = get_standard_run_length(run_lengths,
                                                          standard_run_length_quantile,
                                                          resample_freq_minutes)
            df_resampled = df_intersect_run_ids.groupby(level=0).apply(
                resample_df_to_standard_run_length,
                keep_start=keep_start,
                sample_freq=resample_freq_minutes,
                standard_run_length=standard_run_length
            )

            assert df_resampled.groupby(level=0).size().min() == standard_run_length / resample_freq_minutes, \
                f"Min obs per group is {df_resampled.groupby(level=0).size().min()}, " \
                f"not standard of {standard_run_length / resample_freq_minutes}"

            assert df_resampled.groupby(level=0).size().max() == standard_run_length / resample_freq_minutes, \
                f"Max obs per group is {df_resampled.groupby(level=0).size().max()}, " \
                f"not standard of {standard_run_length / resample_freq_minutes}"

            feature_data_intersect['ts_features_resampled'] = df_resampled
            feature_columns_intersect['ts_features_resampled'] = df_resampled.columns.to_list()

            metadata['ts_resampled_index'] = feature_data_intersect['ts_features_resampled'].index.to_frame()
            metadata['ts_index'] = feature_data_intersect['ts_features'].index.to_frame()
            metadata['run_lengths'] = run_lengths

    # Package metadata
    metadata['feature_columns'] = feature_columns_intersect

    if metadata_dir is not None:
        print("---Writing metadata from data thaw.")
        write_data_thaw_metadata(metadata, metadata_dir)
    else:
        print('---Skipping writing metadata')

    return labels, feature_data_intersect, metadata
