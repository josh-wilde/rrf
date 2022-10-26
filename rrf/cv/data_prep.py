import numpy as np
import pandas as pd
from math import ceil
from typing import Dict, Any, Tuple
import torch
from skorch.helper import SliceDict
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler

from rrf.data_thaw import thaw_data_to_pandas, convert_pandas_to_tensors


def collect_nested_cv_data(config: Dict[str, Any]):
    '''
    Parameters
    ----------
    config: dictionary
        Parameters that define the experiment. This should combine the data thaw
        And the model configuration parameters.
    '''

    # Make sure config has everything
    assert 'feature_specification' in config, 'feature specification not in config'
    assert 'label_specification' in config, 'label specification not in config'
    assert 'ts_standardization' in config, 'Time series standardization not in config'
    assert 'data_thaw' in config, 'Data thaw parameters not in config'

    # Thaw the data
    # The initial thaw gives labels and feature_data back as pandas dataframes, in long format for time series
    labels, feature_data, metadata = thaw_data_to_pandas(
        config['feature_specification'],
        config['label_specification'],
        **config['ts_standardization'],
        metadata_dir=config['data_thaw'].get('pandas_metadata_dir')
    )
    # This will convert everything to tensors, and time series data will now be 3D
    # feature_data is a dictionary still, not a SliceDict
    labels, feature_data = convert_pandas_to_tensors(labels, feature_data)

    n_obs = metadata['n_observations']['final_runs']

    return labels, feature_data, n_obs


def split_feature_data(
        feature_data: Dict[str, Any],
        train_ix: np.ndarray,
        test_ix: np.ndarray
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    '''
    Take a SliceDict with array-like elements and separate into train and test, using the first dimension

    Parameters
    ----------
    feature_data: dictionary with array-like values for string keys
        dictionary with array-like elements, most likely torch tensors
    train_ix: np.ndarray
        list of training observations indices
    test_ix: np.ndarray
        list of test set indices

    Returns
    -------
    train_output, test_output: Dicts
        tuple with two elements, a training Dict and a test Dict
    '''
    train = {}
    test = {}
    for feature_set, df in feature_data.items():
        train[feature_set], test[feature_set] = df[train_ix], df[test_ix]

    return train, test


def random_over_sample(
    X: Dict[str, Any],
    y: Any,
    minority_class_frac: float = None
):
    #Implements the RandomOverSampler for X as dictionary of tensors, y as tensor
    # Create a dataframe and series for use in the RandomOverSampler
    # dataframe has specific structure: index column indexes the tensors
    sampling_X = pd.DataFrame(y, columns=['y']).reset_index()
    sampling_y = pd.Series(y, name='y')

    # Create the number of minority samples
    # Goal: keep # of majority class samples the same
    # Increase all minority classes so they are equal minority_class_frac share of final samples
    if minority_class_frac:
        label_counts = sampling_y.value_counts()
        count_minority_classes = len(label_counts) - 1
        majority_class = label_counts.idxmax()

        # Keep the majority class samples the same
        n_majority_class = len(sampling_y[sampling_y == majority_class])

        # Calculate the minority class sample count
        n_minority_classes = ceil(minority_class_frac * n_majority_class / ((1-minority_class_frac) * count_minority_classes))

        # Create sampling strategy dictionary
        sampling_strategy = {k: n_majority_class if k == majority_class else n_minority_classes
                             for k, v in dict(label_counts).items()}
        print(f"Sampling strategy: {sampling_strategy}")
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=789)

    else:
        ros = RandomOverSampler(random_state=789)

    # Resample
    resampling_X, _ = ros.fit_resample(sampling_X, sampling_y)

    # Gather the resampling indices
    resampled_indices = resampling_X['index'].values

    # Then resample the tensors
    output_X = {k: v[resampled_indices] for k, v in X.items()}
    output_y = y[resampled_indices]

    return output_X, output_y


def configure_outer_split(cv_setup: Dict[str, Any]):
    # Configure outer model selection CV split
    # Could be either model selection, or randomized
    if 'randomized_outer_start_split' in cv_setup:
        start_split_idx = cv_setup['randomized_outer_start_split']
        end_split_idx = cv_setup['randomized_outer_end_split']
        n_outer_splits = end_split_idx - start_split_idx
        split_index_range = range(start_split_idx, end_split_idx)
        cv_outer = StratifiedShuffleSplit(
            n_splits=n_outer_splits,
            test_size=cv_setup['test_frac'],
            random_state=123+start_split_idx
        )
    else:
        n_outer_splits = cv_setup['model_selection_n_splits']
        split_index_range = range(n_outer_splits)
        cv_outer = StratifiedKFold(
            n_splits=n_outer_splits,
            shuffle=True,
            random_state=123
        )

    return cv_outer, split_index_range


def get_outer_split_data(
    feature_data: Dict[str, Any],
    labels: Any,
    train_ix: np.ndarray,
    test_ix: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[SliceDict, Any, SliceDict, Any]:
    # Grab the CV setup parameters
    cv_setup = config['nested_cv_setup']

    # Split into train and test sets
    feature_data_train_raw, feature_data_test = split_feature_data(feature_data, train_ix, test_ix)
    labels_train_raw, labels_test = labels[train_ix], labels[test_ix]

    # Optionally oversample the training data
    if cv_setup['oversample']:
        feature_data_train, labels_train = random_over_sample(
            feature_data_train_raw,
            labels_train_raw,
            minority_class_frac=cv_setup.get('minority_class_frac')
        )
    else:
        feature_data_train, labels_train = feature_data_train_raw, labels_train_raw

    print(f"Training label counts: {np.unique(labels_train, return_counts=True)}")

    # Convert feature dicts to SliceDicts for compatibility with skorch
    feature_data_train, feature_data_test = SliceDict(**feature_data_train), SliceDict(**feature_data_test)

    return feature_data_train, labels_train, feature_data_test, labels_test


def save_outer_split_data(
    config: Dict[str, Any],
    split_id: int,
    n_total_splits: int,
    labels_train: Any,
    labels_test: Any,
    feature_data_train: Dict[str, Any],
    feature_data_test: Dict[str, Any]
):
    data_save_dirpath = config['nested_cv_setup'].get('cv_data_save_dirpath')
    split_save_prefix = f"split{split_id}_{n_total_splits}"

    # Save labels
    pd.DataFrame(labels_train.numpy()).to_csv(f"{data_save_dirpath}/{split_save_prefix}_labels_train.csv")
    pd.DataFrame(labels_test.numpy()).to_csv(f"{data_save_dirpath}/{split_save_prefix}_labels_test.csv")

    # Save features
    for feature_set in feature_data_train:
        if '_ts_' in feature_set:
            torch.save(feature_data_train[feature_set], f"{data_save_dirpath}/{split_save_prefix}_{feature_set}_train.pt")
            torch.save(feature_data_test[feature_set], f"{data_save_dirpath}/{split_save_prefix}_{feature_set}_test.pt")
        else:
            pd.DataFrame(feature_data_train[feature_set].numpy()).to_csv(f"{data_save_dirpath}/{split_save_prefix}_{feature_set}_train.csv")
            pd.DataFrame(feature_data_test[feature_set].numpy()).to_csv(f"{data_save_dirpath}/{split_save_prefix}_{feature_set}_test.csv")
