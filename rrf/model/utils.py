import torch
from typing import List
from skorch.helper import SliceDict


def validate_X_type(X):
    """Assert that X is the right type"""
    assert type(X) == SliceDict, f'{type(X)} is invalid for X. Must be a skorch SliceDict.'


def validate_labels(y, binary: bool = False):
    assert type(y) == torch.Tensor, 'y must be a torch tensor'
    assert y.dtype == torch.int64, f'{y.dtype} is invalid for y, must be torch.int64.'
    assert torch.all(y >= 0), 'y must be nonnegative.'
    #assert len(y[y == 0]) / len(y) >= 0.5, '0 must be the majority class for y.'

    # Must have at least 1 nonzero entry in y (one true negative)
    assert y.sum() > 0, 'Need at least one true negative in y'
    if binary:
        assert torch.all(y.unique() == torch.tensor([0,1])), 'y must be binary'


def validate_decision_tree_features(X_slice_dict: SliceDict):
    allowed_feature_sets = ['X_threshold', 'X_static']
    for feature_set in X_slice_dict:
        assert feature_set in allowed_feature_sets, f"{feature_set} invalid. Must be in {allowed_feature_sets}."
        assert X_slice_dict[feature_set].dim() == 2, f"{feature_set} is {X_slice_dict[feature_set].dim()}-dimensional, must be 2d."


def validate_necessary_params(given_params: List[str], necessary_params: List[str]):
    '''Validate that all params in necessary_params are in given_params'''
    given_params_set, necessary_params_set = set(given_params), set(necessary_params)
    given_necessary_intersection = given_params_set.intersection(necessary_params_set)
    necessary_missing = necessary_params_set - given_necessary_intersection
    assert len(necessary_missing) == 0, (
        f'Parameters {necessary_missing} are missing and required.'
    )


def validate_cnn_necessary_params(given_params: List[str]):
    necessary_params = ['cnn_block_layers',
                        'cnn_seq_len',
                        'cnn_kernel_size',
                        'cnn_initial_conv_channels',
                        'cnn_batch_norm']
    validate_necessary_params(given_params, necessary_params)


def validate_lstm_necessary_params(given_params: List[str]):
    necessary_params = ['lstm_hidden_dim',
                        'lstm_n_layers',
                        'lstm_dropout_frac']
    validate_necessary_params(given_params, necessary_params)


def validate_resnet18_necessary_params(given_params: List[str]):
    necessary_params = ['resnet18_seq_len']
    validate_necessary_params(given_params, necessary_params)


def convert_slice_dict_to_tensor(X_slice_dict: SliceDict) -> torch.Tensor:
    return torch.cat(list(X_slice_dict.values()), dim=1)
