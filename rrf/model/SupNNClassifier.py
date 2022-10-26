from typing import List, Dict, Optional, Any
import numpy as np
import torch
from torch import nn
from torch import optim

from skorch import NeuralNetClassifier
from skorch.helper import SliceDict
from sklearn.base import BaseEstimator, ClassifierMixin

from .MBINet import MBINet
from .SupClassifier import SupClassifier
from .NeuralNetThreshClassifier import NeuralNetThreshClassifier
from .utils import validate_X_type, validate_labels


class SupNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        sup_max_fp_rate: float = 0.0,
        sup_opt_name: str = 'GreedyStepDown',
        restrict_nn_training_data: bool = False,
        module: nn.Module = MBINet,
        ts_module_name: Optional[str] = None,
        nn_feature_sets: List[str] = [],
        nn_n_features: Dict[str, int] = {},
        nn_top_dropout_frac: float = 0.5,
        nn_kwargs: Dict[str, Any] = {},
        criterion: nn.Module = nn.NLLLoss,
        optimizer: optim.Optimizer = optim.Adam,
        lr: float = 0.01,
        max_epochs: int = 10,
        batch_size: int = 128,
        train_split: bool = False,
        fail_threshold: float = 0.5,
        **kwargs
    ):
        self.sup_max_fp_rate = sup_max_fp_rate
        self.sup_opt_name = sup_opt_name
        self.restrict_nn_training_data = restrict_nn_training_data
        self.module = module
        self.ts_module_name = ts_module_name
        self.nn_feature_sets = nn_feature_sets
        self.nn_n_features = nn_n_features
        self.nn_top_dropout_frac = nn_top_dropout_frac
        self.nn_kwargs = nn_kwargs
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.train_split = train_split
        self.fail_threshold = fail_threshold

    def _validate_features(self, X: SliceDict):
        # Need the threshold data for the SupClassifier
        assert 'X_threshold' in X.keys(), 'X_threshold must be a key of X_slice_dict.'

        if 'ts' in self.nn_feature_sets:
            if self.ts_module_name == 'cnn':
                assert 'X_ts_resampled' in X.keys(), 'X_ts_resampled must exist to use cnn'
            elif self.ts_module_name == 'resnet18':
                assert 'X_ts_resampled' in X.keys(), 'X_ts_resampled must exist to use resnet18'
            elif self.ts_module_name == 'lstm':
                assert 'X_ts_resampled' in X.keys() and 'run_lengths' in X.keys(), (
                    'X_ts_padded and run_lengths must exist to use lstm'
                )
        if 'static' in self.nn_feature_sets:
            assert 'X_static' in X.keys(), (
                'X_static must exist to use static features'
            )

    def _validate_input_params(self):
        # Need ts_module_name
        if 'ts' in self.nn_feature_sets:
            assert self.ts_module_name is not None, 'Need ts_module_name'

        # Need n_features for every feature set in feature_sets
        for feature_set in self.nn_feature_sets:
            assert feature_set in self.nn_n_features, (
                f'Need n_features for {feature_set}'
            )

    def fit(
        self,
        X_slice_dict: SliceDict,
        y: torch.Tensor,
        sup_fit_verbose: bool = False,
        write_sup_metadata_dirpath: str = None
    ):
        # Validate y
        validate_labels(y)

        # Validate X_slice_dict
        validate_X_type(X_slice_dict)
        self._validate_features(X_slice_dict)

        # Validate params
        self._validate_input_params()

        self.classes_ = torch.unique(y).numpy()

        # Binary y for the SupClassifier
        y_binary = (y > 0).long()

        # Instantiate the SupClassifier and the NeuralNetClassifier
        self.sup_clf = SupClassifier(max_fp_rate=self.sup_max_fp_rate,
                                     opt_name=self.sup_opt_name)

        # Instantiate the NeuralNetClassifier
        if 'ts' in self.nn_feature_sets:
            if self.ts_module_name == 'cnn':
                self.nn_clf = NeuralNetClassifier(
                    self.module,
                    module__input_feature_sets=self.nn_feature_sets,
                    module__input_n_features=self.nn_n_features,
                    module__ts_module_name='cnn',
                    module__top_dropout_frac=self.nn_top_dropout_frac,
                    module__n_classes=len(self.classes_),
                    module__cnn_block_layers=self.nn_kwargs['cnn_block_layers'],
                    module__cnn_seq_len=self.nn_kwargs['cnn_seq_len'],
                    module__cnn_kernel_size=self.nn_kwargs['cnn_kernel_size'],
                    module__cnn_initial_conv_channels=self.nn_kwargs['cnn_initial_conv_channels'],
                    module__cnn_batch_norm=self.nn_kwargs['cnn_batch_norm'],
                    classes=self.classes_,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    lr=self.lr,
                    max_epochs=self.max_epochs,
                    batch_size=self.batch_size,
                    train_split=self.train_split
                )
            elif self.ts_module_name == 'cnn_threshold':
                self.nn_clf = NeuralNetThreshClassifier(
                    self.module,
                    module__input_feature_sets=self.nn_feature_sets,
                    module__input_n_features=self.nn_n_features,
                    module__ts_module_name='cnn',
                    module__top_dropout_frac=self.nn_top_dropout_frac,
                    module__n_classes=len(self.classes_),
                    module__cnn_block_layers=self.nn_kwargs['cnn_block_layers'],
                    module__cnn_seq_len=self.nn_kwargs['cnn_seq_len'],
                    module__cnn_kernel_size=self.nn_kwargs['cnn_kernel_size'],
                    module__cnn_initial_conv_channels=self.nn_kwargs['cnn_initial_conv_channels'],
                    module__cnn_batch_norm=self.nn_kwargs['cnn_batch_norm'],
                    classes=self.classes_,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    lr=self.lr,
                    max_epochs=self.max_epochs,
                    batch_size=self.batch_size,
                    train_split=self.train_split,
                    fail_threshold=self.fail_threshold
                )
            elif self.ts_module_name == 'resnet18':
                self.nn_clf = NeuralNetClassifier(
                    self.module,
                    module__input_feature_sets=self.nn_feature_sets,
                    module__input_n_features=self.nn_n_features,
                    module__ts_module_name='resnet18',
                    module__top_dropout_frac=self.nn_top_dropout_frac,
                    module__n_classes=len(self.classes_),
                    module__resnet18_seq_len=self.nn_kwargs['resnet18_seq_len'],
                    classes=self.classes_,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    lr=self.lr,
                    max_epochs=self.max_epochs,
                    batch_size=self.batch_size,
                    train_split=self.train_split,
                )
            elif self.ts_module_name == 'lstm':
                self.nn_clf = NeuralNetClassifier(
                    self.module,
                    module__input_feature_sets=self.nn_feature_sets,
                    module__input_n_features=self.nn_n_features,
                    module__ts_module_name='lstm',
                    module__top_dropout_frac=self.nn_top_dropout_frac,
                    module__n_classes=len(self.classes_),
                    module__lstm_hidden_dim=self.nn_kwargs['lstm_hidden_dim'],
                    module__lstm_n_layers=self.nn_kwargs['lstm_n_layers'],
                    module__lstm_dropout_frac=self.nn_kwargs['lstm_dropout_frac'],
                    classes=self.classes_,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    lr=self.lr,
                    max_epochs=self.max_epochs,
                    batch_size=self.batch_size,
                    train_split=self.train_split
                )
            else:
                raise ValueError(
                    f'{self.ts_module_name} is invalid, only lstm, cnn, resnet18 are valid for ts_module_name.'
                )

        else:
            self.nn_clf = NeuralNetClassifier(
                self.module,
                module__input_feature_sets=self.nn_feature_sets,
                module__input_n_features=self.nn_n_features,
                module__ts_module_name=None,
                module__top_dropout_frac=self.nn_top_dropout_frac,
                module__n_classes=len(self.classes_),
                classes=self.classes_,
                criterion=self.criterion,
                optimizer=self.optimizer,
                lr=self.lr,
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                train_split=self.train_split
            )

        # Fit the sup classifier
        self.sup_clf.fit(
            X_slice_dict,
            y_binary,
            verbose=sup_fit_verbose,
            write_sup_metadata_dirpath=write_sup_metadata_dirpath
        )

        # It is very much not clear to me if we should do this restriction when training
        # so should use self.restrict_nn_training_data as parameter and optimize over it
        if self.restrict_nn_training_data:
            # Predict in-sample using the SupClassifier
            yhat_sup = self.sup_clf.predict(X_slice_dict)

            # Take only the observations where the SupClassifier says no pass
            X_slice_dict_to_nn = X_slice_dict[yhat_sup > 0]
            y_to_nn = y[yhat_sup > 0]
        else:
            X_slice_dict_to_nn = X_slice_dict
            y_to_nn = y

        # Then can fit the NN
        self.nn_clf.fit(X_slice_dict_to_nn, y_to_nn)

        # And set the history attribute
        self.history = self.nn_clf.history

        return self

    def predict_proba(self, X_slice_dict: SliceDict) -> np.ndarray:
        # First grab indices where the sup classifier makes the final prediction
        sup_indices = self.get_sup_classified_indices(X_slice_dict)

        # Then predict everything for the nn
        # The NeuralNetClassifier returns np.ndarray predictions
        nn_pred_proba = torch.tensor(self.nn_clf.predict_proba(X_slice_dict), dtype=torch.float32)

        # Then issue the predictions
        # Need to combine the predictions above
        # If sup_pred == 0, then use 1 and 0 out the rest for nn_pred_proba
        # If sup_pred == 1, then use the nn_pred_proba
        pred_proba = nn_pred_proba
        pred_proba[sup_indices] = torch.tensor([1.0] + [0.0] * (nn_pred_proba.shape[1] - 1))

        return pred_proba.numpy()

    def predict(self, X_slice_dict: SliceDict) -> np.ndarray:
        # adapted from solution here https://scikit-learn.org/stable/developers/develop.html
        # using decision_function()
        probs = self.predict_proba(X_slice_dict)

        if len(self.classes_) == 2:
            predicate = probs[:, 1] > self.fail_threshold
            return np.where(predicate, self.classes_[1], self.classes_[0])
        else:
            return np.argmax(probs, axis=1)

    def get_sup_classified_indices(self, X_slice_dict: SliceDict) -> np.ndarray:
        # First need to predict on X
        sup_pred = self.sup_clf.predict(X_slice_dict)

        # Then need to return the indices where the prediction is 0
        return (sup_pred == 0).nonzero()[0]
