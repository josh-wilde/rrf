import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from math import floor
import os

import torch

from skorch.helper import SliceDict
from sklearn.base import BaseEstimator, ClassifierMixin

from rrf.rules.GreedyStepDown import GreedyStepDown
from .utils import validate_labels, validate_X_type


# Use the estimator checks referenced in here: https://scikit-learn.org/stable/developers/develop.html
class SupClassifier(BaseEstimator, ClassifierMixin):

    # Follow https://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
    def __init__(self,
                 max_fp_rate: float = 0.0,
                 opt_name: str = 'GreedyStepDown',
                 fail_threshold: float = 0.5,
                 **kwargs):
        self.max_fp_rate = max_fp_rate
        self.opt_name = opt_name
        self.fail_threshold = fail_threshold

    def _validate_fit_params(self):
        """Validate parameters used in fit()"""
        valid_optimizers = ['GreedyStepDown']
        assert self.opt_name in valid_optimizers, f'{self.opt_name} is invalid for opt_name, must be in {valid_optimizers}'

        assert 1 >= self.max_fp_rate >= 0, f'max_fp_rate {self.max_fp_rate} must be between 0 and 1.'

    @staticmethod
    def _validate_features(X: Dict[str, Any]):
        """Make sure correct key is available in X, which is a dictionary or SliceDict"""
        assert 'X_threshold' in X.keys(), 'X_threshold must be a key of X_slice_dict.'

    # Need to define the fit function
    # Need a requirement that 0 be the pass class and everything else fails
    # Probably should remove unknown before fitting
    def fit(
        self,
        X_slice_dict: SliceDict,
        y: torch.Tensor,
        verbose: bool = False,
        write_sup_metadata_dirpath: Optional[str] = None
    ):
        # Validate parameters
        self._validate_fit_params()

        # Validate y
        validate_labels(y, binary=True)

        # Validate X_slice_dict
        validate_X_type(X_slice_dict)
        self._validate_features(X_slice_dict)

        # Get the features
        X = X_slice_dict['X_threshold']
        self.n_features = X.shape[1]

        # Set the classes
        self.classes_, _ = np.unique(y)

        # Transform target_fp_rate to target_fps
        true_negatives = float(y.sum())
        self.max_fps = floor(true_negatives * self.max_fp_rate)

        # Set the threshold optimizer
        # Currently no else since no other option, but set up to accommodate others
        if self.opt_name == 'GreedyStepDown':
            # This requires X as a dataframe and y as a series
            X_df, y_series = pd.DataFrame(X.numpy()), pd.Series(y.numpy())
            optimizer = GreedyStepDown(features=X_df, labels=y_series, max_fp=self.max_fps)

        # Optimize the thresholds
        # optimizer_metadata should contain all of the fitting time, sup_history, etc
        self.sups, self.fit_metadata = optimizer.optimize(verbose=verbose)
        self.sups = torch.tensor(self.sups, dtype=torch.float32)

        # Optionally write metadata
        if write_sup_metadata_dirpath:
            self.save_fit_metadata(write_sup_metadata_dirpath)

        return self

    def predict(self, X_slice_dict: SliceDict) -> np.ndarray:
        # Validate
        validate_X_type(X_slice_dict)
        self._validate_features(X_slice_dict)

        # Get features
        X = X_slice_dict['X_threshold']

        # yhat should be a tensor with a prediction for each item in X
        # 0 = pass, 1 = not pass
        yhat = 1 - torch.all(X < self.sups, axis=1).long()

        return yhat.numpy()

    def predict_proba(self, X_slice_dict: SliceDict) -> np.ndarray:
        # needs to return a numpy array with a row for each obs in X and a column for each class
        # For now, should just be 0-1, but could think about some sort of probability as well...
        # This could be interesting in combination with the other estimator to indicate when something is near a threshold

        # Validate
        validate_X_type(X_slice_dict)
        self._validate_features(X_slice_dict)

        # Get features
        X = X_slice_dict['X_threshold']

        proba_output = torch.zeros((X.shape[0], 2))
        proba_output[:, 0] = torch.all(X < self.sups, axis=1).float()
        proba_output[:, 1] = 1.0 - proba_output[:, 0]

        return proba_output.numpy()

    def save_fit_metadata(self, output_dir: str):
        """Assumes that self.fit_metadata is a dictionary that exists"""
        # Create the directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # These are pd.DataFrames
        for k in ['cm_history', 'feature_history', 'sup_history']:
            self.fit_metadata[k].to_csv(f"{output_dir}/{k}.csv")

        # This is a string
        with open(f"{output_dir}/fit_time.txt", "w") as text_file:
            print(f"Fit Time: {self.fit_metadata['fit_time']}", file=text_file)
