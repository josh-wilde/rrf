from typing import Any
import numpy as np
import torch
from skorch.helper import SliceDict
from xgboost import XGBClassifier

from .utils import convert_slice_dict_to_tensor, validate_decision_tree_features


class XGBDictClassifier(XGBClassifier):
    def __init__(
        self,
        *,
        objective: str = "binary:logistic",
        fail_threshold: float = 0.5,
        **kwargs: Any
    ) -> None:
        super().__init__(objective=objective, **kwargs)
        self.fail_threshold = fail_threshold

    def fit(self, X_slice_dict: SliceDict, y: torch.Tensor):
        if type(X_slice_dict) == SliceDict:
            # Validate that features are 2D
            validate_decision_tree_features(X_slice_dict)

            # convert to single tensor for fitting
            X = convert_slice_dict_to_tensor(X_slice_dict)
        else:
            X = X_slice_dict

        return super().fit(X, y)

    def predict(self, X_slice_dict: SliceDict) -> np.ndarray:
        if type(X_slice_dict) == SliceDict:
            # Validate that features are 2D
            validate_decision_tree_features(X_slice_dict)

            # convert to single tensor for fitting
            X = convert_slice_dict_to_tensor(X_slice_dict)
        else:
            X = X_slice_dict

        if len(self.classes_) == 2:
            probs = self.predict_proba(X)
            predicate = self.predict_proba(X)[:, 1] > self.fail_threshold
            return np.where(predicate, self.classes_[1], self.classes_[0])
        else:
            return super().predict(X)

    def predict_proba(self, X_slice_dict: SliceDict) -> np.ndarray:
        if type(X_slice_dict) == SliceDict:
            # Validate that features are 2D
            validate_decision_tree_features(X_slice_dict)

            # convert to single tensor for fitting
            X = convert_slice_dict_to_tensor(X_slice_dict)
        else:
            X = X_slice_dict

        return super().predict_proba(X)
