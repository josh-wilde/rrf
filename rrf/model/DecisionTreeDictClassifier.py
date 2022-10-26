import numpy as np

import torch

from skorch.helper import SliceDict
from sklearn.tree import DecisionTreeClassifier

from .utils import convert_slice_dict_to_tensor, validate_decision_tree_features


class DecisionTreeDictClassifier(DecisionTreeClassifier):

    def __init__(
        self,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0,
        fail_threshold=0.5
    ):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            random_state=random_state,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha
        )
        self.fail_threshold = fail_threshold

    def fit(self, X_slice_dict: SliceDict, y: torch.Tensor):
        # Validate that features are 2D
        validate_decision_tree_features(X_slice_dict)

        # convert to single tensor for fitting
        X = convert_slice_dict_to_tensor(X_slice_dict)

        return super().fit(X, y)

    def predict(self, X_slice_dict: SliceDict) -> np.ndarray:
        # Validate that features are 2D
        validate_decision_tree_features(X_slice_dict)

        if len(self.classes_) == 2:
            probs = self.predict_proba(X_slice_dict)
            predicate = probs[:, 1] > self.fail_threshold
            return np.where(predicate, self.classes_[1], self.classes_[0])
        else:
            # convert to single tensor for fitting
            X = convert_slice_dict_to_tensor(X_slice_dict)
            return super().predict(X)

    def predict_proba(self, X_slice_dict: SliceDict) -> np.ndarray:
        # Validate that features are 2D
        validate_decision_tree_features(X_slice_dict)

        # convert to single tensor for fitting
        X = convert_slice_dict_to_tensor(X_slice_dict)

        return super().predict_proba(X)
