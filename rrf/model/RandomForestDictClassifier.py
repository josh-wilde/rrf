import numpy as np
import torch
from skorch.helper import SliceDict
from sklearn.ensemble import RandomForestClassifier

from .utils import convert_slice_dict_to_tensor, validate_decision_tree_features


class RandomForestDictClassifier(RandomForestClassifier):

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        fail_threshold=0.5,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples
        )
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
