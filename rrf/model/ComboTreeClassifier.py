from typing import Optional

import numpy as np
import torch
from skorch.helper import SliceDict
from sklearn.base import BaseEstimator, ClassifierMixin

from .SupClassifier import SupClassifier
from .DecisionTreeDictClassifier import DecisionTreeDictClassifier
from .RandomForestDictClassifier import RandomForestDictClassifier
from .XGBDictClassifier import XGBDictClassifier
from .utils import convert_slice_dict_to_tensor, validate_X_type, validate_labels


class ComboTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            clf1_type: str = 'DecisionTreeDictClassifier', # or SupClassifier
            clf1_tree_max_depth: Optional[int] = None,
            clf1_tree_max_impurity: float = 0.0,
            clf1_tree_random_state: int = 123,
            clf1_sup_max_fp_rate: float = 0.0,
            clf1_sup_opt_name: str = 'GreedyStepDown',
            clf2_type: str = 'DecisionTreeDictClassifier',
            clf2_tree_max_depth: Optional[int] = None,
            clf2_tree_random_state: int = 123,
            clf2_ens_n_estimators: int = 100,
            clf2_xgb_max_delta_step: int = 0,
            clf2_xgb_objective: str = 'binary:logistic',
            restrict_clf2_training_data: bool = True,
            fail_threshold: float = 0.5,
            **kwargs
    ):
        self.clf1_type = clf1_type
        self.clf1_tree_max_depth = clf1_tree_max_depth
        self.clf1_tree_max_impurity = clf1_tree_max_impurity
        self.clf1_tree_random_state = clf1_tree_random_state
        self.clf1_sup_max_fp_rate = clf1_sup_max_fp_rate
        self.clf1_sup_opt_name = clf1_sup_opt_name
        self.clf2_type = clf2_type
        self.clf2_tree_max_depth = clf2_tree_max_depth
        self.clf2_tree_random_state = clf2_tree_random_state
        self.clf2_ens_n_estimators = clf2_ens_n_estimators
        self.clf2_xgb_max_delta_step = clf2_xgb_max_delta_step
        self.clf2_xgb_objective = clf2_xgb_objective
        self.restrict_clf2_training_data = restrict_clf2_training_data
        self.fail_threshold = fail_threshold

    def _validate_features(self, X: SliceDict):
        # Need the threshold data for the SupClassifier
        assert 'X_threshold' in X.keys(), 'X_threshold must be a key of X_slice_dict.'

    def _validate_input_params(self):
        allowed_clf1_types = ['DecisionTreeDictClassifier', 'SupClassifier']
        allowed_clf2_types = ['DecisionTreeDictClassifier', 'RandomForestDictClassifier', 'XGBDictClassifier']
        assert self.clf1_type in allowed_clf1_types, (
            f"{self.clf1_type} first classifier type not in {allowed_clf1_types}")

        assert self.clf2_type in allowed_clf2_types, (
            f"{self.clf2_type} second classifier type not in {allowed_clf2_types}")

    def fit(
        self,
        X_slice_dict: SliceDict,
        y: torch.Tensor,
        clf1_sup_fit_verbose: bool = False,
        clf1_write_sup_metadata_dirpath: str = None
    ):
        # Validate y
        validate_labels(y)

        # Validate X_slice_dict
        validate_X_type(X_slice_dict)
        self._validate_features(X_slice_dict)

        # Validate params
        self._validate_input_params()

        self.classes_ = torch.unique(y).numpy()
        self.n_features = sum([X_slice_dict[feature_set].shape[1] for feature_set in X_slice_dict])

        # Binary y for the SupClassifier
        y_binary = (y > 0).long()

        # Instantiate the first classifier
        if self.clf1_type == 'DecisionTreeDictClassifier':
            self.clf1 = DecisionTreeDictClassifier(
                max_depth=self.clf1_tree_max_depth,
                random_state=self.clf1_tree_random_state
            )
        else:
            self.clf1 = SupClassifier(
                max_fp_rate=self.clf1_sup_max_fp_rate,
                opt_name=self.clf1_sup_opt_name
            )

        # Instantiate the second classifier
        if self.clf2_type == 'DecisionTreeDictClassifier':
            self.clf2 = DecisionTreeDictClassifier(
                max_depth=self.clf2_tree_max_depth,
                random_state=self.clf2_tree_random_state
            )
        elif self.clf2_type == 'RandomForestDictClassifier':
            self.clf2 = RandomForestDictClassifier(
                n_estimators=self.clf2_ens_n_estimators,
                max_depth=self.clf2_tree_max_depth,
                random_state=self.clf2_tree_random_state
            )
        else:
            self.clf2 = XGBDictClassifier(
                n_estimators=self.clf2_ens_n_estimators,
                max_depth=self.clf2_tree_max_depth,
                max_delta_step=self.clf2_xgb_max_delta_step,
                objective=self.clf2_xgb_objective,
                random_state=self.clf2_tree_random_state
            )

        # Fit the first classifier
        if self.clf1_type == 'DecisionTreeDictClassifier':
            self.clf1.fit(X_slice_dict, y_binary)
            # Also need to set the best leaf info for later reference
            self._set_clf1_tree_best_leaf()
        else:
            self.clf1.fit(
                X_slice_dict,
                y_binary,
                verbose=clf1_sup_fit_verbose,
                write_sup_metadata_dirpath=clf1_write_sup_metadata_dirpath
            )

        # Decide whether to use all data or not
        if self.restrict_clf2_training_data:
            # Get the indices where the second classifier should take over
            clf1_classified_indices = self.get_clf1_classified_indices(X_slice_dict)
            clf1_fail_indices = np.setdiff1d(range(y.shape[0]), clf1_classified_indices)
            X_slice_dict_to_clf2 = X_slice_dict[clf1_fail_indices]
            y_to_clf2 = y[clf1_fail_indices]

            # Then can fit the second classifier
            self.clf2.fit(X_slice_dict_to_clf2, y_to_clf2)
        else:
            self.clf2.fit(X_slice_dict, y)

        return self

    def _set_clf1_tree_best_leaf(self):
        # Look through the tree to find the leaf with the most 0 predictions with impurity below max
        # Define arrays from the clf1 tree
        children_left = self.clf1.tree_.children_left
        children_right = self.clf1.tree_.children_right
        count_pred_pass = [v[0][0] for v in self.clf1.tree_.value]
        impurities = self.clf1.tree_.impurity

        # Stack of nodes where each item contains (node_id, list of parents, directions taken from parents)
        stack = [(0, list(), '')]

        # Initialize the max leaf node
        max_leaf_info = (-1, list(), '')
        max_leaf_count_pred_pass = -1

        # Search the tree starting at the root node
        while len(stack) > 0:
            node_id, parents, parent_directions = stack.pop()
            is_split_node = children_left[node_id] != children_right[node_id]
            if is_split_node:
                stack.append((
                    children_left[node_id],
                    parents + [node_id],
                    parent_directions + 'L'
                ))
                stack.append((
                    children_right[node_id],
                    parents + [node_id],
                    parent_directions + 'R'
                ))
            else:
                if (count_pred_pass[node_id] > max_leaf_count_pred_pass) and \
                        (impurities[node_id] <= self.clf1_tree_max_impurity):
                    max_leaf_count_pred_pass = count_pred_pass[node_id]
                    max_leaf_info = (node_id, parents, parent_directions)

        # Throw an error if there are no leaves that meet the max impurity threshold
        if max_leaf_info[0] == -1:
            raise ValueError(f'No leaves below max impurity threshold of {self.clf1_tree_max_impurity}.')

        print(f"Best leaf has {max_leaf_count_pred_pass} predicted pass.")

        self.clf1_tree_best_leaf_info = max_leaf_info

    def get_clf1_classified_indices(self, X_slice_dict) -> np.ndarray:
        if self.clf1_type == 'DecisionTreeDictClassifier':
            best_leaf_idx = self.clf1_tree_best_leaf_info[0]
            X = convert_slice_dict_to_tensor(X_slice_dict)
            leaves = self.clf1.apply(X)
            return (leaves == best_leaf_idx).nonzero()[0]
        else:  # if it is a SupClassifier
            return (self.clf1.predict(X_slice_dict) == 0).nonzero()[0]

    def predict_proba(self, X_slice_dict: SliceDict) -> np.ndarray:
        # First grab indices where the first classifier makes the final prediction
        all_indices = np.array(range(convert_slice_dict_to_tensor(X_slice_dict).shape[0]))
        clf1_indices = self.get_clf1_classified_indices(X_slice_dict)
        clf2_indices = np.setdiff1d(all_indices, clf1_indices)

        pred_proba = np.zeros((all_indices.shape[0], len(self.classes_)))

        if self.clf1_type == 'DecisionTreeDictClassifier':
            best_leaf_idx = self.clf1_tree_best_leaf_info[0]
            leaf_values = self.clf1.tree_.value[best_leaf_idx]
            pred_proba[clf1_indices] = leaf_values / np.sum(leaf_values)
        else:
            pred_proba[clf1_indices] = np.array([1.0] + [0.0] * (pred_proba.shape[1] - 1))

        if len(clf2_indices) > 0:
            pred_proba[clf2_indices] = self.clf2.predict_proba(X_slice_dict[clf2_indices])

        return pred_proba

    def predict(self, X_slice_dict: SliceDict) -> np.ndarray:
        # adapted from solution here https://scikit-learn.org/stable/developers/develop.html
        # using decision_function()
        probs = self.predict_proba(X_slice_dict)

        if len(self.classes_) == 2:
            predicate = probs[:, 1] > self.fail_threshold
            return np.where(predicate, self.classes_[1], self.classes_[0])
        else:
            return np.argmax(probs, axis=1)
