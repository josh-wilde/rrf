from typing import Dict, Any
import numpy as np
from math import log10
from skorch.helper import SliceDict
from skorch import NeuralNetClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from rrf.model import (
    SupClassifier,
    SupNNClassifier,
    MBINet,
    NeuralNetThreshClassifier,
    DecisionTreeDictClassifier,
    RandomForestDictClassifier,
    XGBDictClassifier,
    ComboTreeClassifier
)

from .metrics import define_grid_search_scoring_map


def initialize_classifier(classifier_config: Dict[str, Any]):
    '''
    Returns the classifier object based on the config parameters
    '''
    if classifier_config['name'] == 'SupClassifier':
        model = SupClassifier(**classifier_config['initial_params'])
    elif classifier_config['name'] == 'SupNNClassifier':
        model = SupNNClassifier(**classifier_config['initial_params'])
    elif classifier_config['name'] == 'NeuralNetClassifier':
        model = NeuralNetClassifier(module=MBINet, **classifier_config['initial_params'])
    elif classifier_config['name'] == 'NeuralNetThreshClassifier':
        model = NeuralNetThreshClassifier(module=MBINet, **classifier_config['initial_params'])
    elif classifier_config['name'] == 'DecisionTreeDictClassifier':
        model = DecisionTreeDictClassifier(**classifier_config['initial_params'])
    elif classifier_config['name'] == 'RandomForestDictClassifier':
        model = RandomForestDictClassifier(**classifier_config['initial_params'])
    elif classifier_config['name'] == 'XGBDictClassifier':
        model = XGBDictClassifier(**classifier_config['initial_params'])
    elif classifier_config['name'] == 'ComboTreeClassifier':
        model = ComboTreeClassifier(**classifier_config['initial_params'])
    else:
        raise ValueError(
            "name of classifier must be one of: SupNNClassifier, NeuralNetClassifier, SupClassifier, DecisionTreeDictClassifier, RandomForestDictClassifier, XGBDictClassifier, ComboTreeClassifier"
        )

    return model


def fix_search_space(ss: Dict[str, Any]) -> Dict[str, Any]:
    # Allows us to use parse ranges in search space, not just literals
    new_ss = {}

    for k,v in ss.items():
        if type(v) == list:
            new_ss[k] = v
        elif type(v) == dict:
            if 'range' in v:
                range_parameters = v['range']
                assert 'min_inclusive' in range_parameters, 'min_inclusive missing from range parameters'
                assert 'max_inclusive' in range_parameters, 'max_inclusive missing from range parameters'
                assert 'n_values' in range_parameters, 'n_values missing from range parameters'
                is_log10 = range_parameters.get('is_log10', False)

                if is_log10:
                    new_ss[k] = list(np.logspace(log10(range_parameters['min_inclusive']),
                                                 log10(range_parameters['max_inclusive']),
                                                 range_parameters['n_values']))
                else:
                    new_ss[k] = list(np.linspace(range_parameters['min_inclusive'],
                                                 range_parameters['max_inclusive'],
                                                 range_parameters['n_values']))
            else:
                raise ValueError('Only know how to parse a range for a non-list search space')
        else:
            raise ValueError(f'Only know how to parse list and dict non-list search spaces, not {type(v)}')

    return new_ss


def get_inner_best_model(
    feature_data_train: SliceDict,
    labels_train: Any,
    n_classes: int,
    config: Dict[str, Any]
):
    # Grab the CV setup parameters
    cv_setup = config['nested_cv_setup']

    # Configure inner HP CV split
    cv_inner = StratifiedKFold(
        n_splits=cv_setup['hp_selection_n_splits'],
        shuffle=True,
        random_state=456
    )

    # Define the model
    model = initialize_classifier(config['classifier'])

    # The search space should already be defined directly from the config
    # Almost - we need a helper function to parse ranges
    search_space = fix_search_space(config['classifier']['search_space'])

    # Define scoring
    grid_search_scoring = define_grid_search_scoring_map(
        n_classes,
        beta=cv_setup.get('f_beta', 1),
        min_recall=cv_setup.get('min_recall_threshold', 0),
        min_npv=cv_setup.get('min_npv_threshold', 0)
    )

    # Define the grid search over this search space
    search = GridSearchCV(
        model,
        search_space,
        scoring=grid_search_scoring,
        cv=cv_inner,
        **cv_setup['grid_search_setup']
    )
    # Fit the CV procedure
    result = search.fit(feature_data_train, labels_train)

    # Get the fitted model
    return result


def get_initial_classifier_indices(
    best_model,
    feature_data: SliceDict,
    classifier_name: str
) -> np.ndarray:
    if classifier_name == 'SupNNClassifier':
        # These indices align with indices in yhat_proba_outer_test
        initial_classifier_indices = best_model.get_sup_classified_indices(feature_data)
    elif classifier_name == 'ComboTreeClassifier':
        initial_classifier_indices = best_model.get_clf1_classified_indices(feature_data)
    elif classifier_name == 'SupClassifier':
        initial_classifier_indices = np.array(range(len(feature_data)))
    else:  # NeuralNetClassifier, DecisionTreeDictClassifier, RandomForestDictClassifier, XGBDictClassifier
        initial_classifier_indices = np.array([])

    return initial_classifier_indices
