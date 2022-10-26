# Need to take in data, split into train/test
#  For training, fit the StandardScalar. Then, transform.
#  Then, create tensors, packing and padding as necessary. Then package into SliceDict.
#  For testing, transform with the fit StandardScalar.
#  Then create tensors, packing and padding and package into SliceDict.
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any
import torch
from sklearn.metrics import precision_recall_curve

from .data_prep import get_outer_split_data, save_outer_split_data, configure_outer_split
from .param_validation import validate_nested_cv_config
from .metrics import calculate_test_set_metrics
from .utils import initialize_classifier, get_inner_best_model, get_initial_classifier_indices


def run_nested_cv(
    labels: torch.Tensor,
    feature_data: Dict[str, torch.Tensor],
    n_obs: int,
    config: Dict[str, Any],
    load_models: bool = False,
    save_best_refit_inner_model: bool = True,
    save_cv_results: bool = True,
    save_data_only: bool = False
):
    '''
    Parameters
    ----------
    config: dictionary
        Parameters that define the experiment. This should combine the data thaw
        And the model configuration parameters.

    Returns
    -------
    '''
    validate_nested_cv_config(config,
                              load_models,
                              save_best_refit_inner_model,
                              save_cv_results,
                              save_data_only)

    # Grab the CV setup parameters
    cv_setup = config['nested_cv_setup']

    # Configure outer model selection CV split
    # Could be either model selection, or randomized
    # cv_outer: sklearn cross validation object
    # split_index_range: list of split ids
    cv_outer, split_index_range = configure_outer_split(cv_setup)

    (outer_results, outer_individual_probas,
     outer_pr_curves, inner_cv_results, inner_best_index) = [], [], [], [], {}

    # Determine if the neural net training history should be saved
    if save_cv_results:
        if config['classifier']['name'] in ['SupNNClassifier', 'NeuralNetClassifier']:
            save_nn_history = True
            best_model_history_list = list()
        else:
            save_nn_history = False
    else:
        save_nn_history = False

    # Recover the number of classes
    n_classes = len(labels.unique())

    # Outer loop for testing
    for split_id, (train_ix, test_ix) in zip(split_index_range, cv_outer.split(np.zeros(n_obs), labels)):
        # Split into train/test sets
        (feature_data_train,
         labels_train,
         feature_data_test,
         labels_test) = get_outer_split_data(feature_data, labels, train_ix, test_ix, config)

        # Save data if necessary
        if save_data_only:
            # This branch saves the outer split data
            save_outer_split_data(
                config,
                split_id,
                len(split_index_range),
                labels_train,
                labels_test,
                feature_data_train,
                feature_data_test
            )
        else:
            skip_inner_hp_selection = config['nested_cv_setup'].get('hp_selection_n_splits') is None
            if skip_inner_hp_selection:
                # This means that there is no HP selection - just train the model on the training data
                best_model = initialize_classifier(config['classifier'])
                best_model.fit(feature_data_train, labels_train)
            else:
                # This is all setup that is necessary to train the models
                # Not necessary if loaded from file
                if not load_models:
                    # Get best model from the inner grid search
                    result = get_inner_best_model(feature_data_train,
                                                  labels_train,
                                                  n_classes,
                                                  config)
                    best_model = result.best_estimator_

                    # Get the CV results and the index of the best model
                    # cv_results_ is a dictionary,
                    # see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#:~:text=cv_results_dict%20of%20numpy%20(masked)%20ndarrays
                    inner_cv_results_item = pd.DataFrame(result.cv_results_)
                    inner_cv_results_item['outer_split_id'] = split_id
                    inner_cv_results.append(inner_cv_results_item)

                    inner_best_index[split_id] = result.best_index_
                else:
                    load_model_dirpath = cv_setup['models_save_dirpath']
                    load_model_fname = f'best_model_outer_split_{split_id}.pkl'
                    best_model = pickle.load(open(f"{load_model_dirpath}/{load_model_fname}", 'rb'))

            # Save the best model, optionally
            if save_best_refit_inner_model:
                save_dirpath = cv_setup['models_save_dirpath']
                save_fname = f'best_model_outer_split_{split_id}.pkl'
                pickle.dump(best_model, open(f"{save_dirpath}/{save_fname}", 'wb'))

            # Evaluate model on the test set from outer loop
            #print(f"in the nested_cv loop, feature_data_test is {feature_data_test}")
            yhat_class_outer_test = best_model.predict(feature_data_test)
            yhat_proba_outer_test = best_model.predict_proba(feature_data_test)

            # If the model is a SupNNClassifier, then grab the indices where sup makes the final call
            sup_classified_indices_test = get_initial_classifier_indices(best_model,
                                                                         feature_data_test,
                                                                         config['classifier']['name']
                                                                         )

            # Evaluate the model
            # test_set_metrics is a dictionary of metrics for this outer split
            test_set_metrics = calculate_test_set_metrics(labels_test,
                                                          yhat_proba_outer_test,
                                                          yhat_class_outer_test,
                                                          n_classes,
                                                          beta=cv_setup.get('f_beta', 1.0))

            # Convert the dictionary into a single dataframe row
            test_set_metrics = pd.DataFrame(test_set_metrics, index=[split_id])

            # Store the results in a list of dataframes
            outer_results.append(test_set_metrics)

            # Concatenate the outer loop results
            outer_results_df = pd.concat(outer_results, axis=0)

            # Store outer loop test set predictions
            yhat_proba_outer_test_df = pd.DataFrame(yhat_proba_outer_test)
            yhat_proba_outer_test_df['outer_split_id'] = split_id
            yhat_proba_outer_test_df['y'] = labels_test.numpy()
            yhat_proba_outer_test_df['sup_classified'] = 0
            yhat_proba_outer_test_df.loc[sup_classified_indices_test, 'sup_classified'] = 1
            yhat_proba_outer_test_df.index = test_ix
            outer_individual_probas.append(yhat_proba_outer_test_df)
            outer_individual_probas_df = pd.concat(outer_individual_probas, axis=0)

            # Store precision-recall scores to generate curves if binary
            if n_classes == 2:
                precision, recall, thresholds = precision_recall_curve(labels_test,
                                                                       yhat_proba_outer_test[:, 1],
                                                                       pos_label=1)
                pr_curve = pd.DataFrame({'precision': precision,
                                         'recall': recall,
                                         'thresholds': np.append(thresholds, np.inf)})
                pr_curve['outer_split_id'] = split_id
                outer_pr_curves.append(pr_curve)
                outer_pr_curves_df = pd.concat(outer_pr_curves, axis=0)
            else:
                outer_pr_curves.append(pd.DataFrame({'precision': [], 'recall': [], 'thresholds': []}))
                outer_pr_curves_df = pd.concat(outer_pr_curves, axis=0)

            # Compile the inner loop results and save training history
            if not load_models:
                if not skip_inner_hp_selection:
                    inner_best_index_series = pd.Series(inner_best_index, name='inner_best_index')
                    inner_results_df = pd.concat(inner_cv_results, axis=0)
                else:
                    inner_best_index_series = pd.Series()
                    inner_results_df = pd.DataFrame()

                # Store the training history for the best estimator
                if save_nn_history:
                    nn_history_cols = ['epoch', 'dur', 'train_loss']
                    nn = best_model
                    nn_history = nn.history[:, tuple(nn_history_cols)]
                    nn_history = pd.DataFrame(nn_history, columns=nn_history_cols)
                    nn_history['outer_split'] = split_id
                    best_model_history_list.append(nn_history)
                    best_model_history_df = pd.concat(best_model_history_list, axis=0)
                else:
                    best_model_history_df = pd.DataFrame()

            else:
                inner_best_index_series = pd.Series()
                inner_results_df = pd.DataFrame()

            # Save the CV outer results
            if save_cv_results:
                results_save_dirpath = cv_setup['results_save_dirpath']
                outer_results_df.to_csv(f"{results_save_dirpath}/outer_split_results.csv")
                outer_individual_probas_df.to_csv(f"{results_save_dirpath}/outer_individual_probas.csv")
                outer_pr_curves_df.to_csv(f"{results_save_dirpath}/outer_pr_curves.csv")

                # Save the sup metadata if necessary
                if config['classifier']['name'] == 'SupNNClassifier':
                    best_model.sup_clf.save_fit_metadata(f"{results_save_dirpath}/sup_metadata/split_{split_id}")
                elif config['classifier']['name'] == 'SupClassifier':
                    best_model.save_fit_metadata(f"{results_save_dirpath}/sup_metadata/split_{split_id}")
                elif (config['classifier']['name'] == 'ComboTreeClassifier' and
                       config['classifier']['initial_params']['clf1_type'] == 'SupClassifier'):
                    best_model.clf1.save_fit_metadata(f"{results_save_dirpath}/sup_metadata/split_{split_id}")

                if not load_models:
                    inner_best_index_series.to_csv(f"{results_save_dirpath}/best_inner_index.csv")
                    inner_results_df.to_csv(f"{results_save_dirpath}/inner_grid_search_results.csv")

                    if save_nn_history:
                        best_model_history_df.to_csv(f"{results_save_dirpath}/best_model_history.csv", index=False)

    if save_data_only:
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.DataFrame()
    else:
        return outer_results_df, inner_best_index_series, inner_results_df, best_model_history_df
