from typing import Dict, Any

import numpy as np
import torch

from sklearn.metrics import (
    precision_score,
    make_scorer,
    balanced_accuracy_score,
    accuracy_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    confusion_matrix
    )


def define_grid_search_scoring_map(
    n_classes: int,
    beta: float = 1.0,
    min_recall: float = 0.0,
    min_npv: float = 0.0,
) -> Dict[str, Any]:
    '''
    Returns a scoring dictionary that can be passed to GridSearchCV like
    https://scikit-learn.org/stable/modules/model_evaluation.html#:~:text=As%20a%20dict%20mapping%20the%20scorer%20name%20to%20the%20scoring%20function
    '''

    scoring_map = {
        'Accuracy':'accuracy',
        'Balanced Accuracy':'balanced_accuracy',
    }
    if n_classes == 2:
        scoring_map['Not Pass Precision'] = 'precision'
        scoring_map['Not Pass Recall'] = 'recall'
        scoring_map['Not Pass Precision (Min Recall)'] = make_scorer(precision_with_min_recall,
                                                                     min_recall=min_recall)
        scoring_map['Not Pass Precision (Min NPV)'] = make_scorer(precision_with_min_npv,
                                                                  min_npv=min_npv)
        scoring_map['Not Pass Precision (Min NPV), -5'] = make_scorer(precision_with_min_npv,
                                                                      min_npv=min_npv, penalty=-5)
        scoring_map['Pass Precision'] = make_scorer(precision_score, pos_label=0)
        scoring_map['Pass Recall'] = make_scorer(recall_score, pos_label=0)
    else:
        scoring_map['Precision (macro avg)'] = make_scorer(precision_score, average='macro')
        scoring_map['Retest/Fail Precision (macro avg)'] = make_scorer(precision_score, labels=[1,2], average='macro')
        scoring_map['Pass Precision'] = make_scorer(precision_score, labels=[0], average='macro')
        scoring_map['Retest Precision'] = make_scorer(precision_score, labels=[1], average='macro')
        scoring_map['Fail Precision'] = make_scorer(precision_score, labels=[2], average='macro')
        scoring_map['Recall (macro avg)'] = make_scorer(recall_score, average='macro')
        scoring_map['Retest/Fail Recall (macro avg)'] = make_scorer(recall_score, labels=[1,2], average='macro')
        scoring_map['Pass Recall'] = make_scorer(precision_score, labels=[0], average='macro')
        scoring_map['Retest Recall'] = make_scorer(recall_score, labels=[1], average='macro')
        scoring_map['Fail Recall'] = make_scorer(recall_score, labels=[2], average='macro')

    return scoring_map


def calculate_test_set_metrics(y: torch.Tensor,
                               yhat_prob: np.ndarray,
                               yhat_class: np.ndarray,
                               n_classes: int,
                               beta: float = 1.0) -> Dict[str, float]:

    test_metrics = {
        'Accuracy': accuracy_score(y, yhat_class),
        'Balanced Accuracy': balanced_accuracy_score(y, yhat_class)
    }
    if n_classes == 2:
        test_metrics['AUC'] = roc_auc_score(y, yhat_prob[:, 1])
        test_metrics['F1 Score'] = f1_score(y, yhat_class)
        test_metrics['Fbeta Score'] = fbeta_score(y, yhat_class, beta=beta)
        test_metrics['Pass Precision'] = precision_score(y, yhat_class, pos_label=0)
        test_metrics['Not Pass Precision'] = precision_score(y, yhat_class, pos_label=1)
        test_metrics['Pass Recall'] = recall_score(y, yhat_class, pos_label=0)
        test_metrics['Not Pass Recall'] = recall_score(y, yhat_class, pos_label=1)
        (
            test_metrics['TN'], test_metrics['FP'], test_metrics['FN'], test_metrics['TP']
        ) = confusion_matrix(y, yhat_class).ravel()
    else:
        test_metrics['AUC (OvR, macro)'] = roc_auc_score(y, yhat_prob, multi_class='ovr', average='macro')
        test_metrics['Retest/Fail F1 Score (macro avg)'] = f1_score(y, yhat_class, labels=[1, 2], average='macro')
        test_metrics['Retest/Fail Fbeta Score (macro avg)'] = fbeta_score(y, yhat_class, beta=beta, labels=[1, 2], average='macro')
        test_metrics['Precision (macro avg)'] = precision_score(y, yhat_class, average='macro')
        test_metrics['Retest/Fail Precision (macro avg)'] = precision_score(y, yhat_class, labels=[1, 2], average='macro')
        test_metrics['Pass Precision'] = precision_score(y, yhat_class, labels=[0], average='macro')
        test_metrics['Retest Precision'] = precision_score(y, yhat_class, labels=[1], average='macro')
        test_metrics['Fail Precision'] = precision_score(y, yhat_class, labels=[2], average='macro')
        test_metrics['Recall (macro avg)'] = recall_score(y, yhat_class, average='macro')
        test_metrics['Retest/Fail Recall (macro avg)'] = recall_score(y, yhat_class, labels=[1, 2], average='macro')
        test_metrics['Retest Recall'] = recall_score(y, yhat_class, labels=[1], average='macro')
        test_metrics['Fail Recall'] = recall_score(y, yhat_class, labels=[2], average='macro')

    return test_metrics


def precision_with_min_recall(y_true: Any, y_pred: Any, min_recall: float = 0.0, **kwargs) -> float:
    '''
    Returns the precision score, unless the recall is below some threshold
    Then returns 0
    '''
    if recall_score(y_true, y_pred, **kwargs) > min_recall:
        return precision_score(y_true, y_pred, **kwargs)
    else:
        return 0.0


def precision_with_min_npv(y_true: Any, y_pred: Any, min_npv: float = 0.0, penalty: float = 0, **kwargs) -> float:
    '''
    Returns the precision score, unless the NPV is below some threshold
    Then returns 0
    '''
    if precision_score(y_true, y_pred, pos_label=0, **kwargs) > min_npv:
        return precision_score(y_true, y_pred, pos_label=1, **kwargs)
    else:
        return penalty
