import pandas as pd
import numpy as np
from datetime import datetime


class GreedyStepDown:
    def __init__(self, features=None, labels=None, max_fp=0):
        '''
        Parameters
        ----------
        features: pd.DataFrame
            Index: run_id
            Rows: unique run_ids
            Columns: features

        labels: pd.Series
            Index: run_id, same order as features
            Value: label, in integer form
        '''

        self.max_fp = max_fp  # max_fp is the false positive budget
        self.features = features.columns  # [f for f in df.columns if f not in ['run_id', 'label']]
        self.n_features = len(self.features)  # total number of features

        # Initialize the array that holds labels and features for all predicted passes and false positives
        # Get a vector for all predicted pass labels
        # Anything STRICTLY LESS THAN threshold is passing
        self.pred_pass = np.concatenate(
            [np.reshape(labels.values, (-1, 1)), features.values],
            axis=1
        )  # df.drop('run_id', axis=1).to_numpy()
        self.pred_pass_run_ids = features.index.to_numpy()  # df['run_id'].to_numpy()
        self.negative_run_ids = labels.index[(labels == 1) | (labels == 2)].to_numpy()  # df.loc[(df['label'] == 1) | (df['label'] == 2), 'run_id'].to_numpy()
        self.initial_negative_run_ids = self.negative_run_ids.copy()
        self.pred_fps = self.pred_pass[self.pred_pass[:, 0] > 0, :]
        self.fp_run_ids = self.pred_pass_run_ids[self.pred_pass[:, 0] > 0]
        self.pred_pass_labels = self.pred_pass[:, 0]

        # Initialize thresholds with max + 1 for each feature
        self.sups = [self.pred_pass[:, i].max() + 1 for i in range(1, self.n_features + 1)]

        # Set target TN, which is just negatives - allowed false positives
        self.N = np.sum(labels > 0)
        self.P = np.sum(labels == 0)
        self.target_tn = self.N - self.max_fp

        # Fill in the initial confusion matrix
        self.cm = {
            'tp': sum(self.pred_pass_labels == 0),
            'fp': sum(self.pred_pass_labels > 0)
        }
        self.cm['tn'] = self.N - self.cm['fp']
        self.cm['fn'] = self.P - self.cm['tp']

        # History dataframes
        self.cm_history = pd.DataFrame(self.cm, index=[0])
        self.feature_history = pd.DataFrame({'chosen_feature_idx': np.nan}, index=[0])
        self.sup_history = pd.DataFrame({i: self.sups[i] for i in range(len(self.sups))}, index=[0])

    def optimize(self, verbose=False):
        return self.optimize_thresholds(verbose)

    def optimize_thresholds(self, verbose):
        start_time = datetime.now()  # Timing log

        # If there is a tie in the best feature to decrease, the tiebreaker rotates
        self.tiebreaker = list(range(self.n_features))
        self.counter = 0  # Iteration counter

        # While you are above the false positive budget...
        while self.cm['tn'] < self.target_tn:
            # ...reduce some threshold and increment the step counter
            self.step_down()
            if verbose:
                print(f"Step: {self.counter+1}")
                print(f"CM: {self.cm}")
                print(f"Target TN: {self.target_tn}")
                print(f"")

            self.counter += 1

        end_time = datetime.now()

        # Create metadata dictionary
        metadata = {'fit_time': str(end_time - start_time),
                    'cm_history': self.cm_history,
                    'feature_history': self.feature_history,
                    'sup_history': self.sup_history
                    }
        return self.sups, metadata

    def step_down(self):
        # Each iteration MUST reduce the number of false positives by at least 1
        # Equivalently, increase the number of true negatives by at least 1

        # Set up the proposal dict that contains proposed new thresholds
        # And eventually CM data after update proposal
        proposed = {}
        for i in range(self.n_features):
            proposed[i] = {}
            proposed[i]['sup'] = self.sups[i]
        for f_idx in range(self.n_features):
            proposed = self.update_proposal(f_idx, proposed)

        # Check to see if it is possible to increase TNs for free
        # if you can increase TNs without increasing FNs, that is always the correct move
        zero_fn_idxs = [f_idx for f_idx in range(self.n_features) if proposed[f_idx]['inc_fn'] == 0]
        if len(zero_fn_idxs) > 0:
            max_inc_tn = max([proposed[i]['inc_tn'] for i in zero_fn_idxs]) # pick the option with the most TN increase
            best_idxs = [f_idx for f_idx in zero_fn_idxs if proposed[f_idx]['inc_tn'] == max_inc_tn] # randomize between them
            #print(f"There are features that increase TN with zero FN: {zero_fn_idxs}")
            if len(best_idxs) == 1:
                step_idx = best_idxs[0]
            else:
                step_idx = self.break_tie(best_idxs)

        # Once you get here, you know that every proposal increases BOTH TNs and FNs
        else:
            # So you can take the option with the best ratio
            inc_fn_tn = [proposed[i]['inc_fn']/proposed[i]['inc_tn'] for i in range(self.n_features)]
            min_inc_fn_tn = min(inc_fn_tn)
            best_idxs = [f_idx for f_idx in range(self.n_features) if inc_fn_tn[f_idx] == min_inc_fn_tn]
            if len(best_idxs) == 1:
                step_idx = best_idxs[0]
            else:
                step_idx = self.break_tie(best_idxs)

        # Update the sup that has changed
        # THen update the CM data
        self.sups[step_idx] = proposed[step_idx]['sup']
        self.pred_pass_run_ids = self.pred_pass_run_ids[np.all(self.pred_pass[:, 1:] < np.array(self.sups), axis=1)]
        self.pred_pass = self.pred_pass[np.all(self.pred_pass[:, 1:] < np.array(self.sups), axis=1)]  # fast_filter(np.array(self.sups), self.pred_pass)
        self.pred_fps = self.pred_pass[self.pred_pass[:, 0] > 0, :]
        self.fp_run_ids = self.pred_pass_run_ids[self.pred_pass[:, 0] > 0]
        self.pred_pass_labels = self.pred_pass[:, 0]
        self.cm = {'tp': sum(self.pred_pass_labels == 0), 'fp': sum(self.pred_pass_labels > 0)}
        self.cm['tn'] = self.N - self.cm['fp']
        self.cm['fn'] = self.P - self.cm['tp']

        # Add history
        self.cm_history = self.cm_history.append(self.cm, ignore_index=True)
        self.feature_history = self.feature_history.append({'chosen_feature_idx':step_idx}, ignore_index=True)
        self.sup_history = self.sup_history.append({i: self.sups[i] for i in range(len(self.sups))}, ignore_index=True)

    def update_proposal(self, f_idx, proposed):
        # The sup for feature f_idx is the maximum remaining false positive feature value
        proposed[f_idx]['sup'] = self.pred_fps[:, f_idx+1].max()

        # This list is all of the proposed sups, which are the prior sups except for feature f_idx
        prop_sups = [self.sups[i] if i != f_idx else proposed[i]['sup'] for i in proposed]

        # Easy way to figure out which modules still pass
        prop_pred_pass = self.pred_pass[np.all(self.pred_pass[:, 1:] < prop_sups, axis=1)]

        # Calculate the confusion matrix stats
        proposed[f_idx]['tn'] = self.N - np.sum(prop_pred_pass[:, 0] > 0)
        proposed[f_idx]['fn'] = self.P - np.sum(prop_pred_pass[:, 0] == 0)

        # Gives the incremental true negatives and false negatives
        proposed[f_idx]['inc_tn'] = proposed[f_idx]['tn'] - self.cm['tn']
        proposed[f_idx]['inc_fn'] = proposed[f_idx]['fn'] - self.cm['fn']

        return proposed

    def break_tie(self, candidates):
        smallest_cand_index = np.inf
        for candidate in candidates:
            if self.tiebreaker.index(candidate) < smallest_cand_index:
                smallest_cand_index = self.tiebreaker.index(candidate)
        chosen = self.tiebreaker[smallest_cand_index]
        self.tiebreaker.remove(chosen)
        self.tiebreaker.append(chosen)
        return chosen
