"""Created on Wed Feb  8 16:45:42 2023.

@author: u0138175
"""

import torch.nn.functional as F
import random
import numpy as np
from operator import itemgetter
import pandas as pd
import torch
import torch.nn as nn
from itertools import cycle, islice
import collections
from sklearn.metrics import roc_auc_score, accuracy_score
torch.manual_seed(32)


class AdversarialAttacks:
    """class to define adversarial attacks."""

    def __init__(self, max_prefix_length, payload_values, datacreator, feature_combiner, scaler, dataset_manager):
        self.max_prefix_length = max_prefix_length
        self.payload_values = payload_values
        self.datacreator = datacreator
        self.feature_combiner = feature_combiner
        self.scaler = scaler
        self.dataset_manager = dataset_manager
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def create_adversarial_dt_named_test(self, attack_type, attack_col, dt_prefixes, classifier):
        dt_prefixes2 = dt_prefixes.copy()
        dt_prefixes_correct, y_correct, correct_caseIDs = self.correctly_predicted_prefixes(dt_prefixes2, classifier)
        caseIDs = list(dt_prefixes2['Case ID'].unique())
        caseIDs_incorrect = list(set(caseIDs)-set(correct_caseIDs))
        incorrect_dt_prefixes = dt_prefixes2[dt_prefixes2['Case ID'].isin(caseIDs_incorrect)].copy()
        adversarial_prefixes = self.create_adversarial_prefixes(attack_type, attack_col, correct_caseIDs, dt_prefixes2, dt_prefixes_correct, classifier, 'test')
        adversarial_prefixes = adversarial_prefixes.append(incorrect_dt_prefixes)
        return adversarial_prefixes

    def create_adversarial_dt_named_train(self, attack_type, attack_col, dt_prefixes, classifier):
        dt_prefixes2 = dt_prefixes.copy()
        dt_prefixes_correct, y_correct, correct_caseIDs = self.correctly_predicted_prefixes(dt_prefixes2, classifier)
        adversarial_prefixes = self.create_adversarial_prefixes(attack_type, attack_col, correct_caseIDs, dt_prefixes2, dt_prefixes_correct, classifier, 'train')
        return adversarial_prefixes

    def create_adversarial_prefixes(self, attack_type, attack_col, caseIDs, dt_prefixes, dt_prefixes_correct, classifier, traintest):
        """Create the adversarial prefixes."""
        loop_count = 0
        total_adversarial_cases = []
        adversarial_prefixes = pd.DataFrame()
        # FROM HERE, YOU HAVE TO START YOUR LOOP
        if traintest == 'train':
            total_caseIDs = list(dt_prefixes['Case ID'].unique())
        else:
            total_caseIDs = caseIDs
        print('amount of total case IDs', len(total_caseIDs))

        while len(total_adversarial_cases) < len(total_caseIDs):
            adversarial_cases = 0
            loop_count += 1
            print('loop count', loop_count)
            if loop_count < 25:
                dt_prefixes2 = dt_prefixes_correct.copy()
                # These are the adversarially attacked examples
                if attack_type == 'all_event':
                    dt_prefixes_adv = self.permute_all_event(dt_prefixes2, attack_col)
                elif attack_type == 'last_event':
                    dt_prefixes_adv = self.permute_last_event(dt_prefixes2, attack_col)
                y = self.datacreator.get_label_numeric_adversarial(dt_prefixes_adv)
                adversarial_cases = self.check_adversarial_cases(dt_prefixes_adv, y, classifier, caseIDs)
                if len(adversarial_cases) > 0:
                    total_adversarial_cases.extend(adversarial_cases)
                    created_adversarial_df = dt_prefixes_adv[dt_prefixes_adv['Case ID'].isin(adversarial_cases)].copy()
                    created_adversarial_df.loc[:, 'Case ID'] = created_adversarial_df.loc[:, 'Case ID'] + '_adv' + str(loop_count)
                    adversarial_prefixes = adversarial_prefixes.append(created_adversarial_df)
                    print('total adversarial cases after loop', loop_count, 'is:', len(total_adversarial_cases))
                else:
                    print('amount of succesful adversarial cases:', len(total_adversarial_cases), 'versus', len(total_caseIDs), 'total cases')
                    print('amount of total adversarial traces:', len(adversarial_prefixes), 'versus amount of total normal traces', len(dt_prefixes))
            else:

                print('amount of succesful adversarial cases:', len(total_adversarial_cases), 'versus', len(total_caseIDs), 'total cases')
                print('amount of total adversarial traces:', len(adversarial_prefixes), 'versus amount of total normal traces', len(dt_prefixes))
                return adversarial_prefixes
        print('amount of succesful adversarial cases:', len(total_adversarial_cases), 'versus', len(total_caseIDs), 'total cases')
        print('amount of total adversarial traces:', len(adversarial_prefixes), 'versus amount of total normal traces', len(dt_prefixes))
        print('percentage of adversarial cases:', len(total_adversarial_cases)/len(total_caseIDs))
        return adversarial_prefixes

    def permute_last_event(self, data, j):
        """Permute the last event."""
        data_copy = data.copy()
        permuted_values = set(data_copy[j].values)
        caseIDs = list(data_copy['Case ID'].unique())
        for i in caseIDs:
            max_value = data_copy.loc[(data_copy['Case ID'] == i, 'event_nr')].max()
            value = data_copy.loc[(data_copy['Case ID'] == i) & (data_copy.event_nr == max_value)][j].iloc[0]
            permuted_list = np.setdiff1d(list(permuted_values), value)
            random_value = random.choice(permuted_list)
            data_copy.loc[((data_copy['Case ID'] == i) & (data_copy.event_nr == max_value)), j] = random_value
        return data_copy

    def check_adversarial_cases(self, data, y2, classifier, caseIDs):
        """Check for adversarial cases."""
        dt_named = self.datacreator.transform_data_test(self.feature_combiner, self.scaler, data)
        y2 = self.datacreator.get_label_numeric_adversarial(data)
        pred = classifier.predict(dt_named)
        indices = self.return_indices_adversarial_guess(y2, pred)
        if len(indices) == 0:
            adversarial_cases = []
            return adversarial_cases
        adversarial_cases = list(itemgetter(*indices)(caseIDs))
        return adversarial_cases

    def correctly_predicted_prefixes(self, dt_prefixes, classifier):
        """Check which prefixes are correctly predicted."""
        # take the correctly predicted ones
        data_last = dt_prefixes.copy()
        data_last = data_last.groupby(['Case ID']).last()
        caseIDs = list(data_last.index)
        dt_named = self.datacreator.transform_data_test(self.feature_combiner, self.scaler, dt_prefixes)
        y = self.dataset_manager.get_label_numeric(dt_prefixes)
        print('amount of original cases', len(caseIDs))
        pred = classifier.predict(dt_named)
        indices = self.return_indices_correlated_guess(y, pred)
        self.correct_cases = list(itemgetter(*indices)(caseIDs))
        correct_y = list(itemgetter(*indices)(y))
        dt_prefixes2 = dt_prefixes.copy()
        dt_prefixes2.loc[(dt_prefixes2['label'] == 'deviant'), 'label'] = 1
        dt_prefixes2.loc[(dt_prefixes2['label'] == 'regular'), 'label'] = 0
        data_last = dt_prefixes2.groupby(['Case ID']).last()
        data_last = data_last.reset_index()
        dt_prefixes2 = dt_prefixes2[dt_prefixes2['Case ID'].isin(self.correct_cases)]
        y2 = data_last[data_last['Case ID'].isin(self.correct_cases)]['label']
        assert len(correct_y) == len(self.correct_cases) == len(dt_prefixes2.groupby(['Case ID']).last()) == len(y2)
        print('amount of original traces:', len(dt_prefixes))
        print('amount of correctly predicted traces:', len(dt_prefixes2))
        print('amount of correctly predicted cases:', len(self.correct_cases))
        dt_prefixes_correct = dt_prefixes2.copy()
        return dt_prefixes_correct, y2, self.correct_cases

    def return_indices_correlated_guess(self, a, b):
        """Return the indices that are the same."""
        return [i for i, v in enumerate(a) if v == b[i]]

    def return_indices_adversarial_guess(self, a, b):
        """Return the indices that are different."""
        return [i for i, v in enumerate(a) if v != b[i]]

    def permute_all_event(self, prefixes, j):
        """Permute all the events."""
        prefixes[j] = random.choices(self.payload_values[j], k=len(prefixes))
        return prefixes

    def prefix_lengths_adversarial(self, ans, x_hat_param):
        """Obtain the prefix lengths of the adversarial traces."""
        ans2 = ans.copy()
        for i in range(0, len(ans)):
            ans2[i] = torch.argmax(x_hat_param[i], dim=1)[0:len(ans[i])]
        return ans2


class AdversarialAttacks_LSTM:
    """class to define adversarial attacks."""

    def __init__(self, max_prefix_length, payload_values, datacreator, cols, cat_cols, activity_col, resource_col, no_cols_list):
        self.max_prefix_length = max_prefix_length
        self.payload_values = payload_values
        self.datacreator = datacreator
        self.cat_cols = cat_cols
        self.cols = cols
        self.activity_col = activity_col
        self.resource_col = resource_col
        self.no_cols_list = no_cols_list
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_dataset_LSTM(self, adversarial):
        adversarial = adversarial[self.cols]
        y_A = self.datacreator.get_label_numeric_adversarial(adversarial)
        # groupby case ID
        ans_act = self.datacreator.groupby_caseID(adversarial, self.cols, self.activity_col)
        ans_res = self.datacreator.groupby_caseID(adversarial, self.cols, self.resource_col)
        ans_act[0] = nn.ConstantPad1d((0, self.max_prefix_length - ans_act[0].shape[0]), 0)(ans_act[0])
        ans_res[0] = nn.ConstantPad1d((0, self.max_prefix_length - ans_res[0].shape[0]), 0)(ans_res[0])
        ######ACTIVITY AND RESOURCE COL########
        activity = self.datacreator.pad_data(ans_act).to(self.device)
        resource = self.datacreator.pad_data(ans_res).to(self.device)
        return activity, resource, y_A

    def create_adversarial_dt_named_LSTM_train(self, attack_type, attack_col, dt_prefixes, activity, resource, classifier):
        dt_prefixes = dt_prefixes.copy()
        dt_prefixes_correct, y_correct, caseIDs, correct_cases = self.correctly_predicted_prefixes_LSTM(dt_prefixes, activity, resource, classifier)
        # create adversarial train prefixes of the original instances. These can be adversarial examples of the same trace,
        # but they should have a different case ID
        adversarial_prefixes = self.create_adversarial_prefixes_LSTM(attack_type, attack_col, caseIDs, dt_prefixes, dt_prefixes_correct, y_correct, classifier, 'train')
        return adversarial_prefixes

    def create_adversarial_dt_named_LSTM_test(self, attack_type, attack_col, dt_prefixes_original, activity, resource, classifier):
        dt_prefixes = dt_prefixes_original.copy()
        dt_prefixes_correct, y_correct, caseIDs, correct_cases = self.correctly_predicted_prefixes_LSTM(dt_prefixes, activity, resource, classifier)
        # create adversarial train prefixes of the original instances. These can be adversarial examples of the same trace,
        # but they should have a different case ID
        adversarial_prefixes = self.create_adversarial_prefixes_LSTM(attack_type, attack_col, correct_cases, dt_prefixes, dt_prefixes_correct, y_correct, classifier, 'test')
        caseIDs = list(dt_prefixes['Case ID'].unique())
        caseIDs_incorrect = list(set(caseIDs)-set(correct_cases))
        incorrect_dt_prefixes = dt_prefixes[dt_prefixes['Case ID'].isin(caseIDs_incorrect)].copy()
        adversarial_prefixes_total = adversarial_prefixes.append(incorrect_dt_prefixes)
        return adversarial_prefixes_total

    def create_adversarial_prefixes_LSTM(self, attack_type, attack_col, caseIDs, dt_prefixes, dt_prefixes_correct, y_correct, classifier, traintest):
        """Create the adversarial prefixes."""
        loop_count = 0
        total_adversarial_cases = []
        adversarial_prefixes = pd.DataFrame()
        dt_prefixes_original = dt_prefixes.copy()

        # FROM HERE, YOU HAVE TO START YOUR LOOP
        if traintest == 'train':
            total_caseIDs = list(dt_prefixes['Case ID'].unique())
        else:
            total_caseIDs = caseIDs
        print('amount of Case IDs', len(total_caseIDs))
        while len(total_adversarial_cases) < len(total_caseIDs):
            adversarial_cases = 0
            loop_count += 1
            print('loop count', loop_count)
            if loop_count < 25:
                dt_prefixes_adv = dt_prefixes_correct.copy()
                # These are the adversarially attacked examples
                if attack_type == 'all_event':
                    dt_prefixes_adv = self.permute_all_event(dt_prefixes_adv, attack_col)
                elif attack_type == 'last_event':
                    dt_prefixes_adv = self.permute_last_event(dt_prefixes_adv, attack_col)
                adversarial_cases = self.check_adversarial_cases_LSTM(dt_prefixes_adv, dt_prefixes_original, y_correct, caseIDs, classifier)
                if len(adversarial_cases) > 0:
                    total_adversarial_cases.extend(adversarial_cases)
                    created_adversarial_df = dt_prefixes_adv[dt_prefixes_adv['Case ID'].isin(adversarial_cases)].copy()
                    created_adversarial_df.loc[:, 'Case ID'] = created_adversarial_df.loc[:, 'Case ID'] + '_adv' + str(loop_count)
                    adversarial_prefixes = adversarial_prefixes.append(created_adversarial_df)
                    print('total adversarial cases after loop', loop_count, 'is:', len(total_adversarial_cases))
                else:
                    print('amount of succesful adversarial cases:', len(total_adversarial_cases), 'versus', len(total_caseIDs), 'total cases')
                    print('amount of total adversarial traces:', len(adversarial_prefixes), 'versus amount of total normal traces', len(dt_prefixes))
            else:

                print('amount of succesful adversarial cases:', len(total_adversarial_cases), 'versus', len(total_caseIDs), 'total cases')
                print('amount of total adversarial traces:', len(adversarial_prefixes), 'versus amount of total normal traces', len(dt_prefixes))
                return adversarial_prefixes
        print('amount of succesful adversarial cases:', len(total_adversarial_cases), 'versus', len(total_caseIDs), 'total cases')
        print('amount of total adversarial traces:', len(adversarial_prefixes), 'versus amount of total normal traces', len(dt_prefixes))
        print('percentage of adversarial cases:', len(total_adversarial_cases)/len(total_caseIDs))
        return adversarial_prefixes

    def permute_last_event(self, data, j):
        """Permute the last event."""
        data_copy = data.copy()
        permuted_values = set(data_copy[j].values)
        caseIDs = list(data_copy['Case ID'].unique())
        for i in caseIDs:
            max_value = data_copy.loc[(data_copy['Case ID'] == i, 'event_nr')].max()
            value = data_copy.loc[(data_copy['Case ID'] == i) & (data_copy.event_nr == max_value)][j].iloc[0]
            permuted_list = np.setdiff1d(list(permuted_values), value)
            random_value = random.choice(permuted_list)
            data_copy.loc[((data_copy['Case ID'] == i) & (data_copy.event_nr == max_value)), j] = random_value
        return data_copy

    def check_adversarial_cases_LSTM(self, data_adv, dt_prefixes_original, y2, correct_cases, classifier):
        """Check for adversarial cases."""
        classifier = classifier.to(self.device)
        dt_prefixes2 = dt_prefixes_original.copy()
        dt_prefixes2 = dt_prefixes2[self.cols]
        activity, resource, y_A1_act = self.make_dataset_LSTM(data_adv)
        batch_size = activity.shape[0]
        states = classifier.init_hidden(batch_size)
        classifier.eval()
        with torch.no_grad():
            pred = classifier(activity, resource, states)
            pred = (np.where(np.array(pred.cpu().flatten()) > 0.50, 1, 0))
        indices = self.return_indices_adversarial_guess(y2, pred)
        if len(indices) == 0:
            adversarial_cases = []
            return adversarial_cases
        adversarial_cases = list(itemgetter(*indices)(correct_cases))
        return adversarial_cases

    def correctly_predicted_prefixes_LSTM(self, dt_prefixes, activity_train, resource_train, classifier):
        """Check which prefixes are correctly predicted."""
        # take the correctly predicted ones
        data_last = dt_prefixes.copy()
        y = self.datacreator.get_label_numeric_adversarial(data_last)
        data_last = data_last.groupby(['Case ID']).last()
        caseIDs = list(data_last.index)
        print('amount of original cases', len(caseIDs))
        batch_size = activity_train.shape[0]
        states = classifier.init_hidden(batch_size)
        classifier.eval()
        with torch.no_grad():
            pred = classifier(activity_train, resource_train, states).cpu().numpy()
        pred = (np.where(np.array(pred.flatten()) > 0.50, 1, 0))
        indices = self.return_indices_correlated_guess(y, pred)
        correct_cases = list(itemgetter(*indices)(caseIDs))
        correct_y = list(itemgetter(*indices)(y))
        dt_prefixes2 = dt_prefixes.copy()
        dt_prefixes2.loc[(dt_prefixes2['label'] == 'deviant'), 'label'] = 1
        dt_prefixes2.loc[(dt_prefixes2['label'] == 'regular'), 'label'] = 0
        data_last = dt_prefixes2.groupby(['Case ID']).last()
        data_last = data_last.reset_index()
        dt_prefixes2 = dt_prefixes2[dt_prefixes2['Case ID'].isin(correct_cases)]
        y2 = data_last[data_last['Case ID'].isin(correct_cases)]['label']
        assert len(correct_y) == len(correct_cases) == len(dt_prefixes2.groupby(['Case ID']).last()) == len(y2)
        print('amount of original traces:', len(dt_prefixes))
        print('amount of correctly predicted traces:', len(dt_prefixes2))
        print('amount of correctly predicted cases:', len(correct_cases))
        dt_prefixes_correct = dt_prefixes2.copy()
        return dt_prefixes_correct, y2, caseIDs, correct_cases

    def return_indices_correlated_guess(self, a, b):
        """Return the indices that are the same."""
        return [i for i, v in enumerate(a) if v == b[i]]

    def return_indices_adversarial_guess(self, a, b):
        """Return the indices that are different."""
        return [i for i, v in enumerate(a) if v != b[i]]

    def permute_all_event(self, prefixes, j):
        """Permute all the events."""
        prefixes[j] = random.choices(self.payload_values[j], k=len(prefixes))
        return prefixes

    def prefix_lengths_adversarial(self, ans, x_hat_param):
        """Obtain the prefix lengths of the adversarial traces."""
        ans2 = ans.copy()
        for i in range(0, len(ans)):
            ans2[i] = torch.argmax(x_hat_param[i], dim=1)[0:len(ans[i])]
        return ans2
