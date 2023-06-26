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
from sklearn.metrics import accuracy_score
torch.manual_seed(32)
seed = 42
random.seed(seed)

class AdversarialAttacks_LSTM:
    """class to define adversarial attacks."""

    def __init__(self, train, train_prefixes, test_prefixes, max_prefix_length, payload_values, datacreator, cols, cat_cols, activity_col, resource_col, no_cols_list):
        self.max_prefix_length = max_prefix_length
        self.payload_values = payload_values
        self.datacreator = datacreator
        self.cat_cols = cat_cols
        self.cols = cols
        self.activity_col = activity_col
        self.resource_col = resource_col
        self.no_cols_list = no_cols_list
        self.train = train
        self.train_prefixes = train_prefixes
        self.test_prefixes = test_prefixes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def create_adversarial_dt_named_train(self, attack_type, attack_col, classifier):
        print('create adversarial train prefixes')
        dt_prefixes_correct, y_correct, caseIDs, correct_cases = self.correctly_predicted_prefixes(self.train_prefixes, classifier)
        # create adversarial train prefixes of the original instances. These can be adversarial examples of the same trace,
        # but they should have a different case ID
        adversarial_prefixes = self.create_adversarial_prefixes(attack_type, attack_col, correct_cases, self.train_prefixes, dt_prefixes_correct, y_correct, classifier, 'train')
        return adversarial_prefixes, self.threshold

    def create_adversarial_dt_named_test(self, attack_type, attack_col, classifier):
        print('create adversarial test prefixes')
        dt_prefixes_correct, y_correct, caseIDs, correct_cases = self.correctly_predicted_prefixes(self.test_prefixes, classifier)
        # create adversarial train prefixes of the original instances. These can be adversarial examples of the same trace,
        # but they should have a different case ID
        adversarial_prefixes = self.create_adversarial_prefixes(attack_type, attack_col, correct_cases, self.test_prefixes, dt_prefixes_correct, y_correct, classifier, 'test')
        caseIDs = list(self.test_prefixes['Case ID'].unique())
        caseIDs_incorrect = list(set(caseIDs)-set(correct_cases))
        incorrect_dt_prefixes = self.test_prefixes[self.test_prefixes['Case ID'].isin(caseIDs_incorrect)].copy()
        adversarial_prefixes_total = pd.concat([adversarial_prefixes, incorrect_dt_prefixes])
        return adversarial_prefixes_total
    
    def create_adversarial_prefixes(self, attack_type, attack_col, caseIDs, dt_prefixes, dt_prefixes_correct, y_correct, classifier, traintest):
        """Create the adversarial prefixes."""
        loop_count = 0
        total_adversarial_cases = []
        adversarial_prefixes = pd.DataFrame()
        if traintest == 'train':
            total_caseIDs = list(dt_prefixes['Case ID'].unique())
        elif traintest =='test':
            total_caseIDs = caseIDs
        print('amount of total case IDs', len(total_caseIDs))

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
                y = self.datacreator.get_label_numeric_adversarial(dt_prefixes_adv)
                adversarial_cases = self.check_adversarial_cases(dt_prefixes_adv, classifier)
                
                if len(adversarial_cases) > 0:
                    total_adversarial_cases.extend(adversarial_cases)
                    created_adversarial_df = dt_prefixes_adv[dt_prefixes_adv['Case ID'].isin(adversarial_cases)].copy()
                    created_adversarial_df.loc[:, 'Case ID'] = created_adversarial_df.loc[:, 'Case ID'] + '_adv' + str(loop_count)
                    adversarial_prefixes= pd.concat([adversarial_prefixes, created_adversarial_df])
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

    def check_adversarial_cases(self, data_adv, classifier):
        """Check for adversarial cases."""
        classifier = classifier.to(self.device)
        #dt_prefixes2 = dt_prefixes_original.copy()
        #dt_prefixes2 = dt_prefixes2[self.cols]
        activity, resource, y_A, cases_adv = self.datacreator.groupby_pad(data_adv, self.cols, self.activity_col, self.resource_col)
        activity = activity.to(self.device)
        resource = resource.to(self.device)
        #batch_size = activity.shape[0]
        classifier.eval()
        pred = classifier(activity, resource).cpu().detach().numpy()
        pred = (np.where(np.array(pred.flatten()) > self.threshold , 1, 0))
        indices = self.return_indices_adversarial_guess(y_A, pred)
        if len(indices) == 0:
            adversarial_cases = []
            return adversarial_cases
        adversarial_cases = list(itemgetter(*indices)(cases_adv))
        dt_prefixes2 = data_adv[data_adv['Case ID'].isin(adversarial_cases)]
        activity, resource, y_A, cases_adv = self.datacreator.groupby_pad(dt_prefixes2, self.cols, self.activity_col, self.resource_col)
        activity = activity.to(self.device)
        resource = resource.to(self.device)
        #batch_size = activity.shape[0]
        classifier.eval()
        pred = classifier(activity, resource).cpu().detach().numpy()
        pred = classifier(activity, resource).cpu().detach().numpy()
        pred = (np.where(np.array(pred.flatten()) > self.threshold , 1, 0))
        print('adversarial predicted', accuracy_score(y_A , pred)) 

        return adversarial_cases
    
    def correctly_predicted_prefixes(self, dt_prefixes, classifier):
        print('correctly predicted')
        """Check which prefixes are correctly predicted."""
        #groupby case ID and pad to the max prefix length for the column activity and resource
        activity_train, resource_train, activity_label, train_cases = self.datacreator.groupby_pad(dt_prefixes, self.cols, self.activity_col, self.resource_col)
        activity_train = activity_train.to(self.device)
        resource_train = resource_train.to(self.device)
        classifier = classifier.to(self.device)
        print('amount of original cases', len(train_cases))
        
        classifier.eval()
        pred = classifier(activity_train, resource_train).cpu().detach().numpy()
        self.threshold = self.datacreator.Find_Optimal_Cutoff(activity_label, pred.flatten())
        pred = (np.where(np.array(pred.flatten()) > self.threshold , 1, 0))   
        indices = self.return_indices_correlated_guess(activity_label, pred)
        correct_cases = list(itemgetter(*indices)(train_cases))
        correct_pred = list(itemgetter(*indices)(pred))
        correct_y = list(itemgetter(*indices)(activity_label))
        print('correctly predicted accuracy', accuracy_score(correct_y , correct_pred)) 
        dt_prefixes2 = dt_prefixes.copy()
        dt_prefixes2 = dt_prefixes2[dt_prefixes2['Case ID'].isin(correct_cases)]
       
        #assert len(correct_y) == len(correct_cases) == len(dt_prefixes2.groupby(['Case ID']).last())
        print('amount of original traces:', len(dt_prefixes))
        print('amount of correctly predicted traces:', len(dt_prefixes2))
        print('amount of correctly predicted cases:', len(correct_cases))
       
        activity_correct, resource_correct, y_correct, cases_correct = self.datacreator.groupby_pad(dt_prefixes2, self.cols, self.activity_col, self.resource_col)
        #activity_correct = torch.tensor([[3, 2],[5, 0]])
        #resource_correct = torch.tensor([[3, 2],[1, 0]])
        activity_correct = activity_correct.to(self.device)
        resource_correct = resource_correct.to(self.device)
        assert cases_correct == correct_cases

        #batch_size = activity.shape[0]
        classifier.eval()

        pred = classifier(activity_correct, resource_correct).cpu().detach().numpy()
        print('threshold', self.threshold)
        pred = (np.where(np.array(pred.flatten()) > self.threshold , 1, 0))
        print('correctly predicted accuracy 2', accuracy_score(pred , pred)) 
        print('correctly predicted accuracy 3', accuracy_score(y_correct , pred)) 
        assert  accuracy_score(pred , y_correct) ==1.0
        return dt_prefixes2, y_correct, train_cases, correct_cases

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
        for i in range(0, len(ans2)):
            ans2[i] = torch.argmax(x_hat_param[i], dim=1)[0:len(ans2[i])]
        return ans2

class AdversarialAttacks(AdversarialAttacks_LSTM):
    """class to define adversarial attacks."""

    def __init__(self, train, train_prefixes, test_prefixes, max_prefix_length, payload_values, datacreator, cols, cat_cols, activity_col, resource_col, no_cols_list, feature_combiner, scaler, dataset_manager):
        super().__init__(train, train_prefixes, test_prefixes, max_prefix_length, payload_values, datacreator, cols, cat_cols, activity_col, resource_col, no_cols_list)
        self.feature_combiner = feature_combiner
        self.scaler = scaler
        self.dataset_manager = dataset_manager
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def create_adversarial_dt_named_test(self, attack_type, attack_col, classifier):
        dt_prefixes_correct, y_correct, correct_caseIDs = self.correctly_predicted_prefixes(self.test_prefixes, classifier)
        caseIDs = list(self.test_prefixes['Case ID'].unique())
        caseIDs_incorrect = list(set(caseIDs)-set(correct_caseIDs))
        incorrect_dt_prefixes = self.test_prefixes[self.test_prefixes['Case ID'].isin(caseIDs_incorrect)].copy()
        adversarial_prefixes = self.create_adversarial_prefixes(attack_type, attack_col, correct_caseIDs, self.test_prefixes, dt_prefixes_correct, classifier, 'test')
        adversarial_prefixes_concat = pd.concat([adversarial_prefixes, incorrect_dt_prefixes])
        return adversarial_prefixes_concat

    def create_adversarial_dt_named_train(self, attack_type, attack_col, classifier):
        dt_prefixes_correct, y_correct, correct_caseIDs = self.correctly_predicted_prefixes(self.train_prefixes, classifier)
        adversarial_prefixes = self.create_adversarial_prefixes(attack_type, attack_col, correct_caseIDs, self.train_prefixes, dt_prefixes_correct, classifier, 'train')
        return adversarial_prefixes, self.threshold

    def create_adversarial_prefixes(self, attack_type, attack_col, caseIDs, dt_prefixes, dt_prefixes_correct, classifier, traintest):
        """Create the adversarial prefixes."""
        loop_count = 0
        total_adversarial_cases = []
        adversarial_prefixes = pd.DataFrame()
        # FROM HERE, YOU HAVE TO START YOUR LOOP
        if traintest == 'train':
            total_caseIDs = list(dt_prefixes['Case ID'].unique())
        elif traintest =="test":
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
                    adversarial_prefixes = pd.concat([adversarial_prefixes, created_adversarial_df])
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

    def check_adversarial_cases(self, data, y2, classifier, caseIDs):
        """Check for adversarial cases."""
        dt_named = self.datacreator.transform_data_test(self.feature_combiner, self.scaler, data)
        y2 = self.datacreator.get_label_numeric_adversarial(data)
        pred = classifier.predict_proba(dt_named)[:,-1]
        pred = (np.where(np.array(pred) > self.threshold , 1, 0))
        indices = self.return_indices_adversarial_guess(y2, pred)
        if len(indices) == 0:
            adversarial_cases = []
            return adversarial_cases
        adversarial_cases = list(itemgetter(*indices)(caseIDs))
        return adversarial_cases

    def correctly_predicted_prefixes(self, dt_prefixes, classifier):
        """Check which prefixes are correctly predicted."""
        # take the correctly predicted onescorrectly_predicted_prefixes
        data_last = dt_prefixes.copy()
        data_last = data_last.groupby(['Case ID']).last()
        caseIDs = list(data_last.index)
        dt_named = self.datacreator.transform_data_test(self.feature_combiner, self.scaler, dt_prefixes)
        y = self.datacreator.get_label_numeric_adversarial(dt_prefixes)
        print('amount of original cases', len(caseIDs))
        pred = classifier.predict_proba(dt_named)[:,-1]
        self.threshold = self.datacreator.Find_Optimal_Cutoff(y, pred)
        pred = (np.where(pred > self.threshold , 1, 0))
        indices = self.return_indices_correlated_guess(y, pred)
        self.correct_cases = list(itemgetter(*indices)(caseIDs))
        correct_y = list(itemgetter(*indices)(y))
        dt_prefixes2 = dt_prefixes.copy()
        data_last = dt_prefixes2.groupby(['Case ID']).last()
        data_last = data_last.reset_index()
        dt_prefixes2 = dt_prefixes2[dt_prefixes2['Case ID'].isin(self.correct_cases)]
        y2 = data_last[data_last['Case ID'].isin(self.correct_cases)]['label']
        print('amount of original cases', len(caseIDs))
        assert len(correct_y) == len(self.correct_cases) == len(dt_prefixes2.groupby(['Case ID']).last()) == len(y2)
        print('amount of original traces:', len(dt_prefixes))
        print('amount of correctly predicted traces:', len(dt_prefixes2))
        print('amount of correctly predicted cases:', len(self.correct_cases))
        dt_prefixes_correct = dt_prefixes2.copy()
        return dt_prefixes_correct, y2, self.correct_cases