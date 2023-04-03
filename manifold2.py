# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:12:34 2023

@author: u0138175
"""
from operator import itemgetter
import torch.nn.functional as F
from attacks import AdversarialAttacks, AdversarialAttacks_LSTM
from train import Trainer
import os
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from settings import global_setting, training_setting
from VAE import LSTM_VAE
from loss import VAE_Loss
import torchaudio
import numpy as np
from itertools import cycle, islice
import collections
from sklearn.metrics import roc_auc_score, accuracy_score
g = torch.Generator()
g.manual_seed(0)
torch.manual_seed(32)


class Manifold:
    """class to define adversarial attacks."""

    def __init__(self, dataset_name, scaler, feature_combiner, dataset_manager, datacreator, min_prefix_length, max_prefix_length, activity_col, resource_col, cat_cols, cols, vocab_size, payload_values):
        self.dataset_name = dataset_name
        self.dataset_manager = dataset_manager
        self.seed = global_setting['seed']
        self.path = global_setting['models']
        self.train_ratio = global_setting['train_ratio']
        self.params_dir = global_setting['params_dir_DL']
        self.best_manifold_path = global_setting['best_manifolds']
        self.results_dir = global_setting['results_dir'] + '/' + self.dataset_name
        self.min_prefix_length = min_prefix_length
        self.max_prefix_length = max_prefix_length
        self.activity_col = activity_col
        self.resource_col = resource_col
        self.datacreator = datacreator
        self.cat_cols = cat_cols
        self.cols = cols
        self.vocab_size = vocab_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.payload_values = payload_values
        self.feature_combiner = feature_combiner
        self.scaler = scaler

    def project_on_manifold(self, train, adversarial_prefixes, label):
        """Project onto the manifold."""
        path_data_label = 'best_manifolds'+'/' + self.dataset_name + '_' + label + '/'
        dir_list = os.listdir(path_data_label)
        if 'desktop.ini' in dir_list:
            dir_list.remove('desktop.ini')
        VAE_name = dir_list[0]
        split = VAE_name.split("_")
        checkpoint = torch.load(path_data_label+'/'+VAE_name, map_location=torch.device('cpu'))
        latent_size = int(split[3])
        optimizer = split[4]
        adversarial_prefixes_label = adversarial_prefixes.copy()

        # groupby case ID
        ans_act = self.datacreator.groupby_caseID(adversarial_prefixes_label, self.cols, self.activity_col)
        ans_res = self.datacreator.groupby_caseID(adversarial_prefixes_label, self.cols, self.resource_col)
        # padding
        activity_padded = self.datacreator.pad_data(ans_act).to(self.device)
        resource_padded = self.datacreator.pad_data(ans_res).to(self.device)

        # MODEL ARCHITECTURE
        # create the input layers and embeddings
        batch_size = len(ans_act)
        dataset = torch.utils.data.TensorDataset(activity_padded, resource_padded)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = LSTM_VAE(vocab_size=self.vocab_size, embed_size=16, hidden_size=latent_size, latent_size=latent_size).to(self.device)
        model.load_state_dict(checkpoint)

        states = model.init_hidden(batch_size)
        Loss = VAE_Loss()
        trainer = Trainer(None, dataset, model, Loss, optimizer)
        attack = AdversarialAttacks(self.max_prefix_length, self.payload_values, self.datacreator, self.feature_combiner, self.scaler, self.dataset_manager)
        with torch.no_grad():
            i = 0
            for i, (data_act, data_res) in enumerate(dataset, 0):
                i += 1
                data_act = data_act.to(self.device)
                data_res = data_res.to(self.device)
                sentences_length = trainer.get_batch(data_act)
                x_hat_param_act1, x_hat_param_res1, mu1, log_var, z1, states = model(data_act, data_res, sentences_length, states)
        x_hat_param_act1 = x_hat_param_act1.cpu()
        x_hat_param_res1 = x_hat_param_res1.cpu()
        manifold_activities = attack.prefix_lengths_adversarial(ans_act, x_hat_param_act1)
        activities = np.concatenate([x.ravel() for x in manifold_activities])
        manifold_resources = attack.prefix_lengths_adversarial(ans_res, x_hat_param_res1)
        resources = np.concatenate([x.ravel() for x in manifold_resources])
        on_manifold_adversarial = adversarial_prefixes_label.copy()
        on_manifold_adversarial['Case ID_label'] = on_manifold_adversarial['Case ID'] + label
        on_manifold_adversarial['Activity'] = activities
        on_manifold_adversarial['Resource'] = resources
        #on_manifold_adversarial = ce.inverse_transform(on_manifold_adversarial)
        return on_manifold_adversarial

    def project_on_manifold_ML(self, train, adversarial_prefixes, label):
        """Project onto the manifold."""
        path_data_label = 'best_manifolds'+'/' + self.dataset_name + '_' + label + '/'
        dir_list = os.listdir(path_data_label)
        if 'desktop.ini' in dir_list:
            dir_list.remove('desktop.ini')
        VAE_name = dir_list[0]
        split = VAE_name.split("_")
        checkpoint = torch.load(path_data_label+'/'+VAE_name, map_location=torch.device('cpu'))
        latent_size = int(split[3])
        optimizer = split[4]
        adversarial_prefixes_label = adversarial_prefixes.copy()
        _, train_cat_cols, ce = self.datacreator.prepare_inputs(train.loc[:, self.cat_cols], adversarial_prefixes_label.loc[:, self.cat_cols])
        adversarial_prefixes_label[self.cat_cols] = train_cat_cols

        # groupby case ID
        ans_act = self.datacreator.groupby_caseID(adversarial_prefixes_label, self.cols, self.activity_col)
        ans_res = self.datacreator.groupby_caseID(adversarial_prefixes_label, self.cols, self.resource_col)

        # padding
        activity_padded = self.datacreator.pad_data(ans_act)
        resource_padded = self.datacreator.pad_data(ans_res)

        # MODEL ARCHITECTURE
        # create the input layers and embeddings
        batch_size = len(ans_act)
        dataset = torch.utils.data.TensorDataset(activity_padded, resource_padded)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = LSTM_VAE(vocab_size=self.vocab_size, embed_size=16, hidden_size=latent_size, latent_size=latent_size).to(self.device)
        model.load_state_dict(checkpoint)
        states = model.init_hidden(batch_size)
        Loss = VAE_Loss()
        trainer = Trainer(None, dataset, model, Loss, optimizer)
        attack = AdversarialAttacks(self.max_prefix_length, self.payload_values, self.datacreator, self.feature_combiner, self.scaler, self.dataset_manager)
        with torch.no_grad():
            i = 0
            for i, (data_act, data_res) in enumerate(dataset, 0):
                i += 1
                data_act = data_act.to(self.device)
                data_res = data_res.to(self.device)
                sentences_length = trainer.get_batch(data_act)
                x_hat_param_act1, x_hat_param_res1, mu1, log_var, z1, states = model(data_act, data_res, sentences_length, states)

        manifold_activities = attack.prefix_lengths_adversarial(ans_act, x_hat_param_act1)
        activities = np.concatenate([x.ravel() for x in manifold_activities])
        manifold_resources = attack.prefix_lengths_adversarial(ans_res, x_hat_param_res1)
        resources = np.concatenate([x.ravel() for x in manifold_resources])
        on_manifold_adversarial = adversarial_prefixes_label.copy()
        on_manifold_adversarial['Case ID_label'] = on_manifold_adversarial['Case ID'] + label
        on_manifold_adversarial['Activity'] = activities
        on_manifold_adversarial['Resource'] = resources
        on_manifold_adversarial = ce.inverse_transform(on_manifold_adversarial)
        return on_manifold_adversarial

    def create_manifold_dataset(self, train, adversarial_prefixes_train):
        """Manifold experiment."""
        # on-manifold training data
        on_manifold_adversarial_regular = pd.DataFrame()
        on_manifold_adversarial_deviant = pd.DataFrame()
        adversarial_prefixes_train_regular = adversarial_prefixes_train[adversarial_prefixes_train.label == 0]
        adversarial_prefixes_train_deviant = adversarial_prefixes_train[adversarial_prefixes_train.label == 1]
        if len(adversarial_prefixes_train_regular) > 0:
            on_manifold_adversarial_regular = self.project_on_manifold_ML(train, adversarial_prefixes_train_regular, 'regular')
        if len(adversarial_prefixes_train_deviant) > 0:
            on_manifold_adversarial_deviant = self.project_on_manifold_ML(train, adversarial_prefixes_train_deviant, 'deviant')
            on_manifold_prefixes_train_total = on_manifold_adversarial_deviant.append(on_manifold_adversarial_regular)

        if len(adversarial_prefixes_train_deviant) > 0:
            on_manifold_prefixes_train_total = on_manifold_adversarial_regular.append(on_manifold_adversarial_deviant)
        else:
            on_manifold_prefixes_train_total = on_manifold_adversarial_deviant.append(on_manifold_adversarial_regular)

        train_y_manifold = self.datacreator.get_label_numeric_adversarial(on_manifold_prefixes_train_total)
        print('done')
        return on_manifold_prefixes_train_total, train_y_manifold

    def create_manifold_dataset_LSTM(self, train, adversarial_prefixes_train):
        """Manifold experiment."""
        # on-manifold training data
        on_manifold_adversarial_regular = pd.DataFrame()
        on_manifold_adversarial_deviant = pd.DataFrame()
        adversarial_prefixes_train_regular = adversarial_prefixes_train[adversarial_prefixes_train.label == 0]
        adversarial_prefixes_train_deviant = adversarial_prefixes_train[adversarial_prefixes_train.label == 1]
        if len(adversarial_prefixes_train_regular) > 0:
            on_manifold_adversarial_regular = self.project_on_manifold(train, adversarial_prefixes_train_regular, 'regular')
        if len(adversarial_prefixes_train_deviant) > 0:
            on_manifold_adversarial_deviant = self.project_on_manifold(train, adversarial_prefixes_train_deviant, 'deviant')
            on_manifold_prefixes_train_total = on_manifold_adversarial_deviant.append(on_manifold_adversarial_regular)

        if len(adversarial_prefixes_train_deviant) > 0:
            on_manifold_prefixes_train_total = on_manifold_adversarial_regular.append(on_manifold_adversarial_deviant)
        else:
            on_manifold_prefixes_train_total = on_manifold_adversarial_deviant.append(on_manifold_adversarial_regular)

        train_y_manifold = self.datacreator.get_label_numeric_adversarial(on_manifold_prefixes_train_total)
        return on_manifold_prefixes_train_total, train_y_manifold


class AdversarialAttacks_manifold:
    """class to define adversarial attacks."""

    def __init__(self, max_prefix_length, payload_values, datacreator, feature_combiner, scaler, dataset_manager, manifold_creator):
        self.max_prefix_length = max_prefix_length
        self.payload_values = payload_values
        self.datacreator = datacreator
        self.feature_combiner = feature_combiner
        self.scaler = scaler
        self.dataset_manager = dataset_manager
        self.manifold_creator = manifold_creator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def create_adversarial_dt_named_test(self, attack_type, attack_col, train, dt_test_prefixes, classifier):
        dt_prefixes2 = dt_test_prefixes.copy()
        dt_prefixes_correct, y_correct, caseIDs = self.correctly_predicted_prefixes(dt_prefixes2, classifier)
        total_cases = list(dt_prefixes2['Case ID'].unique())
        caseIDs_incorrect = list(set(total_cases)-set(caseIDs))
        incorrect_dt_prefixes = dt_prefixes2[dt_prefixes2['Case ID'].isin(caseIDs_incorrect)].copy()
        # create adversarial prefixes of the original instances. These can be adversarial examples of the same trace,
        # but they should have a different case ID
        adversarial_prefixes = self.create_adversarial_prefixes(attack_type, attack_col, caseIDs, train, dt_prefixes2, dt_prefixes_correct, classifier, 'test')
        adversarial_prefixes = adversarial_prefixes.append(incorrect_dt_prefixes)
        return adversarial_prefixes

    def create_adversarial_dt_named_train(self, attack_type, attack_col, train, dt_train_prefixes, classifier):
        dt_prefixes2 = dt_train_prefixes.copy()
        dt_prefixes_correct, y_correct, caseIDs = self.correctly_predicted_prefixes(dt_prefixes2, classifier)
        adversarial_prefixes = self.create_adversarial_prefixes(attack_type, attack_col, caseIDs, train, dt_prefixes2, dt_prefixes_correct, classifier, 'train')
        return adversarial_prefixes

    def create_adversarial_prefixes(self, attack_type, attack_col, caseIDs, train, dt_prefixes, dt_prefixes_correct, classifier, traintest):
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
                dt_prefixes_adv.loc[dt_prefixes_adv["label"] == "deviant", "label"] = 1
                dt_prefixes_adv.loc[dt_prefixes_adv["label"] == "regular", "label"] = 0
                dt_prefixes_adv, y_manifold = self.manifold_creator.create_manifold_dataset(train, dt_prefixes_adv)
                adversarial_cases = self.check_adversarial_cases(dt_prefixes_adv, y_manifold, classifier, caseIDs)
                if len(adversarial_cases) > 0:
                    total_adversarial_cases.extend(adversarial_cases)
                    if len(total_adversarial_cases) > len(total_caseIDs):
                        total_adversarial_cases = total_adversarial_cases[0:len(total_caseIDs)]
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

    def check_adversarial_cases(self, data, y2, classifier, caseIDs):
        """Check for adversarial cases."""
        dt_named = self.datacreator.transform_data_test(self.feature_combiner, self.scaler, data)
        y2 = self.datacreator.get_label_numeric_adversarial(data)
        pred = classifier.predict(dt_named)
        print('check adversarial cases:', 1-roc_auc_score(y2, pred))
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


"""


Adversarial attacks of the manifold



"""


class AdversarialAttacks_manifold_LSTM:
    """class to define adversarial attacks."""

    def __init__(self, max_prefix_length, payload_values, datacreator, dataset_manager, manifold_creator, cat_cols, cols, activity_col, resource_col, no_cols_list, attack_manifold):
        self.max_prefix_length = max_prefix_length
        self.payload_values = payload_values
        self.datacreator = datacreator
        self.dataset_manager = dataset_manager
        self.manifold_creator = manifold_creator
        self.cat_cols = cat_cols
        self.cols = cols
        self.activity_col = activity_col
        self.resource_col = resource_col
        self.no_cols_list = no_cols_list
        self.attack_manifold = attack_manifold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_dataset_LSTM(self, adversarial_train):
        adversarial_train = adversarial_train[self.cols]
        train_y_A = self.datacreator.get_label_numeric_adversarial(adversarial_train)
        # cat columns integerencoded
        # groupby case ID
        ans_train_act = self.datacreator.groupby_caseID(adversarial_train, self.cols, self.activity_col)
        ans_train_res = self.datacreator.groupby_caseID(adversarial_train, self.cols, self.resource_col)
        ######ACTIVITY AND RESOURCE COL########
        activity_train = self.datacreator.pad_data(ans_train_act).to(self.device)
        resource_train = self.datacreator.pad_data(ans_train_res).to(self.device)
        ###################MODEL ARCHITECTURE#################################################
        # create the input layers and embeddings
        input_size = self.no_cols_list

        return activity_train, resource_train, train_y_A

    def create_adversarial_dt_named_LSTM_train(self, attack_type, attack_col, train, dt_prefixes_original, activity, resource, classifier):
        print('correct')
        dt_prefixes_correct, y_correct, caseIDs, correct_cases = self.correctly_predicted_prefixes_LSTM(dt_prefixes_original, activity, resource, classifier)
        # create adversarial train prefixes of the original instances. These can be adversarial examples of the same trace,
        # but they should have a different case ID
        print('adversarial')
        adversarial_prefixes = self.create_adversarial_prefixes_LSTM(attack_type, attack_col, caseIDs, train, dt_prefixes_original, dt_prefixes_correct, classifier, 'train')
        return adversarial_prefixes

    def create_adversarial_dt_named_LSTM_test(self, attack_type, attack_col, train, dt_prefixes_original, activity, resource, classifier):
        dt_prefixes_correct, y_correct, caseIDs, correct_cases = self.correctly_predicted_prefixes_LSTM(dt_prefixes_original, activity, resource, classifier)
        # create adversarial train prefixes of the original instances. These can be adversarial examples of the same trace,
        # but they should have a different case ID
        caseIDs_incorrect = list(set(caseIDs)-set(correct_cases))
        incorrect_dt_prefixes = dt_prefixes_original[dt_prefixes_original['Case ID'].isin(caseIDs_incorrect)].copy()
        adversarial_prefixes = self.create_adversarial_prefixes_LSTM(attack_type, attack_col, correct_cases, train, dt_prefixes_original, dt_prefixes_correct, classifier, 'test')
        adversarial_prefixes = adversarial_prefixes.append(incorrect_dt_prefixes)
        return adversarial_prefixes

    def create_adversarial_prefixes_LSTM(self, attack_type, attack_col, caseIDs, train, dt_prefixes, dt_prefixes_correct, classifier, traintest):
        """Create the adversarial prefixes."""
        loop_count = 0
        total_adversarial_cases = []
        adversarial_prefixes = pd.DataFrame()
        # FROM HERE, YOU HAVE TO START YOUR LOOP
        if traintest == 'train':
            total_caseIDs = list(dt_prefixes['Case ID'].unique())
        elif traintest == 'test':
            total_caseIDs = caseIDs.copy()
        print('amount of total case IDs', len(total_caseIDs))
        while len(total_adversarial_cases) < len(total_caseIDs):
            adversarial_cases = 0
            loop_count += 1
            print('loop count', loop_count)
            if loop_count < 25:
                dt_prefixes2 = dt_prefixes_correct.copy()
                # These are the adversarially attacked examples
                if attack_type == 'all_event':
                    dt_prefixes_adv = self.attack_manifold.permute_all_event(dt_prefixes2, attack_col)
                elif attack_type == 'last_event':
                    dt_prefixes_adv = self.attack_manifold.permute_last_event(dt_prefixes2, attack_col)
                #dt_prefixes_adv.loc[dt_prefixes_adv["label"] == "deviant", "label"] = 1
                #dt_prefixes_adv.loc[dt_prefixes_adv["label"] == "regular", "label"] = 0
                dt_prefixes_adv, y_manifold = self.manifold_creator.create_manifold_dataset_LSTM(train, dt_prefixes_adv)

                adversarial_cases = self.check_adversarial_cases_LSTM(dt_prefixes_adv, y_manifold, caseIDs, classifier)
                if len(adversarial_cases) > 0:
                    total_adversarial_cases.extend(adversarial_cases)
                    if len(total_adversarial_cases) > len(total_caseIDs):
                        total_adversarial_cases = total_adversarial_cases[0:len(total_caseIDs)]
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

    def check_adversarial_cases_LSTM(self, data_adv, y2, correct_cases, classifier):
        """Check for adversarial cases."""
        classifier = classifier.to(self.device)
        activity, resource, _ = self.make_dataset_LSTM(data_adv)
        batch_size = activity.shape[0]
        states = classifier.init_hidden(batch_size)
        classifier.eval()
        with torch.no_grad():
            pred = classifier(activity, resource, states)
            pred = (np.where(np.array(pred.cpu().flatten()) > 0.50, 1, 0))
        indices = self.attack_manifold.return_indices_adversarial_guess(y2, pred)
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
            pred = (np.where(np.array(pred.flatten()) > 0.5, 1, 0))
        indices = self.attack_manifold.return_indices_correlated_guess(y, pred)
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
