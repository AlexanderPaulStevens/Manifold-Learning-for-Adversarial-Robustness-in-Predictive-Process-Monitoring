from operator import itemgetter
import torch.nn.functional as F
from attacks import AdversarialAttacks
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
import numpy as np
from itertools import cycle, islice
import collections
from sklearn.metrics import roc_auc_score, accuracy_score
g = torch.Generator()
g.manual_seed(0)
torch.manual_seed(32)
seed = global_setting['seed']
random.seed(seed)
#os.chdir('G:\My Drive\CurrentWork\Manifold\AdversarialRobustnessGeneralization')
class Manifold:
    """class to define adversarial attacks."""

    def __init__(self, dataset_name, scaler, feature_combiner, dataset_manager, datacreator, max_prefix_length, train, train_prefixes, test_prefixes, attack, activity_col, resource_col, cat_cols, cols, vocab_size, payload_values):
        self.dataset_name = dataset_name
        self.dataset_manager = dataset_manager
        self.seed = global_setting['seed']
        self.path = global_setting['models']
        self.train_ratio = global_setting['train_ratio']
        self.params_dir = global_setting['params_dir_DL']
        self.best_manifold_path = global_setting['best_manifolds']
        self.results_dir = global_setting['results_dir'] + '/' + self.dataset_name
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
        self.no_cols_list = [self.vocab_size[0]-1, self.vocab_size[1]-1]
        self.attack = attack
        self.train = train
        self.train_prefixes = train_prefixes
        self.test_prefixes = test_prefixes

    def project_on_manifold(self, adversarial_prefixes, label):
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
        
        activity, resource, label2, cases = self.datacreator.groupby_pad(adversarial_prefixes_label, self.cols, self.activity_col, self.resource_col)

        # MODEL ARCHITECTURE
        # create the input layers and embeddings
        
        batch_size = activity.size(0)
        dataset = torch.utils.data.TensorDataset(activity, resource)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model = LSTM_VAE(vocab_size=self.vocab_size, embed_size=16, hidden_size=latent_size, latent_size=latent_size).to(self.device)
        model.load_state_dict(checkpoint)
        states = model.init_hidden(batch_size)
        Loss = VAE_Loss()
        trainer = Trainer(None, dataset, model, Loss, optimizer)
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
        ans,_,_ = self.datacreator.groupby_caseID(adversarial_prefixes_label, self.cols, self.activity_col)
        manifold_activities = self.attack.prefix_lengths_adversarial(ans, x_hat_param_act1)
        activities = np.concatenate([x.ravel() for x in manifold_activities])
        manifold_resources = self.attack.prefix_lengths_adversarial(ans, x_hat_param_res1)
        resources = np.concatenate([x.ravel() for x in manifold_resources])
        on_manifold_adversarial = adversarial_prefixes_label.copy()
        on_manifold_adversarial['Case ID_label'] = on_manifold_adversarial['Case ID'] + label
        on_manifold_adversarial['Activity'] = activities
        on_manifold_adversarial['Resource'] = resources
        #on_manifold_adversarial = ce.inverse_transform(on_manifold_adversarial)
        return on_manifold_adversarial

    def project_on_manifold_ML(self, adversarial_prefixes, label):
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
        _, train_cat_cols, ce = self.datacreator.prepare_inputs(self.train.loc[:, self.cat_cols], adversarial_prefixes_label.loc[:, self.cat_cols])
        adversarial_prefixes_label[self.cat_cols] = train_cat_cols

        activity, resource, label2, cases = self.datacreator.groupby_pad(adversarial_prefixes_label, self.cols, self.activity_col, self.resource_col)

        # MODEL ARCHITECTURE
        # create the input layers and embeddings
        batch_size = activity.size(0)
        model = LSTM_VAE(vocab_size=self.vocab_size, embed_size=16, hidden_size=latent_size, latent_size=latent_size).to(self.device)
        model.load_state_dict(checkpoint)
        states = model.init_hidden(batch_size)
        Loss = VAE_Loss()
        model.eval()
        activity = activity.to(self.device)
        resource = resource.to(self.device)
        sentences_length  = torch.tensor([activity.shape[1]]*activity.shape[0])
        x_hat_param_act1, x_hat_param_res1, mu1, log_var, z1, states = model(activity, resource, sentences_length, states)
        ans,_,_ = self.datacreator.groupby_caseID(adversarial_prefixes_label, self.cols, self.activity_col)
        manifold_activities = self.attack.prefix_lengths_adversarial(ans, x_hat_param_act1)
        activities = np.concatenate([x.ravel() for x in manifold_activities])
        manifold_resources = self.attack.prefix_lengths_adversarial(ans, x_hat_param_res1)
        resources = np.concatenate([x.ravel() for x in manifold_resources])
        on_manifold_adversarial = adversarial_prefixes_label.copy()
        on_manifold_adversarial['Case ID_label'] = on_manifold_adversarial['Case ID'] + label
        on_manifold_adversarial['Activity'] = activities
        on_manifold_adversarial['Resource'] = resources
        on_manifold_adversarial = ce.inverse_transform(on_manifold_adversarial)
        return on_manifold_adversarial

    def create_manifold_dataset(self, adversarial_prefixes_train):
        """Manifold experiment."""
        # on-manifold training data
        on_manifold_adversarial_regular = pd.DataFrame()
        on_manifold_adversarial_deviant = pd.DataFrame()
        adversarial_prefixes_train_regular = adversarial_prefixes_train[adversarial_prefixes_train.label == 0]
        adversarial_prefixes_train_deviant = adversarial_prefixes_train[adversarial_prefixes_train.label == 1]
        if len(adversarial_prefixes_train_regular) > 0:
            on_manifold_adversarial_regular = self.project_on_manifold_ML(adversarial_prefixes_train_regular, 'regular')
        if len(adversarial_prefixes_train_deviant) > 0:
            on_manifold_adversarial_deviant = self.project_on_manifold_ML(adversarial_prefixes_train_deviant, 'deviant')
            on_manifold_prefixes_train_total = pd.concat([on_manifold_adversarial_deviant, on_manifold_adversarial_regular])
            
        if len(adversarial_prefixes_train_deviant) > 0:
            on_manifold_prefixes_train_total = pd.concat([on_manifold_adversarial_deviant, on_manifold_adversarial_regular])

        else:
            on_manifold_prefixes_train_total = pd.concat([on_manifold_adversarial_deviant, on_manifold_adversarial_regular])

        train_y_manifold = self.datacreator.get_label_numeric_adversarial(on_manifold_prefixes_train_total)
        print('done')
        return on_manifold_prefixes_train_total, train_y_manifold

    def create_manifold_dataset_LSTM(self, adversarial_prefixes_train):
        """Manifold experiment."""
        # on-manifold training data
        on_manifold_adversarial_regular = pd.DataFrame()
        on_manifold_adversarial_deviant = pd.DataFrame()
        adversarial_prefixes_train_regular = adversarial_prefixes_train[adversarial_prefixes_train.label == 0]
        adversarial_prefixes_train_deviant = adversarial_prefixes_train[adversarial_prefixes_train.label == 1]
        if len(adversarial_prefixes_train_regular) > 0:
            on_manifold_adversarial_regular = self.project_on_manifold(adversarial_prefixes_train_regular, 'regular')
        if len(adversarial_prefixes_train_deviant) > 0:
            on_manifold_adversarial_deviant = self.project_on_manifold(adversarial_prefixes_train_deviant, 'deviant')
            on_manifold_prefixes_train_total = pd.concat([on_manifold_adversarial_deviant, on_manifold_adversarial_regular])

        if len(adversarial_prefixes_train_deviant) > 0:
            on_manifold_prefixes_train_total = pd.concat([on_manifold_adversarial_deviant, on_manifold_adversarial_regular])
        else:
            on_manifold_prefixes_train_total = pd.concat([on_manifold_adversarial_deviant, on_manifold_adversarial_regular])

        train_y_manifold = self.datacreator.get_label_numeric_adversarial(on_manifold_prefixes_train_total)
        return on_manifold_prefixes_train_total, train_y_manifold
    
    
    def perform_attack(self, attack_type, attack_col, cls, attack_manifold, modelmaker, results_cls, results_test):
        print('attack', attack_type, attack_col)
        # Adversarial - Training Data
        print('perform attack')
        adversarial_prefixes_train, _ = self.attack.create_adversarial_dt_named_train(attack_type, attack_col, cls)
        adversarial_train_prefixes = pd.concat([self.train_prefixes, adversarial_prefixes_train])
        train_y = self.datacreator.get_label_numeric_adversarial(adversarial_train_prefixes)
        dt_train_named = self.datacreator.transform_data_test(self.feature_combiner, self.scaler, adversarial_train_prefixes)
        cls_adversarial = modelmaker.model_maker(dt_train_named, train_y)
        dt_test_named = self.datacreator.transform_data_test(self.feature_combiner, self.scaler, self.test_prefixes)
        test_y = self.datacreator.get_label_numeric_adversarial(self.test_prefixes)
        predictions = cls_adversarial.predict_proba(dt_test_named)[:, -1]
        auc = roc_auc_score(test_y, predictions)
        print('auc adversarial', auc)
        # Adversarial - Testing Data
        #adversarial_prefixes_test = attack.create_adversarial_dt_named_test(attack_type, attack_col, cls)
        #test_y = datacreator.get_label_numeric_adversarial(adversarial_prefixes_test)
        #dt_test_named = datacreator.transform_data_test(feature_combiner, scaler, adversarial_prefixes_test)

        # Save Adversarial Results to Dictionary
        results_cls['adv_cls_' + attack_type + '_' + attack_col] = cls_adversarial
        results_test['adv_test_' + attack_type + '_' + attack_col] = (dt_test_named, test_y)
        print('manifold')  
        # On-Manifold - Training Data
        manifold_prefixes_train = attack_manifold.create_adversarial_dt_named_train(attack_type, attack_col, cls)
        manifold_train_prefixes = pd.concat([self.train_prefixes, manifold_prefixes_train])
        train_y_manifold = self.datacreator.get_label_numeric_adversarial(manifold_train_prefixes)
        dt_train_named_manifold = self.datacreator.transform_data_test(self.feature_combiner, self.scaler, manifold_train_prefixes)
        cls_manifold = modelmaker.model_maker(dt_train_named_manifold, train_y_manifold)
        dt_test_named = self.datacreator.transform_data_test(self.feature_combiner, self.scaler, self.test_prefixes)
        test_y = self.datacreator.get_label_numeric_adversarial(self.test_prefixes)
        predictions = cls_manifold.predict_proba(dt_test_named)[:, -1]
        auc = roc_auc_score(test_y, predictions)
        print('auc manifold', auc)
        # On-Manifold - Testing Data
        manifold_prefixes_test = attack_manifold.create_adversarial_dt_named_test(attack_type, attack_col, cls)
        test_y_manifold = self.datacreator.get_label_numeric_adversarial(manifold_prefixes_test)
        dt_test_named_manifold = self.datacreator.transform_data_test(self.feature_combiner, self.scaler, manifold_prefixes_test)

        # Save On-Manifold Results to Dictionary
        results_cls['manifold_cls_' + attack_type + '_' + attack_col] = cls_manifold
        results_test['manifold_test_' + attack_type + '_' + attack_col] = (dt_test_named_manifold, test_y_manifold)
        print('saved to dictionary')
        return results_cls, results_test
    
    def perform_attack_DL(self, attack_type, attack_col, activity_val, resource_val, val_y, cls, attack_manifold, lstmmodel, results_cls, results_test):
        print('attack', attack_type, attack_col)
        
        # Adversarial - Training Data
        adversarial_prefixes_train, _ = self.attack.create_adversarial_dt_named_train(attack_type, attack_col, cls)
        adversarial_train_prefixes = pd.concat([self.train_prefixes, adversarial_prefixes_train])
        activity_train, resource_train, train_y,_ = self.datacreator.groupby_pad(adversarial_train_prefixes, self.cols, self.activity_col, self.resource_col)
        cls_adv = lstmmodel.make_LSTM_model(activity_train, resource_train, activity_val, resource_val, train_y, val_y)
        
        # Adversarial - Testing Data
        adversarial_prefixes_test = self.attack.create_adversarial_dt_named_test(attack_type, attack_col, cls)
        activity_test_adv, resource_test_adv, test_y_adv, _ = self.datacreator.groupby_pad(adversarial_prefixes_test, self.cols, self.activity_col, self.resource_col)
        
        # Save Adversarial Results to Dictionary
        results_cls['adv_cls_' + attack_type + '_' + attack_col] = cls_adv
        results_test['adv_test_' + attack_type + '_' + attack_col] = (activity_test_adv, resource_test_adv, test_y_adv)
        
        # On-Manifold - Training Data
        manifold_prefixes_train = attack_manifold.create_adversarial_dt_named_LSTM_train(attack_type, attack_col, cls)
        manifold_train_prefixes = pd.concat([self.train_prefixes, manifold_prefixes_train])
        activity_train_manifold, resource_train_manifold, train_y_manifold,_ = self.datacreator.groupby_pad(manifold_train_prefixes, self.cols, self.activity_col, self.resource_col)
        cls_manifold = lstmmodel.make_LSTM_model(activity_train_manifold, resource_train_manifold, activity_val, resource_val, train_y_manifold, val_y)

        # On-Manifold - Testing Data
        manifold_prefixes_test = attack_manifold.create_adversarial_dt_named_LSTM_test(attack_type, attack_col, cls)
        activity_test_manifold, resource_test_manifold, test_y_manifold,_ = self.datacreator.groupby_pad(manifold_prefixes_test, self.cols, self.activity_col, self.resource_col)
    
        # Save On-Manifold Results to Dictionary
        results_cls['manifold_cls_' + attack_type + '_' + attack_col] = cls_manifold
        results_test['manifold_test_' + attack_type + '_' + attack_col] = (activity_test_manifold, resource_test_manifold, test_y_manifold)
        print('saved to dictionary')
        return results_cls, results_test

class AdversarialAttacks_manifold:
    """class to define adversarial attacks."""

    def __init__(self, dataset_name,dataset_manager, max_prefix_length, train, train_prefixes, test_prefixes, payload_values, datacreator, attack, manifold_creator):
        self.dataset_name = dataset_name
        self.dataset_manager = dataset_manager
        self.max_prefix_length = max_prefix_length
        self.payload_values = payload_values
        self.datacreator = datacreator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.attack = attack
        self.train = train
        self.train_prefixes = train_prefixes
        self.test_prefixes = test_prefixes
        self.manifold_creator = manifold_creator 
        
    def create_adversarial_dt_named_test(self, attack_type, attack_col, classifier):
        dt_prefixes_correct, _, caseIDs = self.attack.correctly_predicted_prefixes(self.test_prefixes, classifier)
        total_cases = list(self.test_prefixes['Case ID'].unique())
        caseIDs_incorrect = list(set(total_cases)-set(caseIDs))
        incorrect_dt_prefixes = self.test_prefixes[self.test_prefixes['Case ID'].isin(caseIDs_incorrect)].copy()
        # create adversarial prefixes of the original instances. These can be adversarial examples of the same trace,
        # but they should have a different case ID
        adversarial_prefixes = self.create_adversarial_prefixes(attack_type, attack_col, caseIDs, self.test_prefixes, dt_prefixes_correct, classifier, 'test')
        adversarial_prefixes = pd.concat([adversarial_prefixes, incorrect_dt_prefixes])
        return adversarial_prefixes

    def create_adversarial_dt_named_train(self, attack_type, attack_col, classifier):
        dt_prefixes_correct,_ , caseIDs = self.attack.correctly_predicted_prefixes(self.train_prefixes, classifier)
        adversarial_prefixes = self.create_adversarial_prefixes(attack_type, attack_col, caseIDs, self.train_prefixes, dt_prefixes_correct, classifier, 'train')
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
                    dt_prefixes_adv = self.attack.permute_all_event(dt_prefixes2, attack_col)
                elif attack_type == 'last_event':
                    dt_prefixes_adv = self.attack.permute_last_event(dt_prefixes2, attack_col)
                dt_prefixes_adv.loc[dt_prefixes_adv["label"] == "deviant", "label"] = 1
                dt_prefixes_adv.loc[dt_prefixes_adv["label"] == "regular", "label"] = 0
                dt_prefixes_adv, y_manifold = self.manifold_creator.create_manifold_dataset(dt_prefixes_adv)
                adversarial_cases = self.attack.check_adversarial_cases(dt_prefixes_adv, y_manifold, classifier, caseIDs)
                if len(adversarial_cases) > 0:
                    total_adversarial_cases.extend(adversarial_cases)
                    if len(total_adversarial_cases) > len(total_caseIDs):
                        total_adversarial_cases = total_adversarial_cases[0:len(total_caseIDs)]
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
"""
Adversarial attacks of the manifold
"""

class AdversarialAttacks_manifold_LSTM:
    """class to define adversarial attacks."""

    def __init__(self, dataset_name,dataset_manager, max_prefix_length, train, train_prefixes,test_prefixes, activity_col, resource_col, cat_cols, cols, payload_values, vocab_size, datacreator, attack, no_cols_list, manifold_creator):
        self.max_prefix_length = max_prefix_length
        self.payload_values = payload_values
        self.datacreator = datacreator
        self.dataset_manager = dataset_manager
        self.cat_cols = cat_cols
        self.cols = cols
        self.activity_col = activity_col
        self.resource_col = resource_col
        self.no_cols_list = no_cols_list
        self.attack = attack
        self.train = train
        self.train_prefixes = train_prefixes
        self.test_prefixes = test_prefixes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.manifold_creator = manifold_creator
        
    def create_adversarial_dt_named_LSTM_train(self, attack_type, attack_col, classifier):
        print('correct')
        dt_prefixes_correct, y_correct, caseIDs, correct_cases = self.attack.correctly_predicted_prefixes(self.train_prefixes, classifier)
        # create adversarial train prefixes of the original instances. These can be adversarial examples of the same trace,
        # but they should have a different case ID
        print('adversarial')
        adversarial_prefixes = self.create_adversarial_prefixes_LSTM(attack_type, attack_col, correct_cases, self.train_prefixes, dt_prefixes_correct, classifier, 'train')
        return adversarial_prefixes

    def create_adversarial_dt_named_LSTM_test(self, attack_type, attack_col, classifier):
        dt_prefixes_correct, y_correct, caseIDs, correct_cases = self.attack.correctly_predicted_prefixes(self.test_prefixes, classifier)
        # create adversarial train prefixes of the original instances. These can be adversarial examples of the same trace,
        # but they should have a different case ID
        caseIDs_incorrect = list(set(caseIDs)-set(correct_cases))
        incorrect_dt_prefixes = self.test_prefixes[self.test_prefixes['Case ID'].isin(caseIDs_incorrect)].copy()
        adversarial_prefixes = self.create_adversarial_prefixes_LSTM(attack_type, attack_col, correct_cases, self.test_prefixes, dt_prefixes_correct, classifier, 'test')
        adversarial_prefixes_total = pd.concat([incorrect_dt_prefixes, adversarial_prefixes])
        return adversarial_prefixes_total

    def create_adversarial_prefixes_LSTM(self, attack_type, attack_col, caseIDs, dt_prefixes, dt_prefixes_correct, classifier, traintest):
        """Create the adversarial prefixes."""
        loop_count = 0
        total_adversarial_cases = []
        adversarial_prefixes = pd.DataFrame()
        dt_prefixes_original = dt_prefixes.copy()
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
                dt_prefixes_adv = dt_prefixes_correct.copy()
                # These are the adversarially attacked examples
                if attack_type == 'all_event':
                    dt_prefixes_adv = self.attack.permute_all_event(dt_prefixes_adv, attack_col)
                elif attack_type == 'last_event':
                    dt_prefixes_adv = self.attack.permute_last_event(dt_prefixes_adv, attack_col)
                dt_prefixes_adv, y_manifold = self.manifold_creator.create_manifold_dataset_LSTM(adversarial_prefixes_train= dt_prefixes_adv)
                adversarial_cases = self.attack.check_adversarial_cases(dt_prefixes_adv, classifier)
                if len(adversarial_cases) > 0:
                    total_adversarial_cases.extend(adversarial_cases)
                    if len(total_adversarial_cases) > len(total_caseIDs):
                        total_adversarial_cases = total_adversarial_cases[0:len(total_caseIDs)]
                    created_adversarial_df = dt_prefixes_adv[dt_prefixes_adv['Case ID'].isin(adversarial_cases)].copy()
                    created_adversarial_df.loc[:, 'Case ID'] = created_adversarial_df.loc[:, 'Case ID'] + '_adv' + str(loop_count)
                    adversarial_prefixes = pd.concat([created_adversarial_df, adversarial_prefixes])
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