# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:07:08 2023

@author: u0138175
"""

####################PACKAGES AND FUNCTIONS#######################
from loss import VAE_Loss
from train import Trainer
import torchaudio
import os
import warnings
import numpy as np
import pandas as pd
import torch
import pickle
# packages from https://github.com/irhete/predictive-monitoring-benchmark/blob/master/experiments/experiments.py
from DatasetManager import DatasetManager
# packages from https://github.com/Khamies/LSTM-Variational-AutoEncoder/tree/50476dd3bfe146bf8f4a74a205b78fb142e99423
from settings import global_setting, model_setting, training_setting
from VAE import LSTM_VAE
# user-specified packages
from util.DataCreation import DataCreation
from util.Arguments import Args
warnings.simplefilter(action='ignore', category=FutureWarning)
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# to explicitly raise an error with a stack trace to easier debug which operation might have created the invalid values
torch.autograd.set_detect_anomaly(True)
##################################################################

#####################PARAMETERS###################################
os.chdir('G:\My Drive\CurrentWork\Manifold\AdversarialRobustnessGeneralization')
dataset_ref_to_datasets = {
    # "production": ["production"],
    "bpic2012": ["bpic2012_cancelled"]
    # , "bpic2012_accepted", "bpic2012_declined"],
    #"bpic2015": ["bpic2015_%s_f2" % (municipality) for municipality in range(1, 3)],
    # sepsis_cases": ["sepsis_cases_4"],
    # "bpic2011": ["bpic2011_f%s"%formula for formula in range(2,3)],
    # "bpic2017": ["bpic2017_accepted", "bpic2017_cancelled", "bpic2017_refused"],
    # "traffic_fines": ["traffic_fines_%s" % formula for formula in range(1, 2)],
    # "hospital_billing": ["hospital_billing_%s" % suffix for suffix in [2, 3]]
}
datasets = []
for k, v in dataset_ref_to_datasets.items():
    datasets.extend(v)
labels = ['regular', 'deviant']

best_manifolds = global_setting['best_manifolds']
# create params directory
if not os.path.exists(os.path.join(best_manifolds)):
    os.makedirs(os.path.join(best_manifolds))

for dataset_name in datasets:
    arguments = Args(dataset_name)
    for label in labels:
        print('Dataset:', dataset_name)
        print('label:', label)
        method = 'manifold'
        dataset_manager = DatasetManager(dataset_name)
        data = dataset_manager.read_dataset()
        cls_encoder_args, min_prefix_length, max_prefix_length, activity_col, resource_col = arguments.extract_args(data, dataset_manager)
        cat_cols = [activity_col, resource_col]
        cols = ['Case ID', 'label', 'event_nr'] + cat_cols
        no_cols_list = []
        seed = global_setting['seed']
        train_ratio = global_setting['train_ratio']

        clip = training_setting["clip"]
        epochs = training_setting["epochs"]
        embed_size = training_setting["embed_size"]
        ###################################################################

        #######################METHODOLOGY#################################
        datacreator = DataCreation(dataset_manager, dataset_name)
        no_cols_list = []
        # split into training and test
        train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
        for i in cat_cols:
            _, _, _, no_cols = datacreator.create_indexes(i, train)
            no_cols_list.append(no_cols)
        vocab_size = [no_cols_list[0]+1, no_cols_list[1]+1]
        # prefix generation of train and test data
        dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
        dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
        # cat columns integerencoded
        train_cat_cols, _, _ = datacreator.prepare_inputs(dt_train_prefixes.loc[:, cat_cols], dt_test_prefixes.loc[:, cat_cols])
        dt_train_prefixes[cat_cols] = train_cat_cols
        del train, test
        dt_train_prefixes = dt_train_prefixes[cols].copy()
        dt_test_prefixes = dt_test_prefixes[cols].copy()

        # groupby case ID
        ans_train_act = datacreator.groupby_caseID(dt_train_prefixes, cols, activity_col)
        ans_train_res = datacreator.groupby_caseID(dt_train_prefixes, cols, resource_col)
        ######ACTIVITY_COL########
        activity_train = datacreator.pad_data(ans_train_act)
        del ans_train_act
        # ######RESOURCE COL########
        resource_train = datacreator.pad_data(ans_train_res)
        ###################MODEL ARCHITECTURE#################################################
        # create the input layers and embeddings
        input_size = no_cols_list
        batch_size = len(activity_train)
        dataset = torch.utils.data.TensorDataset(activity_train, resource_train)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        del activity_train, resource_train

        path_manifold = global_setting['manifolds'] + '/' + dataset_name + '_' + label

        dir_list = os.listdir(path_manifold)
        if 'desktop.ini' in dir_list:
            dir_list.remove('desktop.ini')
        best_min_loss = 99999
        best_VAE_name = ""
        for VAE_name in dir_list:
            print(VAE_name)
            split = VAE_name.split("_")
            checkpoint = torch.load(path_manifold+'/'+VAE_name, map_location=torch.device('cpu'))
            latent_size = int(split[3])
            optimizer = split[4]
            hidden_size = latent_size
            model = LSTM_VAE(vocab_size=vocab_size, embed_size=16, hidden_size=hidden_size, latent_size=latent_size).to(device)
            model.load_state_dict(checkpoint)
            Loss = VAE_Loss()
            trainer = Trainer(_, dataset, model, Loss, optimizer)
            distances = []
            hidden_cell = torch.zeros(1, batch_size, hidden_size)
            state_cell = torch.zeros(1, batch_size, hidden_size)
            states = (hidden_cell, state_cell)
            manifold_activities, manifold_resources = [], []
            with torch.no_grad():
                states = model.init_hidden(batch_size)
                for i, (data_act, data_res) in enumerate(dataset, 0):
                    # get the labels
                    sentences_length = trainer.get_batch(data_act)
                    data_act = data_act.to(device)
                    data_res = data_res.to(device)
                    x_hat_param_act, x_hat_param_res, mu, log_var, z, states = model(data_act, data_res, sentences_length, states)
                    # detach hidden states
                    states = states[0].detach(), states[1].detach()

                    # compute the loss
                    mloss, KL_loss, recon_loss = Loss(mu=mu, log_var=log_var, z=z, x_hat_param_act=x_hat_param_act, x_hat_param_res=x_hat_param_res, x_act=data_act, x_res=data_res)

                    distances.append((mloss.item(), KL_loss.item(), recon_loss.item()))
                print('mean', np.mean(distances))
                if np.mean(distances) < best_min_loss:
                    best_min_loss = np.mean(distances)
                    best_VAE_name = VAE_name
                    best_model = model
                    print('best loss now is:', best_min_loss, 'from VAE name', best_VAE_name)
        print('best model with name:', best_VAE_name, 'and loss', best_min_loss)
        path_data_label = best_manifolds+'/' + dataset_name+'_'+label+'/'
        if not os.path.exists(os.path.join(path_data_label)):
            os.makedirs(os.path.join(path_data_label))
        best_model_path = os.path.join(path_data_label, best_VAE_name + '.pt')
        torch.save(best_model.state_dict(), best_model_path)
