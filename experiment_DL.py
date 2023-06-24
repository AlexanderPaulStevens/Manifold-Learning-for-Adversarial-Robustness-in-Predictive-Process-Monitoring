# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 09:37:13 2023

@author: u0138175
"""

####################PACKAGES AND FUNCTIONS#######################
import torchaudio
from train import Trainer
from loss import VAE_Loss
from operator import itemgetter
import torch
from settings import global_setting, model_setting, training_setting
from util.MakeModel import MakeModel
from util.Arguments import Args
from util.ModelCreation import ModelCreation
from util.DataCreation import DataCreation
import os
import pickle
import warnings
import random
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
# packages from https://github.com/irhete/predictive-monitoring-benchmark/blob/master/experiments/experiments.py
from DatasetManager import DatasetManager
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# to explicitly raise an error with a stack trace to easier debug which operation might have created the invalid values
torch.autograd.set_detect_anomaly(True)
####PARAMETERS####
# encodings dictionary
encoding = ['agg']
encoding_dict = {"agg": ["agg"]}
os.chdir('G:\My Drive\CurrentWork\Manifold\AdversarialRobustnessGeneralization')
# datasets dictionary
dataset_ref_to_datasets = {
    #"production": ["production"],
    #"bpic2015": ["bpic2015_%s_f2" % (municipality) for municipality in range(1, 2)],
     "bpic2012": ["bpic2012_accepted"]
       #           , "bpic2012_cancelled", "bpic2012_declined","bpic2012_accepted"],
    # "hospital_billing": ["hospital_billing_%s"%suffix for suffix in [2,3]],
    # "traffic_fines": ["traffic_fines_%s" % formula for formula in range(1, 2)],
    # "bpic2017": ["bpic2017_accepted","bpic2017_cancelled","bpic2017_refused"],
    # "sepsis_cases": ["sepsis_cases_2","sepsis_cases_4"],
    # "bpic2011": ["bpic2011_f%s"%formula for formula in range(2,4)],
}

datasets = []
for k, v in dataset_ref_to_datasets.items():
    datasets.extend(v)
# classifiers dictionary
classifier_ref_to_classifiers = {
    "DLmodels": ['LSTM'],
}
classifiers = []
for k, v in classifier_ref_to_classifiers.items():
    classifiers.extend(v)
# hyperparameters dictionary
cls_encoding = 'agg'
labels = ['regular']
seed = global_setting['seed']
train_ratio = global_setting['train_ratio']
params_dir = global_setting['params_dir_DL']
path = global_setting['models']

clip = training_setting["clip"]

# create params directory
if not os.path.exists(os.path.join(path)):
    os.makedirs(os.path.join(path))

# torch.load('manifolds/production_regular/LSTM_VAE_epoch9.pt')
for dataset_name in datasets:
    arguments = Args(dataset_name)
    for cls_method in classifiers:
        print('Dataset:', dataset_name)
        print('Classifier', cls_method)
        dataset_manager = DatasetManager(dataset_name)
        data = dataset_manager.read_dataset()
        method = 'manifold'
        cls_encoder_args, min_prefix_length, max_prefix_length, activity_col, resource_col = arguments.extract_args(data, dataset_manager)
        cat_cols = [activity_col, resource_col]
        cols = ['Case ID', 'label', 'event_nr'] + cat_cols
        no_cols_list = []
        seed = global_setting['seed']
        torch.manual_seed(22)
        train_ratio = global_setting['train_ratio']
        epochs = training_setting["epochs"]
        ###################################################################

        #######################METHODOLOGY#################################
        datacreator = DataCreation(dataset_manager, dataset_name)
        # split into training and test
        cat_cols = [activity_col, resource_col]
        no_cols_list = []
        cols = [cat_cols[0], cat_cols[1], cls_encoder_args['case_id_col'], 'label', 'event_nr', 'prefix_nr']
        train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
        train, _ = dataset_manager.split_data_strict(train, train_ratio, split="temporal")
        for i in cat_cols:
            _, _, _, no_cols = datacreator.create_indexes(i, train)
            no_cols_list.append(no_cols)
        vocab_size = [no_cols_list[0]+1, no_cols_list[1]+1]
        # you don't need to do that for the test data, as the prepare inputs is only fitted on the training data
        # prepare chunks for CV
        dt_prefixes = []

        # prefix generation of test data
        dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
        dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
        train_cat_cols, test_cat_cols, _ = datacreator.prepare_inputs(dt_train_prefixes.loc[:,cat_cols], dt_test_prefixes.loc[:,cat_cols])
        dt_test_prefixes[cat_cols] = test_cat_cols
        dt_train_prefixes[cat_cols] = train_cat_cols
        del train_cat_cols, test_cat_cols

        train_y = dataset_manager.get_label_numeric(dt_train_prefixes)
        test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
        
        #######################METHODOLOGY#################################
        dt_train_prefixes = dt_train_prefixes[cols].copy()
        dt_test_prefixes = dt_test_prefixes[cols].copy()

        # groupby case ID
        ans_train_act = datacreator.groupby_caseID(dt_train_prefixes, cols, activity_col)
        ans_test_act = datacreator.groupby_caseID(dt_test_prefixes, cols, activity_col)
        ans_train_res = datacreator.groupby_caseID(dt_train_prefixes, cols, resource_col)
        ans_test_res = datacreator.groupby_caseID(dt_test_prefixes, cols, resource_col)
        del dt_train_prefixes, dt_test_prefixes
            
        ######ACTIVITY_COL########
        activity_train = datacreator.pad_data(ans_train_act).to(device)
        activity_test = datacreator.pad_data(ans_test_act).to(device)
        del ans_train_act, ans_test_act
        
        # ######RESOURCE COL########
        resource_train = datacreator.pad_data(ans_train_res).to(device)
        resource_test = datacreator.pad_data(ans_test_res).to(device)
        del ans_train_res, ans_test_res
        
        # create the input layers and embeddings
        input_size = no_cols_list
        batch_size = len(activity_train)
        dataset = torch.utils.data.TensorDataset(activity_train, resource_train)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_y_all = []
        train_y_all.extend(train_y)
        # create original model
        optimal_params_filename = os.path.join(params_dir, "optimal_params_%s_%s.pickle" % (cls_method, dataset_name))
        args = arguments.params_args(optimal_params_filename)
        print(args)