# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:07:08 2023

@author: u0138175
"""

####################PACKAGES AND FUNCTIONS#######################
from util.Arguments import Args
from util.DataCreation import DataCreation
from sklearn.metrics import roc_auc_score
from LSTM import Model, LSTMModel
from settings import global_setting, model_setting, training_setting
from DatasetManager import DatasetManager
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import warnings
# packages from https://github.com/irhete/predictive-monitoring-benchmark/blob/master/experiments/experiments.py
# packages from https://github.com/Khamies/LSTM-Variational-AutoEncoder/tree/50476dd3bfe146bf8f4a74a205b78fb142e99423
# user-specified packages
warnings.simplefilter(action='ignore', category=FutureWarning)
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# to explicitly raise an error with a stack trace to easier debug which operation might have created the invalid values
torch.autograd.set_detect_anomaly(True)
##################################################################
torch.manual_seed(22)
#####################PARAMETERS###################################

dataset_ref_to_datasets = {
    #"production": ["production"],
    #"bpic2012": ["bpic2012_accepted"]
    #, "bpic2012_cancelled", "bpic2012_declined"],
     "bpic2015": ["bpic2015_%s_f2" % (municipality) for municipality in range(1, 2)],
    # sepsis_cases": ["sepsis_cases_4"],
    # "bpic2011": ["bpic2011_f%s"%formula for formula in range(2,3)],
    # "bpic2017": ["bpic2017_accepted", "bpic2017_cancelled", "bpic2017_refused"],
    # "traffic_fines": ["traffic_fines_%s" % formula for formula in range(1, 2)],
    # "hospital_billing": ["hospital_billing_%s" % suffix for suffix in [2, 3]]
}
datasets = []
for k, v in dataset_ref_to_datasets.items():
    datasets.extend(v)

best_LSTMs = global_setting['best_LSTMs']
# create params directory
if not os.path.exists(os.path.join(best_LSTMs)):
    os.makedirs(os.path.join(best_LSTMs))

for dataset_name in datasets:
    arguments = Args(dataset_name)
    print('Dataset:', dataset_name)
    method = 'manifold'
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()
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
    batch_size = len(activity_test)
    dataset = torch.utils.data.TensorDataset(activity_test, resource_test)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dir_path = global_setting['params_dir_DL'] + '/hyper_models/'+dataset_name
    dir_list = os.listdir(dir_path)
    if 'desktop.ini' in dir_list:
        dir_list.remove('desktop.ini')
    best_auc = 0.5
    best_LSTM_name = ""
    for LSTM_name in dir_list:
        print(LSTM_name)
        print(best_auc)
        split = LSTM_name.split("_")
        checkpoint = torch.load(dir_path+'/'+LSTM_name, map_location=torch.device('cpu'))
        learning_rate = float(split[1])
        hidden_size = int(split[2])
        
        optimizer_name = split[3]
        batch_size = int(split[4])
        embed_size = training_setting['embed_size']
        model = Model(embed_size, hidden_size, vocab_size, max_prefix_length)
        model.load_state_dict(checkpoint)
        model.eval()
        pred = model(activity_test,resource_test).squeeze(-1).to('cpu').detach().numpy()
        print(pred)
        auc = roc_auc_score(test_y, pred)
        print('here',auc)
        if auc > best_auc:
            print('auc',auc,'best_auc', best_auc)
            best_auc = auc.copy()
            best_model = model
            best_LSTM_name = LSTM_name
            print('best auc now is:', best_auc)
    print('best model with name:', best_LSTM_name, 'and auc', best_auc)
    path_data_label = best_LSTMs+'/' + dataset_name+'/'
    if not os.path.exists(os.path.join(path_data_label)):
        os.makedirs(os.path.join(path_data_label))
    best_model_path = os.path.join(path_data_label, best_LSTM_name + '.pt')
    torch.save(best_model.state_dict(), best_model_path)
