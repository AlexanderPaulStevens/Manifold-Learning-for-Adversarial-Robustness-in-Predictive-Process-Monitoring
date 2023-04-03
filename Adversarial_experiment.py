# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 10:37:02 2023

@author: u0138175
"""

from VAE import LSTM_VAE
from train import Trainer
from loss import VAE_Loss
import torchaudio
import torch
from manifold import Manifold_ML
from attacks import AdversarialAttacks
from settings import global_setting, training_setting
from util.MakeModel import MakeModel
from util.Arguments import Args
from util.ModelCreation import ModelCreation
from util.DataCreation import DataCreation
from DatasetManager import DatasetManager
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import warnings
import pickle
import os
dataset_name = 'bpic2015_1_f2'
cls_method = 'LR'

"""Experiments."""
# PACKAGES AND FUNCTIONS
# packages from https://github.com/irhete/predictive-monitoring-benchmark/blob/master/experiments/experiments.py

# selfmade
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# to explicitly raise an error with a stack trace to easier debug which operation might have created the invalid values
torch.autograd.set_detect_anomaly(True)
# PARAMETERS
cls_encoding = "agg"
encoding = ['agg']
encoding_dict = {"agg": ["agg"]}
# Datasets dictionary
dataset_ref_to_datasets = {
    # "bpic2011": ["bpic2011_f%s" % formula for formula in range(2, 4)],
    # "bpic2015": ["bpic2015_%s_f2" % municipality for municipality in range(1, 2)],
    # "production": ["production"],
    # "bpic2012": ["bpic2012_accepted","bpic2012_cancelled",'bpic2012_declined'],
    # "bpic2017": ["bpic2017_accepted","bpic2017_cancelled","bpic2017_refused"],
}
classifiers = ['LR', 'XGB', 'RF']
seed = global_setting['seed']
train_ratio = global_setting['train_ratio']
params_dir = global_setting['params_dir']
path = global_setting['models']
labels = ['regular', 'deviant']
attack_types = ['all_event', 'last_event']
clip = training_setting["clip"]
best_manifold_path = global_setting['best_manifolds']
results_dir = global_setting['results_dir'] + '/' + dataset_name+'/'+cls_method
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

arguments = Args(dataset_name)
print('Dataset:', dataset_name)
print('Classifier', cls_method)
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
if "bpic2012" in dataset_name:
    data = data[data['Activity'].str.startswith('W')].copy()
cls_encoder_args, min_prefix_length, max_prefix_length, activity_col, resource_col = arguments.extract_args(data, dataset_manager)
print('prefix lengths', min_prefix_length, 'until', max_prefix_length)
datacreator = DataCreation(dataset_manager, dataset_name, cls_method, cls_encoding)
cat_cols = [activity_col, resource_col]
no_cols_list = []
for i in cat_cols:
    _, _, _, no_cols = datacreator.create_indexes(i, data)
    no_cols_list.append(no_cols)
vocab_size = [no_cols_list[0], no_cols_list[1]]
cols = [cat_cols[0], cat_cols[1], cls_encoder_args['case_id_col'], 'label', 'event_nr', 'prefix_nr']
payload_values = {key: list(data[key].unique()) for key in cat_cols}
train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
# prefix generation of train and test data
dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
dt_train_prefixes_original = dt_train_prefixes.copy()
dt_test_prefixes_original = dt_test_prefixes.copy()
nr_events_original = list(dataset_manager.get_prefix_lengths(dt_test_prefixes))
test_y_original = dataset_manager.get_label_numeric(dt_test_prefixes)
train_y_original = dataset_manager.get_label_numeric(dt_train_prefixes)
dt_train_prefixes, dt_test_prefixes = dt_train_prefixes[cols], dt_test_prefixes[cols]
# CLASSIFICATION
feature_combiner, scaler, dt_train_named = datacreator.transform_data_train(dt_train_prefixes, train_y_original, encoding_dict, cls_encoder_args, cat_cols)
dt_test_named = datacreator.transform_data_test(feature_combiner, scaler, dt_test_prefixes)
dt_train_named_original, dt_test_named_original = dt_train_named.copy(), dt_test_named.copy()
# create original model
optimal_params_filename = os.path.join(params_dir, "optimal_params_%s_%s.pickle" % (cls_method, dataset_name))
args = arguments.params_args(optimal_params_filename)
modelmaker = MakeModel(cls_method, args)
# load original model
filename = os.path.join(path, "model_%s_%s.pickle" % (cls_method, dataset_name))
cls = pickle.load(open(filename, 'rb'))
print('auc no adversarial attack', roc_auc_score(test_y_original, cls.predict(dt_test_named)))

# Adversarial attack (A1)
attack_type = 'last_event'
attack_col = 'Activity'
print('attack', attack_type, attack_col)
attack = AdversarialAttacks(max_prefix_length, payload_values, datacreator, feature_combiner, scaler, dataset_manager)

adversarial_prefixes = attack.create_adversarial_dt_named(attack_type, attack_col, dt_train_prefixes, cls)

y = datacreator.get_label_numeric_adversarial(adversarial_prefixes)
dt_train = datacreator.transform_data_test(feature_combiner, scaler, adversarial_prefixes)
predictions = cls.predict(dt_train)
print('adversarial predictions:', roc_auc_score(y, predictions))
dt_train_prefixes2 = dt_train_prefixes_original[cols]
# 50% original and 50% adversarial training traces
dt_train_prefixes2 = dt_train_prefixes_original.copy()
adversarial_train_prefixes_A1_act = dt_train_prefixes2.append(adversarial_prefixes)
train_y_A1_act = datacreator.get_label_numeric_adversarial(adversarial_train_prefixes_A1_act)
dt_train_named_A1_act = datacreator.transform_data_test(feature_combiner, scaler, adversarial_train_prefixes_A1_act)
cls_A1_act = modelmaker.model_maker(dt_train_named_A1_act, train_y_A1_act)

adversarial_prefixes_test_A1_act, dt_test_named_A1_act, nr_events_A1_act, test_y_A1_act = attack.create_adversarial_dt_named(attack_type, attack_col, dt_test_prefixes, cls)
manifold_creator = Manifold_ML(dataset_name, scaler, feature_combiner, dataset_manager, datacreator, min_prefix_length, max_prefix_length, activity_col, resource_col, cat_cols, cols, vocab_size, payload_values)
cls_manifold_A1_act, on_manifold_test_named_A1_act, test_y_manifold_A1_act, nr_events_manifold_A1_act = manifold_creator.create_manifold_dataset(train, dt_train_prefixes, adversarial_prefixes_test_A1_act, adversarial_prefixes_test_A1_act, modelmaker)
print('kijk hier nu:', roc_auc_score(test_y_A1_act, cls.predict(dt_test_named_A1_act)))
assert len(on_manifold_test_named_A1_act) == len(dt_test_named_A1_act)

# Adversarial attack A1 (Resource)
########ADVERSARIAL attack##########
if dataset_name in ['bpic2017_accepted', 'bpic2017_cancelled', 'bpic2017_refused', 'bpic2015_1_f2', 'bpic2015_2_f2', 'bpic2015_3_f2', 'bpic2015_4_f2', 'bpic2015_5_f2']:
    attack_col = 'org:resource'
else:
    attack_col = 'Resource'
print('attack', attack_type, attack_col)
attack = AdversarialAttacks(max_prefix_length, payload_values, datacreator, feature_combiner, scaler, dataset_manager)
adversarial_prefixes = attack.create_adversarial_dt_named(attack_type, attack_col, dt_train_prefixes, cls)
dt_train_prefixes2 = dt_train_prefixes_original[cols]
# 50% original and 50% adversarial training traces
dt_train_prefixes2 = dt_train_prefixes_original.copy()
adversarial_train_prefixes_A1_res = dt_train_prefixes2.append(adversarial_prefixes)
train_y_A1_res = datacreator.get_label_numeric_adversarial(adversarial_train_prefixes_A1_res)
dt_train_named_A1_res = datacreator.transform_data_test(feature_combiner, scaler, adversarial_train_prefixes_A1_res)
cls_A1_res = modelmaker.model_maker(dt_train_named_A1_res, train_y_A1_res)
adversarial_prefixes_test_A1_res, dt_test_named_A1_res, nr_events_A1_res, test_y_A1_res = attack.create_adversarial_dt_named(attack_type, attack_col, dt_test_prefixes, cls)
cls_manifold_A1_res, on_manifold_test_named_A1_res, test_y_manifold_A1_res, nr_events_manifold_A1_res = manifold_creator.create_manifold_dataset(train, dt_train_prefixes, adversarial_prefixes, adversarial_prefixes_test_A1_res, modelmaker)

# Adversarial attack A2
attack_type = 'all_event'
attack_col = 'Activity'
print('attack', attack_type, attack_col)
attack = AdversarialAttacks(max_prefix_length, payload_values, datacreator, feature_combiner, scaler, dataset_manager)
adversarial_prefixes = attack.create_adversarial_dt_named(attack_type, attack_col, dt_train_prefixes, cls)
dt_train_prefixes2 = dt_train_prefixes_original[cols]
# 50% original and 50% adversarial training traces
dt_train_prefixes2 = dt_train_prefixes_original.copy()
adversarial_train_prefixes_A2_act = dt_train_prefixes2.append(adversarial_prefixes)
train_y_A2_act = datacreator.get_label_numeric_adversarial(adversarial_train_prefixes_A2_act)
dt_train_named_A2_act = datacreator.transform_data_test(feature_combiner, scaler, adversarial_train_prefixes_A2_act)
cls_A2_act = modelmaker.model_maker(dt_train_named_A2_act, train_y_A2_act)
adversarial_prefixes_test_A2_act, dt_test_named_A2_act, nr_events_A2_act, test_y_A2_act = attack.create_adversarial_dt_named(attack_type, attack_col, dt_test_prefixes, cls)
cls_manifold_A2_act, on_manifold_test_named_A2_act, test_y_manifold_A2_act, nr_events_manifold_A2_act = manifold_creator.create_manifold_dataset(train, dt_train_prefixes, adversarial_prefixes, adversarial_prefixes_test_A2_act, modelmaker)

# Adversarial attack A2 (Resource)
attack_type = 'all_event'
if dataset_name in ['bpic2017_accepted', 'bpic2017_cancelled', 'bpic2017_refused', 'bpic2015_1_f2', 'bpic2015_2_f2', 'bpic2015_3_f2', 'bpic2015_4_f2', 'bpic2015_5_f2']:
    attack_col = 'org:resource'
else:
    attack_col = 'Resource'
print('attack', attack_type, attack_col)
attack = AdversarialAttacks(max_prefix_length, payload_values, datacreator, feature_combiner, scaler, dataset_manager)
adversarial_prefixes = attack.create_adversarial_dt_named(attack_type, attack_col, dt_train_prefixes, cls)
dt_train_prefixes2 = dt_train_prefixes_original[cols]
# 50% original and 50% adversarial training traces
dt_train_prefixes2 = dt_train_prefixes_original.copy()
adversarial_train_prefixes_A2_res = dt_train_prefixes2.append(adversarial_prefixes)
train_y_A2_res = datacreator.get_label_numeric_adversarial(adversarial_train_prefixes_A2_res)
dt_train_named_A2_res = datacreator.transform_data_test(feature_combiner, scaler, adversarial_train_prefixes_A2_res)
cls_A2_res = modelmaker.model_maker(dt_train_named_A2_res, train_y_A2_res)
adversarial_prefixes_test_A2_res, dt_test_named_A2_res, nr_events_A2_res, test_y_A2_res = attack.create_adversarial_dt_named(attack_type, attack_col, dt_test_prefixes, cls)
cls_manifold_A2_res, on_manifold_test_named_A2_res, test_y_manifold_A2_res, nr_events_manifold_A2_res = manifold_creator.create_manifold_dataset(train, dt_train_prefixes, adversarial_prefixes, adversarial_prefixes_test_A2_res, modelmaker)

classifiers_trained = [cls, cls_A1_act, cls_A2_act, cls_A1_res, cls_A2_res, cls_manifold_A1_act, cls_manifold_A2_act, cls_manifold_A1_res, cls_manifold_A2_res]

for cls_name in classifiers_trained:
    cls_string = [tpl[0] for tpl in filter(lambda x: cls_name is x[1], globals().items())][0]
    print(cls_string)

    # NORMAL
    predictions = cls_name.predict(dt_test_named_original)
    events = nr_events_original.copy()
    test_y = test_y_original.copy()
    test_string = [tpl[0] for tpl in filter(lambda x: test_y_original is x[1], globals().items())][0]
    method_name = cls_string + '_' + test_string
    print(method_name)
    outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    with open(outfile1, 'w') as fout:
        fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
        dt_results = pd.DataFrame({"actual": test_y, "predicted": predictions, "events": events})
        for events, group in dt_results.groupby("events"):
            if len(set(group.actual)) < 2:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1, "auc", np.nan))
            else:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1,
                                                       "auc", roc_auc_score(group.actual, group.predicted)))
        fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(dt_results.actual, dt_results.predicted)))

    # A1 ACT
    predictions = cls_name.predict(dt_test_named_A1_act)
    events = nr_events_A1_act.copy()
    test_y = test_y_A1_act.copy()
    test_string = [tpl[0] for tpl in filter(lambda x: test_y_A1_act is x[1], globals().items())][0]
    method_name = cls_string + '_' + test_string
    print(method_name)
    print(roc_auc_score(test_y, predictions))
    outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    with open(outfile1, 'w') as fout:
        fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
        dt_results = pd.DataFrame({"actual": test_y, "predicted": predictions, "events": events})
        for events, group in dt_results.groupby("events"):
            if len(set(group.actual)) < 2:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1, "auc", np.nan))
            else:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1,
                                                       "auc", roc_auc_score(group.actual, group.predicted)))
        fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(dt_results.actual, dt_results.predicted)))

    # A1 RES
    predictions = cls_name.predict(dt_test_named_A1_res)
    events = nr_events_A1_res.copy()
    test_y = test_y_A1_res.copy()
    test_string = [tpl[0] for tpl in filter(lambda x: test_y_A1_res is x[1], globals().items())][0]
    method_name = cls_string + '_' + test_string
    print(method_name)
    outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    with open(outfile1, 'w') as fout:
        fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
        dt_results = pd.DataFrame({"actual": test_y, "predicted": predictions, "events": events})
        for events, group in dt_results.groupby("events"):
            if len(set(group.actual)) < 2:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1, "auc", np.nan))
            else:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1,
                                                       "auc", roc_auc_score(group.actual, group.predicted)))
        fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(dt_results.actual, dt_results.predicted)))

    # A2 ACT
    predictions = cls_name.predict(dt_test_named_A2_act)
    events = nr_events_A2_act.copy()
    test_y = test_y_A2_act.copy()
    test_string = [tpl[0] for tpl in filter(lambda x: test_y_A2_act is x[1], globals().items())][0]
    method_name = cls_string + '_' + test_string
    print(method_name)
    outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    with open(outfile1, 'w') as fout:
        fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
        dt_results = pd.DataFrame({"actual": test_y, "predicted": predictions, "events": events})
        for events, group in dt_results.groupby("events"):
            if len(set(group.actual)) < 2:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1, "auc", np.nan))
            else:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1,
                                                       "auc", roc_auc_score(group.actual, group.predicted)))
        fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(dt_results.actual, dt_results.predicted)))

    # A2 RES
    predictions = cls_name.predict(dt_test_named_A2_res)
    events = nr_events_A2_res.copy()
    test_y = test_y_A2_res.copy()
    test_string = [tpl[0] for tpl in filter(lambda x: test_y_A2_res is x[1], globals().items())][0]
    method_name = cls_string + '_' + test_string
    print(method_name)
    outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    with open(outfile1, 'w') as fout:
        fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
        dt_results = pd.DataFrame({"actual": test_y, "predicted": predictions, "events": events})
        for events, group in dt_results.groupby("events"):
            if len(set(group.actual)) < 2:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1, "auc", np.nan))
            else:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1,
                                                       "auc", roc_auc_score(group.actual, group.predicted)))
        fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(dt_results.actual, dt_results.predicted)))

    # MANIFOLD
    # A1 ACT MANIFOLD
    predictions = cls_name.predict(on_manifold_test_named_A1_act)
    events = nr_events_manifold_A1_act.copy()
    test_y = test_y_manifold_A1_act.copy()
    test_string = [tpl[0] for tpl in filter(lambda x: test_y_manifold_A1_act is x[1], globals().items())][0]
    method_name = cls_string + '_' + test_string
    print(method_name)
    outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    with open(outfile1, 'w') as fout:
        fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
        dt_results = pd.DataFrame({"actual": test_y, "predicted": predictions, "events": events})
        for events, group in dt_results.groupby("events"):
            if len(set(group.actual)) < 2:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1, "auc", np.nan))
            else:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1,
                                                       "auc", roc_auc_score(group.actual, group.predicted)))
        fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(dt_results.actual, dt_results.predicted)))

    # A1 RES MANIFOLD
    predictions = cls_name.predict(on_manifold_test_named_A1_res)
    events = nr_events_manifold_A1_res.copy()
    test_y = test_y_manifold_A1_res.copy()
    test_string = [tpl[0] for tpl in filter(lambda x: test_y_manifold_A1_res is x[1], globals().items())][0]
    method_name = cls_string + '_' + test_string
    print(method_name)
    outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    with open(outfile1, 'w') as fout:
        fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
        dt_results = pd.DataFrame({"actual": test_y, "predicted": predictions, "events": events})
        for events, group in dt_results.groupby("events"):
            if len(set(group.actual)) < 2:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1, "auc", np.nan))
            else:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1,
                                                       "auc", roc_auc_score(group.actual, group.predicted)))
        fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(dt_results.actual, dt_results.predicted)))

    # A2 ACT MANIFOLD
    predictions = cls_name.predict(on_manifold_test_named_A2_act)
    events = nr_events_manifold_A2_act.copy()
    test_y = test_y_manifold_A2_act.copy()
    test_string = [tpl[0] for tpl in filter(lambda x: test_y_manifold_A2_act is x[1], globals().items())][0]
    method_name = cls_string + '_' + test_string
    print(method_name)
    outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    with open(outfile1, 'w') as fout:
        fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
        dt_results = pd.DataFrame({"actual": test_y, "predicted": predictions, "events": events})
        for events, group in dt_results.groupby("events"):
            if len(set(group.actual)) < 2:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1, "auc", np.nan))
            else:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1,
                                                       "auc", roc_auc_score(group.actual, group.predicted)))
        fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(dt_results.actual, dt_results.predicted)))

    # A2 RES MANIFOLD
    predictions = cls_name.predict(on_manifold_test_named_A2_res)
    events = nr_events_manifold_A2_res.copy()
    test_y = test_y_manifold_A2_res.copy()
    test_string = [tpl[0] for tpl in filter(lambda x: test_y_manifold_A2_res is x[1], globals().items())][0]
    method_name = cls_string + '_' + test_string
    print(method_name)
    outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    with open(outfile1, 'w') as fout:
        fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
        dt_results = pd.DataFrame({"actual": test_y, "predicted": predictions, "events": events})
        for events, group in dt_results.groupby("events"):
            if len(set(group.actual)) < 2:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1, "auc", np.nan))
            else:
                fout.write("%s;%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, events, -1,
                                                       "auc", roc_auc_score(group.actual, group.predicted)))
        fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(dt_results.actual, dt_results.predicted)))
