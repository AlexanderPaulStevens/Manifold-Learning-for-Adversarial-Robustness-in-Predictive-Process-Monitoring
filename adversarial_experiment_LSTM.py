from attacks import AdversarialAttacks_LSTM
import torch
from manifold import AdversarialAttacks_manifold_LSTM, Manifold
from settings import global_setting, training_setting
from util.Arguments import Args
from util.DataCreation import DataCreation
from DatasetManager import DatasetManager
from sklearn.metrics import roc_auc_score
from LSTM import LSTMModel
import numpy as np
import pandas as pd
import warnings
import os
import random

from util.DataCreation import DataCreation
os.chdir('G:\My Drive\CurrentWork\Manifold\AdversarialRobustnessGeneralization')

dataset_name = 'bpic2012_accepted'
cls_method = 'LSTM'

"""Experiments."""
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# to explicitly raise an error with a stack trace to easier debug which operation might have created the invalid values
torch.autograd.set_detect_anomaly(True)
# PARAMETERS
# Datasets dictionary
dataset_ref_to_datasets = {
    # "bpic2011": ["bpic2011_f%s" % formula for formula in range(2, 4)],
    # "bpic2015": ["bpic2015_%s_f2" % municipality for municipality in range(1, 2)],
    # "production": ["production"],
    # "bpic2012": ["bpic2012_accepted","bpic2012_cancelled",'bpic2012_declined'],
    # "bpic2017": ["bpic2017_accepted","bpic2017_cancelled","bpic2017_refused"],
}
cls_encoding = 'embed'
attack_types = ['all_event', 'last_event']
clip = training_setting["clip"]
seed = global_setting['seed']
path = global_setting['models']
train_ratio = global_setting['train_ratio']
params_dir = global_setting['params_dir_DL']
best_manifold_path = global_setting['best_manifolds']
torch.manual_seed(22)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

best_LSTM_path = 'G:\My Drive\CurrentWork\Manifold\AdversarialRobustnessGeneralization/best_LSTMs'
results_dir = global_setting['results_dir'] + '/' + dataset_name+'/'+cls_method
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

# START
arguments = Args(dataset_name)
print('Dataset:', dataset_name)
print('Classifier', cls_method)
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
cls_encoder_args, min_prefix_length, max_prefix_length, activity_col, resource_col = arguments.extract_args(data, dataset_manager)
print('prefix lengths', min_prefix_length, 'until', max_prefix_length)
datacreator = DataCreation(dataset_manager, dataset_name, max_prefix_length, cls_method, cls_encoding)
cat_cols = [activity_col, resource_col]
no_cols_list = []
cols = [cat_cols[0], cat_cols[1], cls_encoder_args['case_id_col'], 'label', 'event_nr', 'prefix_nr']
train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
_, val = dataset_manager.split_data_strict(train, train_ratio, split="temporal")
for i in cat_cols:
    _, _, _, no_cols = datacreator.create_indexes(i, train)
    no_cols_list.append(no_cols)
vocab_size = [no_cols_list[0]+1, no_cols_list[1]+1]
# you don't need to do that for the test data, as the prepare inputs is only fitted on the training data
# prepare chunks for CV
dt_prefixes = []

# prefix generation of test data
dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
dt_val_prefixes = dataset_manager.generate_prefix_data(val, min_prefix_length, max_prefix_length)
dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

train_cat_cols, val_cat_cols, _ = datacreator.prepare_inputs(dt_train_prefixes.loc[:, cat_cols], dt_val_prefixes.loc[:, cat_cols])
train_cat_cols, test_cat_cols, _ = datacreator.prepare_inputs(dt_train_prefixes.loc[:, cat_cols], dt_test_prefixes.loc[:, cat_cols])

dt_train_prefixes[cat_cols] = train_cat_cols
dt_val_prefixes[cat_cols] = val_cat_cols
dt_test_prefixes[cat_cols] = test_cat_cols

payload_values = {key: list(dt_train_prefixes[key].unique()) for key in cat_cols}
del train_cat_cols, test_cat_cols, val_cat_cols

nr_events_original = list(dataset_manager.get_prefix_lengths(dt_test_prefixes))
test_y_original = dataset_manager.get_label_numeric(dt_test_prefixes)
train_y_original = dataset_manager.get_label_numeric(dt_train_prefixes)
val_y_original = dataset_manager.get_label_numeric(dt_val_prefixes)
dt_train_prefixes = dt_train_prefixes[cols].copy()
dt_test_prefixes = dt_test_prefixes[cols].copy()
dt_val_prefixes = dt_val_prefixes[cols].copy()
train_y = dataset_manager.get_label_numeric(dt_train_prefixes)
test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
val_y = dataset_manager.get_label_numeric(dt_val_prefixes)

dt_train_prefixes.loc[(dt_train_prefixes['label'] == 'deviant'), 'label'] = 1
dt_train_prefixes.loc[(dt_train_prefixes['label'] == 'regular'), 'label'] = 0
dt_test_prefixes.loc[(dt_test_prefixes['label'] == 'deviant'), 'label'] = 1
dt_test_prefixes.loc[(dt_test_prefixes['label'] == 'regular'), 'label'] = 0
dt_val_prefixes.loc[(dt_val_prefixes['label'] == 'deviant'), 'label'] = 1
dt_val_prefixes.loc[(dt_val_prefixes['label'] == 'regular'), 'label'] = 0

#pd.set_option('display.max_rows', 100)

# groupby case ID
activity_train, resource_train, activity_label, cases = datacreator.groupby_pad(dt_train_prefixes, cols, activity_col, resource_col)
activity_test, resource_test, _, _ = datacreator.groupby_pad(dt_test_prefixes, cols, activity_col, resource_col)
activity_val, resource_val, _, _ = datacreator.groupby_pad(dt_val_prefixes, cols, activity_col, resource_col)

###################MODEL ARCHITECTURE#################################################
# create the input layers and embeddings
input_size = no_cols_list
# CLASSIFICATION
path_data_label = best_LSTM_path+'/' + dataset_name + '/'
dir_list = os.listdir(path_data_label)
if 'desktop.ini' in dir_list:
    dir_list.remove('desktop.ini')
#LSTM_name = dir_list[0]
#print(LSTM_name)
#split = LSTM_name.split("_")
#checkpoint = torch.load(path_data_label+'/'+LSTM_name, map_location=torch.device('cpu'))
#learning_rate = float(split[1])
#hidden_size = int(split[2])
#dropout = float(split[3])
#optimizer_name = split[4]
#batch_size = int(split[5])
#embed_size = training_setting['embed_size']
learning_rate = 0.02140

embed_size = 16
optimizer_name='Nadam'
batch_size = 5
lstm_size = 3
dropout = 0.18985

print('embed_size', embed_size)
print('optimizer', optimizer_name)
print('batch_size', batch_size)
print('learning rate', learning_rate)

lstmmodel = LSTMModel(embed_size, dropout, lstm_size, optimizer_name, batch_size, learning_rate, vocab_size, max_prefix_length, dataset_name, cls_method)
cls = lstmmodel.make_LSTM_model(activity_train, resource_train, activity_val, resource_val, train_y,val_y)
#cls = Model(embed_size, hidden_size, dropout, vocab_size, max_prefix_length)
#cls.load_state_dict(checkpoint)

cls.eval()
pred = cls(activity_test, resource_test).detach().numpy()
print('auc no adversarial attacks', roc_auc_score(test_y, pred)) 

results_cls = {}
results_test = {}

attack = AdversarialAttacks_LSTM(train, dt_train_prefixes, dt_test_prefixes, max_prefix_length, payload_values, datacreator, cols, cat_cols, activity_col, resource_col, no_cols_list)
manifold_creator = Manifold(dataset_name, None, None, dataset_manager, datacreator, max_prefix_length, train, dt_train_prefixes, dt_test_prefixes, attack, activity_col, resource_col, cat_cols, cols, vocab_size, payload_values)
attack_manifold = AdversarialAttacks_manifold_LSTM(dataset_name,dataset_manager, max_prefix_length, train, dt_train_prefixes, dt_test_prefixes, activity_col, resource_col, cat_cols, cols, payload_values, vocab_size, datacreator, attack, no_cols_list, manifold_creator)

# Adversarial attack (A1)
attack_type = 'last_event'
attack_col = 'Activity'

results_cls, results_test = manifold_creator.perform_attack_DL(attack_type, attack_col, activity_val, resource_val, val_y, cls, attack_manifold, lstmmodel, results_cls, results_test)

attack_type = 'last_event'
if dataset_name in ['bpic2017_accepted', 'bpic2017_cancelled', 'bpic2017_refused', 'bpic2015_1_f2', 'bpic2015_2_f2', 'bpic2015_3_f2', 'bpic2015_4_f2', 'bpic2015_5_f2']:
    attack_col = 'org:resource'
else:
    attack_col = 'Resource'

results_cls, results_test = manifold_creator.perform_attack_DL(attack_type, attack_col, activity_val, resource_val, val_y, cls, attack_manifold, lstmmodel, results_cls, results_test)

# Adversarial attack (A2)
attack_type = 'all_event'
attack_col = 'Activity'

results_cls, results_test = manifold_creator.perform_attack_DL(attack_type, attack_col, activity_val, resource_val, val_y, cls, attack_manifold, lstmmodel, results_cls, results_test)

# Adversarial attack (A2)
attack_type = 'all_event'
if dataset_name in ['bpic2017_accepted', 'bpic2017_cancelled', 'bpic2017_refused', 'bpic2015_1_f2', 'bpic2015_2_f2', 'bpic2015_3_f2', 'bpic2015_4_f2', 'bpic2015_5_f2']:
    attack_col = 'org:resource'
else:
    attack_col = 'Resource'

results_cls, results_test = manifold_creator.perform_attack_DL(attack_type, attack_col, activity_val, resource_val, val_y, cls, attack_manifold, lstmmodel, results_cls, results_test)

# Save Adversarial Results to Dictionary
results_cls['orig_cls'] = cls
results_test['orig_test'] = (activity_test, resource_test, test_y_original)

def save_auc_scores(results_cls, results_test, results_dir, cls_method, dataset_name):
        for key, value in results_cls.items():
                cls = value
                scenario_name = key.split('_')[0]
                if scenario_name == 'orig':
                        cls_string = 'cls_orig'
                else:
                        attack_type = key.split('_')[2]
                        attack_col = key.split('_')[4]
                        attack_type = 'A1' if attack_type == 'last' else 'A2'
                        attack_col = 'act' if attack_col == 'Activity' else 'res'
                        cls_string = 'cls_' +scenario_name + '_' + attack_type + '_' + attack_col
                for key, value in results_test.items():
                        #results_test['manifold_test_' + attack_type + '_' + attack_col] = (dt_test_named_manifold, test_y_manifold)
                        scenario_name = key.split('_')[0]            
                        if scenario_name == 'orig':
                                test_string = 'test_orig'
                        else:
                                attack_type = key.split('_')[2]
                                attack_col = key.split('_')[4]
                                attack_type = 'A1' if attack_type == 'last' else 'A2'
                                attack_col = 'act' if attack_col == 'Activity' else 'res'
                                test_string = 'test_' +scenario_name + '_' + attack_type + '_' + attack_col
                        activity_test, resource_test, test_y = value
                        cls.eval()
                        predictions = cls(activity_test, resource_test).detach().numpy()
                        auc = roc_auc_score(test_y, predictions)
                        outfile = os.path.join(results_dir, f"performance_results_{cls_method}_{dataset_name}_{cls_string}_{test_string}.csv")
                        print(outfile)
                        with open(outfile, 'w') as fout:
                                        fout.write("dataset;method;cls;events;metric;score\n")
                                        fout.write(f"{dataset_name};{cls_string};{test_string};-1;auc;{auc}\n")

# Call the function to save the AUC scores
save_auc_scores(results_cls, results_test, results_dir, cls_method, dataset_name)