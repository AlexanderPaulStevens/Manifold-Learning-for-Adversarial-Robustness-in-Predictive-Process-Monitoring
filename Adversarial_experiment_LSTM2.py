from attacks import AdversarialAttacks_LSTM
import torch
from manifold import AdversarialAttacks_manifold_LSTM
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
print(len(data))
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
cls = lstmmodel.make_LSTM_model(activity_train, resource_train, activity_test, resource_test, train_y,test_y)
#cls = Model(embed_size, hidden_size, dropout, vocab_size, max_prefix_length)
#cls.load_state_dict(checkpoint)

cls.eval()
pred = cls(activity_test, resource_test).detach().numpy()
print('auc no adversarial attacks', roc_auc_score(test_y, pred)) 

# Adversarial attack (A1)
attack_type = 'last_event'
attack_col = 'Activity'
print('attack', attack_type, attack_col)

attack = AdversarialAttacks_LSTM(max_prefix_length, payload_values, datacreator, cols, cat_cols, activity_col, resource_col, no_cols_list)
attack_manifold_LSTM = AdversarialAttacks_manifold_LSTM(dataset_name,dataset_manager, min_prefix_length, max_prefix_length, activity_col, resource_col, cat_cols, cols, payload_values, vocab_size, datacreator, attack, no_cols_list)

print("TRAINING DATA ADVERSARIAL")
adversarial_prefixes,_ = attack.create_adversarial_dt_named_LSTM_train(attack_type, attack_col, dt_train_prefixes, cls)
adversarial_prefixes_A1_act = pd.concat([dt_train_prefixes, adversarial_prefixes])
activity_train_A1_act, resource_train_A1_act, train_y_A1_act ,_ = datacreator.groupby_pad(adversarial_prefixes_A1_act, cols, activity_col, resource_col)
cls_A1_act = lstmmodel.make_LSTM_model(activity_train_A1_act, resource_train_A1_act, activity_val, resource_val, train_y_A1_act, val_y)

print("TESTING DATA ADVERSARIAL")
adversarial_prefixes_test_A1_act = attack.create_adversarial_dt_named_LSTM_test(attack_type, attack_col, dt_test_prefixes, cls)
nr_events_A1_act = list(dataset_manager.get_prefix_lengths(adversarial_prefixes_test_A1_act))
activity_test_A1_act, resource_test_A1_act, test_y_A1_act ,_ = datacreator.groupby_pad(adversarial_prefixes_test_A1_act, cols, activity_col, resource_col)
cls_A1_act.eval()
pred = cls_A1_act(activity_test_A1_act, resource_test_A1_act).detach().numpy()
print('auc A1 adversarial attacks', roc_auc_score(test_y_A1_act, pred)) 

# on-manifold
print("TRAINING DATA MANIFOLD")
manifold_prefixes_train = attack_manifold_LSTM.create_adversarial_dt_named_LSTM_train(attack_type, attack_col, train, dt_train_prefixes, cls)
manifold_train_prefixes_A1_act = pd.concat([dt_train_prefixes, manifold_prefixes_train])
activity_train_manifold_A1_act, resource_train_manifold_A1_act, train_y_manifold_A1_act,_ = datacreator.groupby_pad(manifold_train_prefixes_A1_act, cols, activity_col, resource_col)
cls_manifold_A1_act = lstmmodel.make_LSTM_model(activity_train_manifold_A1_act, resource_train_manifold_A1_act, activity_val, resource_val, train_y_manifold_A1_act, val_y)

print("TESTING DATA MANIFOLD")
manifold_prefixes_test = attack_manifold_LSTM.create_adversarial_dt_named_LSTM_test(attack_type, attack_col, train, dt_test_prefixes, cls)
activity_test_manifold_A1_act, resource_test_manifold_A1_act, test_y_manifold_A1_act,_ = datacreator.groupby_pad(manifold_prefixes_test, cols, activity_col, resource_col)
nr_events_manifold_A1_act = list(dataset_manager.get_prefix_lengths(manifold_prefixes_test))
cls_manifold_A1_act.eval()
pred = cls_manifold_A1_act(activity_test_manifold_A1_act, resource_test_manifold_A1_act).detach().numpy()
print('auc A1 manifold adversarial attacks', roc_auc_score(test_y_manifold_A1_act, pred)) 
attack_type = 'last_event'
if dataset_name in ['bpic2017_accepted', 'bpic2017_cancelled', 'bpic2017_refused', 'bpic2015_1_f2', 'bpic2015_2_f2', 'bpic2015_3_f2', 'bpic2015_4_f2', 'bpic2015_5_f2']:
    attack_col = 'org:resource'
else:
    attack_col = 'Resource'
print('attack', attack_type, attack_col)

print("TRAINING DATA ADVERSARIAL")
adversarial_prefixes,_ = attack.create_adversarial_dt_named_LSTM_train(attack_type, attack_col, dt_train_prefixes, cls)
adversarial_prefixes_A1_res = pd.concat([dt_train_prefixes, adversarial_prefixes])
activity_train_A1_res, resource_train_A1_res, train_y_A1_res,_ = datacreator.groupby_pad(adversarial_prefixes_A1_res, cols, activity_col, resource_col)
cls_A1_res = lstmmodel.make_LSTM_model(activity_train_A1_res, resource_train_A1_res,  activity_val, resource_val, train_y_A1_res, val_y)

print("TESTING DATA ADVERSARIAL")
adversarial_prefixes_test_A1_res = attack.create_adversarial_dt_named_LSTM_test(attack_type, attack_col, dt_test_prefixes, cls)
nr_events_A1_res = list(dataset_manager.get_prefix_lengths(adversarial_prefixes_test_A1_res))
activity_test_A1_res, resource_test_A1_res, test_y_A1_res,_ = datacreator.groupby_pad(adversarial_prefixes_test_A1_res, cols, activity_col, resource_col)

# on-manifold
print("TRAINING DATA MANIFOLD")
manifold_prefixes_train = attack_manifold_LSTM.create_adversarial_dt_named_LSTM_train(attack_type, attack_col, train, dt_train_prefixes, cls)
manifold_train_prefixes_A1_res = pd.concat([dt_train_prefixes, manifold_prefixes_train])
activity_train_manifold_A1_res, resource_train_manifold_A1_res, train_y_manifold_A1_res,_ = datacreator.groupby_pad(manifold_train_prefixes_A1_res, cols, activity_col, resource_col)
cls_manifold_A1_res = lstmmodel.make_LSTM_model(activity_train_manifold_A1_res, resource_train_manifold_A1_res, activity_val, resource_val, train_y_manifold_A1_res, val_y)

print("TESTING DATA MANIFOLD")
manifold_prefixes_test = attack_manifold_LSTM.create_adversarial_dt_named_LSTM_test(attack_type, attack_col, train, dt_test_prefixes, cls)
activity_test_manifold_A1_res, resource_test_manifold_A1_res, test_y_manifold_A1_res,_ = datacreator.groupby_pad(manifold_prefixes_test, cols, activity_col, resource_col)
nr_events_manifold_A1_res = list(dataset_manager.get_prefix_lengths(manifold_prefixes_test))

# Adversarial attack (A2)
attack_type = 'all_event'
attack_col = 'Activity'
print('attack', attack_type, attack_col)

print("TRAINING DATA ADVERSARIAL")
adversarial_prefixes,_ = attack.create_adversarial_dt_named_LSTM_train(attack_type, attack_col, dt_train_prefixes, cls)
adversarial_prefixes_A2_act = pd.concat([dt_train_prefixes, adversarial_prefixes])
activity_train_A2_act, resource_train_A2_act, train_y_A2_act,_ = datacreator.groupby_pad(adversarial_prefixes_A2_act, cols, activity_col, resource_col)
cls_A2_act = lstmmodel.make_LSTM_model(activity_train_A2_act, resource_train_A2_act, activity_val, resource_val, train_y_A2_act, val_y)

print("TESTING DATA ADVERSARIAL")
adversarial_prefixes_test_A2_act = attack.create_adversarial_dt_named_LSTM_test(attack_type, attack_col, dt_test_prefixes, cls)
nr_events_A2_act = list(dataset_manager.get_prefix_lengths(adversarial_prefixes_test_A2_act))
activity_test_A2_act, resource_test_A2_act, test_y_A2_act,_ = datacreator.groupby_pad(adversarial_prefixes_test_A2_act, cols, activity_col, resource_col)

# on-manifold
print("TRAINING DATA MANIFOLD")
manifold_prefixes_train = attack_manifold_LSTM.create_adversarial_dt_named_LSTM_train(attack_type, attack_col, train, dt_train_prefixes, cls)
manifold_train_prefixes_A2_act = pd.concat([dt_train_prefixes, manifold_prefixes_train])
activity_train_manifold_A2_act, resource_train_manifold_A2_act, train_y_manifold_A2_act,_ = datacreator.groupby_pad(manifold_train_prefixes_A2_act, cols, activity_col, resource_col)
print('cls manifold A2')
cls_manifold_A2_act = lstmmodel.make_LSTM_model(activity_train_manifold_A2_act, resource_train_manifold_A2_act, activity_val, resource_val, train_y_manifold_A2_act, val_y)

print("TESTING DATA MANIFOLD")
manifold_prefixes_test = attack_manifold_LSTM.create_adversarial_dt_named_LSTM_test(attack_type, attack_col, train, dt_test_prefixes, cls)
activity_test_manifold_A2_act, resource_test_manifold_A2_act, test_y_manifold_A2_act,_ = datacreator.groupby_pad(manifold_prefixes_test, cols, activity_col, resource_col)
nr_events_manifold_A2_act = list(dataset_manager.get_prefix_lengths(manifold_prefixes_test))
cls_manifold_A2_act.eval()
pred = cls_manifold_A2_act(activity_test_manifold_A2_act, resource_test_manifold_A2_act).detach().numpy()
print('auc A2 manifold adversarial attacks', roc_auc_score(test_y_manifold_A2_act, pred)) 
# Adversarial attack (A2)
attack_type = 'all_event'
if dataset_name in ['bpic2017_accepted', 'bpic2017_cancelled', 'bpic2017_refused', 'bpic2015_1_f2', 'bpic2015_2_f2', 'bpic2015_3_f2', 'bpic2015_4_f2', 'bpic2015_5_f2']:
    attack_col = 'org:resource'
else:
    attack_col = 'Resource'

print("TRAINING DATA ADVERSARIAL")
adversarial_prefixes,_ = attack.create_adversarial_dt_named_LSTM_train(attack_type, attack_col, dt_train_prefixes, cls)
adversarial_prefixes_A2_res = pd.concat([dt_train_prefixes, adversarial_prefixes])
activity_train_A2_res, resource_train_A2_res, train_y_A2_res,_ = datacreator.groupby_pad(adversarial_prefixes_A2_res, cols, activity_col, resource_col)
cls_A2_res = lstmmodel.make_LSTM_model(activity_train_A2_res, resource_train_A2_res, activity_val, resource_val, train_y_A2_res, val_y)

print("TESTING DATA ADVERSARIAL")
adversarial_prefixes_test_A2_res = attack.create_adversarial_dt_named_LSTM_test(attack_type, attack_col, dt_test_prefixes, cls)
nr_events_A2_res = list(dataset_manager.get_prefix_lengths(adversarial_prefixes_test_A2_res))

activity_test_A2_res, resource_test_A2_res, test_y_A2_res,_ = datacreator.groupby_pad(adversarial_prefixes_test_A2_res, cols, activity_col, resource_col)

# on-manifold
print("TRAINING DATA MANIFOLD")
manifold_prefixes_train = attack_manifold_LSTM.create_adversarial_dt_named_LSTM_train(attack_type, attack_col, train, dt_train_prefixes, cls)
manifold_train_prefixes_A2_res = pd.concat([dt_train_prefixes, manifold_prefixes_train])
activity_train_manifold_A2_res, resource_train_manifold_A2_res, train_y_manifold_A2_res,_ = datacreator.groupby_pad(manifold_train_prefixes_A2_res, cols, activity_col, resource_col)
cls_manifold_A2_res = lstmmodel.make_LSTM_model(activity_train_manifold_A2_res, resource_train_manifold_A2_res, activity_val, resource_val, train_y_manifold_A2_res, val_y)

print("TESTING DATA MANIFOLD")
manifold_prefixes_test = attack_manifold_LSTM.create_adversarial_dt_named_LSTM_test(attack_type, attack_col, train, dt_test_prefixes, cls)
activity_test_manifold_A2_res, resource_test_manifold_A2_res, test_y_manifold_A2_res,_ = datacreator.groupby_pad(manifold_prefixes_test, cols, activity_col, resource_col)
nr_events_manifold_A2_res = list(dataset_manager.get_prefix_lengths(manifold_prefixes_test))

classifiers_trained = [cls, cls_A1_act,cls_A1_res,cls_A2_act, cls_A2_res, cls_manifold_A1_act,cls_manifold_A1_res, cls_manifold_A2_act, cls_manifold_A2_res]
done = True
for cls_name in classifiers_trained:
    if done==True:
        cls_name.eval()
        pred = cls_name(activity_test, resource_test).squeeze(-1).detach().numpy()
        events = nr_events_original.copy()
        test_y = test_y_original.copy()
        test_string = [tpl[0] for tpl in filter(lambda x: test_y_original is x[1], globals().items())][0]
        cls_string = [tpl[0] for tpl in filter(lambda x: cls_name is x[1], globals().items())][0]
        method_name = cls_string + '_' + test_string
        print(method_name)
        outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
        with open(outfile1, 'w') as fout:
            fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
            fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(test_y, pred)))
            
        # A1 ACT
        cls_name.eval()
       
        pred = cls_name(activity_test_A1_act, resource_test_A1_act).squeeze(-1).detach().numpy()
        events = nr_events_A1_act.copy()
        test_y = test_y_A1_act.copy()
        test_string = [tpl[0] for tpl in filter(lambda x: test_y_A1_act is x[1], globals().items())][0]
        cls_string = [tpl[0] for tpl in filter(lambda x: cls_name is x[1], globals().items())][0]
        method_name = cls_string + '_' + test_string
        print(method_name)
        print('auc', cls_string, test_string, roc_auc_score(test_y, pred))
        outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
        with open(outfile1, 'w') as fout:
            fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
            fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(test_y, pred)))

        # A1 RES
        cls_name.eval()
       
        pred = cls_name(activity_test_A1_res, resource_test_A1_res).squeeze(-1).detach().numpy()
        events = nr_events_A1_res.copy()
        test_y = test_y_A1_res.copy()
        test_string = [tpl[0] for tpl in filter(lambda x: test_y_A1_res is x[1], globals().items())][0]
        cls_string = [tpl[0] for tpl in filter(lambda x: cls_name is x[1], globals().items())][0]
        method_name = cls_string + '_' + test_string
        print(method_name)
        print('auc', cls_string, test_string, roc_auc_score(test_y, pred))
        outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
        with open(outfile1, 'w') as fout:
            fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
            fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(test_y, pred)))
        
        # A2 ACT
        cls_name.eval()
        
        pred = cls_name(activity_test_A2_act, resource_test_A2_act).squeeze(-1).detach().numpy()
        events = nr_events_A2_act.copy()
        test_y = test_y_A2_act.copy()
        test_string = [tpl[0] for tpl in filter(lambda x: test_y_A2_act is x[1], globals().items())][0]
        cls_string = [tpl[0] for tpl in filter(lambda x: cls_name is x[1], globals().items())][0]
        method_name = cls_string + '_' + test_string
        print(method_name)
        print('auc', cls_string, test_string, roc_auc_score(test_y, pred))
        outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
        with open(outfile1, 'w') as fout:
            fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
            fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(test_y, pred)))

        # A2 RES
        cls_name.eval()
       
        pred = cls_name(activity_test_A2_res, resource_test_A2_res).squeeze(-1).detach().numpy()
        events = nr_events_A2_res.copy()
        test_y = test_y_A2_res.copy()
        test_string = [tpl[0] for tpl in filter(lambda x: test_y_A2_res is x[1], globals().items())][0]
        cls_string = [tpl[0] for tpl in filter(lambda x: cls_name is x[1], globals().items())][0]
        method_name = cls_string + '_' + test_string
        print(method_name)
        print('auc', cls_string, test_string, roc_auc_score(test_y, pred))
        outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
        with open(outfile1, 'w') as fout:
            fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
            fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(test_y, pred)))
        
        # MANIFOLD
        # A1 ACT MANIFOLD
        cls_name.eval()
      
        pred = cls_name(activity_test_manifold_A1_act, resource_test_manifold_A1_act).squeeze(-1).detach().numpy()
        events = nr_events_manifold_A1_act.copy()
        test_y = test_y_manifold_A1_act.copy()
        test_string = [tpl[0] for tpl in filter(lambda x: test_y_manifold_A1_act is x[1], globals().items())][0]
        cls_string = [tpl[0] for tpl in filter(lambda x: cls_name is x[1], globals().items())][0]
        method_name = cls_string + '_' + test_string
        print(method_name)
        print('auc', cls_string, test_string, roc_auc_score(test_y, pred))
        outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
        with open(outfile1, 'w') as fout:
            fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
            fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(test_y, pred)))

        # A1 RES MANIFOLD
        cls_name.eval()
      
        pred = cls_name(activity_test_manifold_A1_res, resource_test_manifold_A1_res).squeeze(-1).detach().numpy()
        events = nr_events_manifold_A1_res.copy()
        test_y = test_y_manifold_A1_res.copy()
        test_string = [tpl[0] for tpl in filter(lambda x: test_y_manifold_A1_res is x[1], globals().items())][0]
        cls_string = [tpl[0] for tpl in filter(lambda x: cls_name is x[1], globals().items())][0]
        method_name = cls_string + '_' + test_string
        print(method_name)
        print('auc', cls_string, test_string, roc_auc_score(test_y, pred))
        outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
        with open(outfile1, 'w') as fout:
            fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
            fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(test_y, pred)))
        
        # A2 ACT MANIFOLD
        cls_name.eval()
       
        pred = cls_name(activity_test_manifold_A2_act, resource_test_manifold_A2_act).squeeze(-1).detach().numpy()
        events = nr_events_manifold_A2_act.copy()
        test_y = test_y_manifold_A2_act.copy()
        test_string = [tpl[0] for tpl in filter(lambda x: test_y_manifold_A2_act is x[1], globals().items())][0]
        cls_string = [tpl[0] for tpl in filter(lambda x: cls_name is x[1], globals().items())][0]
        method_name = cls_string + '_' + test_string
        print(method_name)
        print('auc', cls_string, test_string, roc_auc_score(test_y, pred))
        outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
        with open(outfile1, 'w') as fout:
            fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
            fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(test_y, pred)))

        # A2 RES MANIFOLD
        cls_name.eval()
       
        pred = cls_name(activity_test_manifold_A2_res, resource_test_manifold_A2_res).squeeze(-1).detach().numpy()
        events = nr_events_manifold_A2_res.copy()
        test_y = test_y_manifold_A2_res.copy()
        test_string = [tpl[0] for tpl in filter(lambda x: test_y_manifold_A2_res is x[1], globals().items())][0]
        cls_string = [tpl[0] for tpl in filter(lambda x: cls_name is x[1], globals().items())][0]
        method_name = cls_string + '_' + test_string
        print(method_name)
        print('auc', cls_string, test_string, roc_auc_score(test_y, pred))
        outfile1 = os.path.join(results_dir, "performance_results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
        with open(outfile1, 'w') as fout:
            fout.write("%s;%s;%s;%s;%s;%s\n" % ("dataset", "method", "cls", "events", "metric", "score"))
            fout.write("%s;%s;%s;%s;%s;%s\n" % (dataset_name, method_name, cls_method, -1, "auc", roc_auc_score(test_y, pred)))
