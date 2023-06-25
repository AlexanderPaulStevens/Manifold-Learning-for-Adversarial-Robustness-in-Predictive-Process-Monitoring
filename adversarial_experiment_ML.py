import torch
from manifold import Manifold, AdversarialAttacks_manifold
from attacks import AdversarialAttacks
from settings import global_setting, training_setting
from util.MakeModel import MakeModel
from util.Arguments import Args
from util.DataCreation import DataCreation
from DatasetManager import DatasetManager
from sklearn.metrics import roc_auc_score
import pandas as pd
import warnings
import pickle
import os
os.chdir('G:\My Drive\CurrentWork\Manifold\AdversarialRobustnessGeneralization')
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# to explicitly raise an error with a stack trace to easier debug which operation might have created the invalid values
torch.autograd.set_detect_anomaly(True)
dataset_name = 'production'
cls_method = 'LR'

"""Experiments."""
# PARAMETERS
cls_encoding = "agg"
encoding_dict = {"agg": ["agg"]}
# Datasets dictionary
dataset_ref_to_datasets = {
    # "bpic2015": ["bpic2015_%s_f2" % municipality for municipality in range(1, 6)],
    # "production": ["production"],
    # "bpic2012": ["bpic2012_accepted","bpic2012_cancelled",'bpic2012_declined'],
}
classifiers = ['LR', 'RF', 'XGB']
labels = ['regular', 'deviant']
attack_types = ['all_event', 'last_event']
seed = global_setting['seed']
train_ratio = global_setting['train_ratio']
clip = training_setting["clip"]
params_dir = global_setting['params_dir']
path = global_setting['models']
best_manifold_path = global_setting['best_manifolds']
results_dir = global_setting['results_dir'] + '/' + dataset_name+'/'+cls_method
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

arguments = Args(dataset_name)
print('Dataset:', dataset_name)
print('Classifier', cls_method)
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
# if "bpic2012" in dataset_name:
#    data = data[data['Activity'].str.startswith('W')].copy()
cls_encoder_args, min_prefix_length, max_prefix_length, activity_col, resource_col = arguments.extract_args(data, dataset_manager)
print('prefix lengths', min_prefix_length, 'until', max_prefix_length)
datacreator = DataCreation(dataset_manager, dataset_name, max_prefix_length, cls_method, cls_encoding)
cat_cols = [activity_col, resource_col]
no_cols_list = []
cols = [cat_cols[0], cat_cols[1], cls_encoder_args['case_id_col'], 'label', 'event_nr', 'prefix_nr']
train, test = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
for i in cat_cols:
    _, _, _, no_cols = datacreator.create_indexes(i, train)
    no_cols_list.append(no_cols)
vocab_size = [no_cols_list[0]+1, no_cols_list[1]+1]
payload_values = {key: list(train[key].unique()) for key in cat_cols}

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
print('auc no adversarial attack', roc_auc_score(test_y_original, cls.predict_proba(dt_test_named)[:,-1]))

def perform_attack(attack_type, attack_col, train_prefixes, test_prefixes, cls, attack, attack_manifold, datacreator, modelmaker, results_cls, results_test):
    print('attack', attack_type, attack_col)
    # Adversarial - Training Data
    adversarial_prefixes_train, _ = attack.create_adversarial_dt_named_train(attack_type, attack_col, train_prefixes, cls)
    adversarial_train_prefixes = pd.concat([train_prefixes, adversarial_prefixes_train])
    train_y = datacreator.get_label_numeric_adversarial(adversarial_train_prefixes)
    dt_train_named = datacreator.transform_data_test(feature_combiner, scaler, adversarial_train_prefixes)
    cls_adversarial = modelmaker.model_maker(dt_train_named, train_y)

    # Adversarial - Testing Data
    adversarial_prefixes_test = attack.create_adversarial_dt_named_test(attack_type, attack_col, test_prefixes, cls)
    test_y = datacreator.get_label_numeric_adversarial(adversarial_prefixes_test)
    dt_test_named = datacreator.transform_data_test(feature_combiner, scaler, adversarial_prefixes_test)

    # Save Adversarial Results to Dictionary
    results_cls['adv_cls_' + attack_type + '_' + attack_col] = cls_adversarial
    results_test['adv_test_' + attack_type + '_' + attack_col] = (dt_test_named, test_y)
        
    # On-Manifold - Training Data
    manifold_prefixes_train = attack_manifold.create_adversarial_dt_named_train(attack_type, attack_col, train_prefixes, test_prefixes, cls)
    manifold_train_prefixes = pd.concat([train_prefixes, manifold_prefixes_train])
    train_y_manifold = datacreator.get_label_numeric_adversarial(manifold_train_prefixes)
    dt_train_named_manifold = datacreator.transform_data_test(feature_combiner, scaler, manifold_train_prefixes)
    cls_manifold = modelmaker.model_maker(dt_train_named_manifold, train_y_manifold)

    # On-Manifold - Testing Data
    manifold_prefixes_test = attack_manifold.create_adversarial_dt_named_test(attack_type, attack_col, train_prefixes, test_prefixes, cls)
    test_y_manifold = datacreator.get_label_numeric_adversarial(manifold_prefixes_test)
    dt_test_named_manifold = datacreator.transform_data_test(feature_combiner, scaler, manifold_prefixes_test)

    # Save On-Manifold Results to Dictionary
    results_cls['manifold_cls_' + attack_type + '_' + attack_col] = cls_manifold
    results_test['manifold_test_' + attack_type + '_' + attack_col] = (dt_test_named_manifold, test_y_manifold)
    print('saved to dictionary')

results_cls = {}
results_test = {}

manifold_creator = Manifold(dataset_name, scaler, feature_combiner, dataset_manager, datacreator, min_prefix_length, max_prefix_length, activity_col, resource_col, cat_cols, cols, vocab_size, payload_values)
attack = AdversarialAttacks(max_prefix_length, payload_values, datacreator, cols, cat_cols, activity_col, resource_col, no_cols_list, feature_combiner, scaler, dataset_manager)
attack_manifold = AdversarialAttacks_manifold(dataset_name,dataset_manager, min_prefix_length, max_prefix_length, activity_col, resource_col, cat_cols, cols, payload_values, vocab_size, datacreator, attack)

# Adversarial attack A1 (Activity)
attack_type = 'last_event'
attack_col = 'Activity'

perform_attack(attack_type, attack_col, dt_train_prefixes, dt_test_prefixes, cls, attack, attack_manifold, datacreator, modelmaker, results_cls, results_test)

# Adversarial attack A1 (Resource)
########ADVERSARIAL attack##########
if dataset_name in ['bpic2017_accepted', 'bpic2017_cancelled', 'bpic2017_refused', 'bpic2015_1_f2', 'bpic2015_2_f2', 'bpic2015_3_f2', 'bpic2015_4_f2', 'bpic2015_5_f2']:
    attack_col = 'org:resource'
else:
    attack_col = 'Resource'

perform_attack(attack_type, attack_col, dt_train_prefixes, dt_test_prefixes, cls, attack, attack_manifold, datacreator, modelmaker, results_cls, results_test)

# Adversarial attack A2
attack_type = 'all_event'
attack_col = 'Activity'

perform_attack(attack_type, attack_col, dt_train_prefixes, dt_test_prefixes, cls, attack, attack_manifold, datacreator, modelmaker, results_cls, results_test)

# Adversarial attack A2 (Resource)
attack_type = 'all_event'
if dataset_name in ['bpic2017_accepted', 'bpic2017_cancelled', 'bpic2017_refused', 'bpic2015_1_f2', 'bpic2015_2_f2', 'bpic2015_3_f2', 'bpic2015_4_f2', 'bpic2015_5_f2']:
    attack_col = 'org:resource'
else:
    attack_col = 'Resource'

perform_attack(attack_type, attack_col, dt_train_prefixes, dt_test_prefixes, cls, attack, attack_manifold, datacreator, modelmaker, results_cls, results_test)

# Save Adversarial Results to Dictionary
results_cls['orig_cls'] = cls
results_test['orig_test'] = (dt_test_named, test_y_original)

def save_auc_scores(results_cls, results_test, results_dir, cls_method, dataset_name):
        for key, value in results_cls.items():
                print('cls key', key)
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
                        dt_test_named, test_y = value
                        predictions = cls.predict_proba(dt_test_named)[:, -1]
                        auc = roc_auc_score(test_y, predictions)
                        outfile = os.path.join(results_dir, f"performance_results_{cls_method}_{dataset_name}_{cls_string}_{test_string}.csv")
                        print(outfile)
                        with open(outfile, 'w') as fout:
                                        fout.write("dataset;method;cls;events;metric;score\n")
                                        fout.write(f"{dataset_name};{cls_string};{test_string};-1;auc;{auc}\n")

# Call the function to save the AUC scores
save_auc_scores(results_cls, results_test, results_dir, cls_method, dataset_name)
