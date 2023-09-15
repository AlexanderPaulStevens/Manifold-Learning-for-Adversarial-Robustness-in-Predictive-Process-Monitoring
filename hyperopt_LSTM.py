import sys
import os

import pandas as pd
sys.path.append(os.getcwd())
import random
import numpy as np
import pandas as pd
# packages from https://github.com/irhete/predictive-monitoring-benchmark/blob/master/experiments/experiments.py
from DatasetManager import DatasetManager
from settings import global_setting, model_setting, training_setting
from LSTM import Model, CheckpointSaver
#user-specified packages
from util.DataCreation import DataCreation
from util.Arguments import Args
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
# to explicitly raise an error with a stack trace to easier debug which operation might have created the invalid values
torch.autograd.set_detect_anomaly(True)
#hyperopt
import hyperopt
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
from hyperopt.pyll.base import scope
# set logging
import logging
logging.getLogger().setLevel(logging.INFO)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
##################################################################

dataset_ref_to_datasets = {
    "production": ["production"],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "bpic2012": ["bpic2012_accepted","bpic2012_cancelled","bpic2012_declined"],
    "hospital_billing": ["hospital_billing_%s"%suffix for suffix in [2,3]],
    "traffic_fines": ["traffic_fines_%s" % formula for formula in range(1, 2)],
    # "bpic2017": ["bpic2017_accepted","bpic2017_cancelled","bpic2017_refused"],
    #"sepsis_cases": ["sepsis_cases_2","sepsis_cases_4"],
    #"bpic2011": ["bpic2011_f%s"%formula for formula in range(2,4)],
    }


def create_and_evaluate_model(args):  
        global trial_nr
        trial_nr += 1
        for cv_iter in range(n_splits):
          dt_test_prefixes_original = dt_prefixes[cv_iter]
          dt_train_prefixes_original = pd.DataFrame()
          for cv_train_iter in range(n_splits): 
              if cv_train_iter != cv_iter:
                  dt_train_prefixes_original = pd.concat([dt_train_prefixes_original, dt_prefixes[cv_train_iter]], axis=0)
        
        preds_all = []
        test_y_all = []

        dt_train_prefixes = dt_test_prefixes_original.copy()
        dt_test_prefixes = dt_test_prefixes_original.copy()
        
        test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
        train_y = dataset_manager.get_label_numeric(dt_train_prefixes)
        test_y_all.extend(test_y)
        dt_train_prefixes = dt_train_prefixes[cols].copy()

        train_cat_cols, test_cat_cols, _ = datacreator.prepare_inputs(dt_train_prefixes.loc[:, cat_cols], dt_test_prefixes.loc[:, cat_cols])
        dt_train_prefixes[cat_cols] = train_cat_cols
        dt_test_prefixes[cat_cols] = test_cat_cols
        payload_values = {key: list(dt_train_prefixes[key].unique()) for key in cat_cols}
        del train_cat_cols, test_cat_cols

        dt_train_prefixes_original, dt_test_prefixes_original, = dt_train_prefixes.copy(), dt_test_prefixes.copy()
        nr_events_original = list(dataset_manager.get_prefix_lengths(dt_test_prefixes))
        test_y_original = dataset_manager.get_label_numeric(dt_test_prefixes)
        train_y_original = dataset_manager.get_label_numeric(dt_train_prefixes)
        dt_train_prefixes = dt_train_prefixes[cols].copy()
        dt_test_prefixes = dt_test_prefixes[cols].copy()
        train_y = dataset_manager.get_label_numeric(dt_train_prefixes)
        test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
        
        # groupby case ID
        ans_train_act = datacreator.groupby_caseID(dt_train_prefixes, cols, activity_col)
        ans_test_act = datacreator.groupby_caseID(dt_test_prefixes, cols, activity_col)
        ans_train_res = datacreator.groupby_caseID(dt_train_prefixes, cols, resource_col)
        ans_test_res = datacreator.groupby_caseID(dt_test_prefixes, cols, resource_col)

        ######ACTIVITY_COL########
        activity_train = datacreator.pad_data(ans_train_act).to(device)
        activity_test = datacreator.pad_data(ans_test_act).to(device)

        # ######RESOURCE COL########
        resource_train = datacreator.pad_data(ans_train_res).to(device)
        resource_test = datacreator.pad_data(ans_test_res).to(device)

        score = 0
        dim = 0        
        batch_size = args['batch_size']
        embed_size = training_setting['embed_size']
        latent_size = args['latent_size']
        learning_rate = args['learning_rate']
        dropout = args['dropout']
        
        l2reg=0.001
        dataset = torch.utils.data.TensorDataset(activity_train, resource_train)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True, worker_init_fn=seed_worker)

        model = Model(embed_size, latent_size, vocab_size, max_prefix_length, dropout).to(device)

        print(model)
        if args['optimizer']=='RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=0.01)
        elif args['optimizer']=='adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
        elif args['optimizer']=='Nadam':
            optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate,weight_decay=0.01)
        print('optimizer', args['optimizer'])
        
        lr_reducer = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False, 
                               threshold=0.0001, cooldown=0, min_lr=0)
        # checkpoint saver
        checkpoint_saver = CheckpointSaver(dirpath=dir_path, decreasing=False, top_n=5)
      
        
        criterion = nn.BCELoss()
        # Define the early stopping patience
        print('training')
        early_start_patience = 15
        early_stop_patience = 60
        best_auc = 9999
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            for i, (data_act, data_res) in enumerate(dataset, 0): # loop over the data, and jump with step = bptt.
                model.train()
                data_act = data_act.to(device)
                data_res = data_res.to(device)
                y_ = model(data_act,data_res).to(device)
                train_batch = torch.tensor(train_y[i*batch_size:(i+1)*batch_size], dtype= torch.float32).to(device)
                train_batch = train_batch[:, None].to(device)
                train_loss = criterion(y_,train_batch).to(device)
                train_loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                optimizer.zero_grad()
          
            with torch.no_grad():
                model.eval()
                #mask = ((activity_test != 0) | (resource_test != 0)).any(dim=1).float()
                print('testing')
                pred = model(activity_test, resource_test).squeeze(-1).to(device)

                validation_loss = criterion(pred, torch.tensor(test_y , dtype= torch.float32).to(device))
                validation_loss = validation_loss.to('cpu').detach().numpy()
                lr_reducer.step(validation_loss)
                # Log evaluation metrics to WandB
                print('validation_loss',validation_loss)
                if epoch > early_start_patience:      
                    checkpoint_saver(model, epoch, validation_loss, learning_rate, latent_size, optimizer, batch_size)
                if validation_loss < best_loss:
                    best_loss = validation_loss
            if epoch > early_stop_patience and validation_loss > best_loss:
                print('best loss', best_loss)
                print('validation loss', validation_loss)
                print("Early stopping triggered.")
                break
            preds_all.extend(pred)
            score += roc_auc_score(test_y_all, preds_all)
            for k, v in args.items():
                fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, k, v, score / n_splits))  
            
            fout_all.write("%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, method_name, 0))   
    
            fout_all.flush()
            return {'loss': validation_loss, 
            'status': STATUS_OK, 
            'model': model, 
            'args': args}
            

dataset_name= 'bpic2015_1_f2'
path = 'params_dir_DL'
print('path',path)

dir_path = path+'/hyper_models/'+dataset_name
# create results directory
if not os.path.exists(os.path.join(dir_path)):
    os.makedirs(os.path.join(dir_path))

seed = global_setting['seed']
epochs = training_setting['epochs']
n_splits = global_setting['n_splits']
max_evals = global_setting['max_evals']
train_ratio = global_setting['train_ratio']
clip = training_setting["clip"]

print('Dataset:', dataset_name)

dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
arguments = Args(dataset_name)
cls_encoder_args, min_prefix_length, max_prefix_length, activity_col, resource_col = arguments.extract_args(data, dataset_manager)
print(data.nunique())
cls_method = 'LSTM'
cls_encoding = 'embed'
method_name = "%s_%s"%("all", cls_encoding) 
datacreator = DataCreation(dataset_manager, dataset_name, cls_method, cls_encoding)
cat_cols = [activity_col, resource_col]
no_cols_list = []
cols = [cat_cols[0], cat_cols[1], cls_encoder_args['case_id_col'], 'label', 'event_nr', 'prefix_nr']

# split into training and test
train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

for i in cat_cols:
    _, _, _, no_cols = datacreator.create_indexes(i, train)
    no_cols_list.append(no_cols)
vocab_size = [no_cols_list[0]+1, no_cols_list[1]+1]   

# prepare chunks for CV
dt_prefixes = []
class_ratios = []
for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
    class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
    # generate data where each prefix is a separate instance
    dt_prefixes.append(dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length))

del train
# set up search space
if cls_method == "LSTM":
    space = {'dropout'  : hp.uniform('dropout',0.01,0.3),
            'latent_size'      : scope.int(hp.quniform('latent_size',8,256,8)),
            'batch_size' :      scope.int(hp.quniform('batch_size',64,256,8)),
            'optimizer': hp.choice('optimizer',['Nadam', 'Adam', 'SGD', 'RMSprop']),
            'learning_rate': hp.uniform('learning_rate',0.0001,0.01)
                 }

# optimize parameters
trial_nr = 1
trials = Trials()
fout_all = open(os.path.join(path, "param_optim_all_trials_%s_%s_%s.csv" % (cls_method, dataset_name, method_name)), "w")
# Set the seed
rstate = random.seed(22)

# Use random.randint to generate random integers
value = random.randint(0, 10)
trials = Trials()
best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=4, trials=trials, rstate = rstate)
fout_all.close()
