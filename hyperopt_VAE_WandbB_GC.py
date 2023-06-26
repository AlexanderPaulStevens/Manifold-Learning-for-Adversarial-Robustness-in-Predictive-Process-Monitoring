import logging
import yaml
import hyperopt
from hyperopt.pyll.base import scope
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
import torch
import warnings
import pandas as pd
import numpy as np
import pickle
import sys
import os
import wandb

# packages from https://github.com/Khamies/LSTM-Variational-AutoEncoder/tree/50476dd3bfe146bf8f4a74a205b78fb142e99423
from VAE import LSTM_VAE, CheckpointSaver
from loss import VAE_Loss
from settings import global_setting, model_setting, training_setting
from train import Trainer

# packages from https://github.com/irhete/predictive-monitoring-benchmark/blob/master/experiments/experiments.py
from DatasetManager import DatasetManager

sys.path.append(os.getcwd())

# user-specified packages
from util.Arguments import Args
from util.DataCreation import DataCreation

warnings.simplefilter(action='ignore', category=FutureWarning)
# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
# to explicitly raise an error with a stack trace to easier debug which operation might have created the invalid values
torch.autograd.set_detect_anomaly(True)
# hyperopt
# set logging
logging.getLogger().setLevel(logging.INFO)
##################################################################
# PARAMETERS

dataset_ref_to_datasets = {
    "production": ["production"],
    "bpic2015": ["bpic2015_%s_f2" % (municipality) for municipality in range(1, 6)],
    "bpic2012": ["bpic2012_accepted", "bpic2012_cancelled", "bpic2012_declined"],
    "hospital_billing": ["hospital_billing_%s" % suffix for suffix in [2, 3]],
    "traffic_fines": ["traffic_fines_%s" % formula for formula in range(1, 2)],
    # "bpic2017": ["bpic2017_accepted","bpic2017_cancelled","bpic2017_refused"],
    # "sepsis_cases": ["sepsis_cases_2","sepsis_cases_4"],
    # "bpic2011": ["bpic2011_f%s"%formula for formula in range(2,4)],
}

labels = ['regular', 'deviant']

dataset_name = 'production'
label = 'deviant'

path = 'manifolds/wandb/'+dataset_name+'_'+label
print('path', path)
os.environ['WANDB_DIR'] = path
os.environ["WANDB_SILENT"] = "true"

seed = global_setting['seed']
epochs = training_setting['epochs']
n_splits = global_setting['n_splits']
max_evals = global_setting['max_evals']
train_ratio = global_setting['train_ratio']
clip = training_setting["clip"]
embed_size = training_setting["embed_size"]

project_name = 'Bayes_VAE'
dataset_group = 'VAE'
name = dataset_group + '_' + dataset_name + '_' + label

print('Dataset:', dataset_name)
print('label:', label)
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
arguments = Args(dataset_name)
cls_encoder_args, min_prefix_length, max_prefix_length, activity_col, resource_col = arguments.extract_args(data, dataset_manager)
print(data.nunique())

cat_cols = [activity_col, resource_col]
cols = ['Case ID', 'label', 'case_length'] + cat_cols
datacreator = DataCreation(dataset_manager, dataset_name)
no_cols_list = []
for i in cat_cols:
    _, _, _, no_cols = datacreator.create_indexes(i, data)
    no_cols_list.append(no_cols)
vocab_size = [no_cols_list[0], no_cols_list[1]]
# split into training and test
train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")

# you don't need to do that for the test data, as the prepare inputs is only fitted on the training data
# prepare chunks for CV
dt_prefixes = []
for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
    # generate data where each prefix is a separate instance
    dt_prefixes.append(dataset_manager.generate_prefix_data(test_chunk, min_prefix_length, max_prefix_length))
del train

#######WANDB################
with open("config/VAE_sweep_production.yaml", 'r') as stream:
    sweep_config = yaml.safe_load(stream)

sweep_config['name'] = name
sweep_id = wandb.sweep(sweep_config, project=project_name, entity="adversarial_robustness")

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config, project=project_name, group=name, save_code=False) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        for cv_iter in range(n_splits):
            dt_test_prefixes = dt_prefixes[cv_iter]
            dt_train_prefixes = pd.DataFrame()
            for cv_train_iter in range(n_splits):
                if cv_train_iter != cv_iter:
                    dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)

        #######################METHODOLOGY#################################
        dt_train_prefixes = dt_train_prefixes[cols].copy()
        dt_test_prefixes = dt_test_prefixes[cols].copy()

        dt_train_prefixes = dt_train_prefixes[dt_train_prefixes['label'] == label]
        dt_test_prefixes = dt_test_prefixes[dt_test_prefixes['label'] == label]

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

        ###################MODEL ARCHITECTURE#################################################
        # create the input layers and embeddings
        input_size = no_cols_list
        dataset = torch.utils.data.TensorDataset(activity_train, resource_train)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
        del activity_train, resource_train
        dataset_test = torch.utils.data.TensorDataset(activity_test, resource_test)
        dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=True, drop_last=True)
        del activity_test, resource_test
        model = LSTM_VAE(vocab_size=vocab_size, embed_size=embed_size, hidden_size=config["latent_size"], latent_size=config["latent_size"]).to(device)
        print(model)
        if config['optimizer'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'])
        elif config['optimizer'] == 'Nadam':
            optimizer = torch.optim.NAdam(model.parameters(), lr=config['learning_rate'])
        elif config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        elif config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
        Loss = VAE_Loss()
        trainer = Trainer(dataset, dataset_test, model, Loss, optimizer)
        # Epochs
        train_losses = []
        test_losses = []
        total_elbo = []
        total_KL = []
        total_recon = []
        # checkpoint saver
        path = 'manifolds/'+dataset_name+'_'+label
        checkpoint_saver = CheckpointSaver(dirpath=path, decreasing=True, top_n=1)
        total_validation_losses = []
        for epoch in range(epochs):
            print("Epoch: ", epoch)
            print("Training.......")
            train_losses = trainer.train(train_losses, epoch, config["batch_size"], clip)
            print('testing........')
            test_losses = trainer.test(test_losses, epoch, config["batch_size"])
            elbo_loss = list(map(lambda x: x[0], test_losses))
            elbo_loss = np.mean([tensor for tensor in elbo_loss])
            print('loss', elbo_loss)
            total_validation_losses.append(elbo_loss)
            validation_loss = np.amin(total_validation_losses)
            checkpoint_saver(model, epoch, elbo_loss, config['learning_rate'], config["latent_size"], config['optimizer'], config["batch_size"])
            wandb.log({"best_val_loss": elbo_loss})

        # Get the lowest validation loss of the training epochs
wandb.agent(sweep_id, function=train, count=max_evals, project=project_name, entity="adversarial_robustness")
