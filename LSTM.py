# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 10:59:47 2023

@author: u0138175
"""
import torch.nn as nn
import os
import torch
import wandb
import logging
import numpy as np
logging.getLogger().setLevel(logging.INFO)
from settings import global_setting, training_setting
clip = training_setting["clip"]
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

import torch.nn.utils.rnn as rnn_utils

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout, lstm_size, max_length):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.lstm_dropout = dropout
        self.lstm_size= lstm_size
        self.embed_size = embed_size
        self.embed_act = nn.Embedding(self.vocab_size[0], self.embed_size)
        self.embed_res = nn.Embedding(self.vocab_size[1], self.embed_size)
        self.lstm = nn.LSTM(2*self.embed_size, self.embed_size, dropout = self.lstm_dropout, num_layers=self.lstm_size, batch_first=True)
        self.final_output = nn.Linear(self.embed_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x_act, x_res):
        batch_size = x_act.size(0)  # Get the batch size
        #hidden_state = torch.zeros(self.lstm_size, batch_size, self.embed_size).to(self.device)
        #cell_state = torch.zeros(self.lstm_size, batch_size, self.embed_size).to(self.device)

        x_act_embed_enc = self.embed_act(x_act).to(self.device)
        x_res_embed_enc = self.embed_res(x_res).to(self.device)
        x_embed_enc = torch.cat([x_act_embed_enc, x_res_embed_enc], dim=2)
        l1_out, _ = self.lstm(x_embed_enc)
        output = l1_out[:, -1, :]
        output = self.final_output(output)
        output = torch.sigmoid(output)
        return output

class LSTMModel:      
    def __init__(self, embed_size, dropout, lstm_size, optimizer_name, batch_size, learning_rate, vocab_size, max_prefix_length, dataset_name, cls_method):
        self.embed_size = embed_size
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.lstm_size = lstm_size
        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_prefix_length = max_prefix_length
        self.path = global_setting['models']
        self.dir_path = self.path+'/adversarial_models/'+dataset_name+'/'+cls_method
        if not os.path.exists(os.path.join(self.dir_path)):
            os.makedirs(os.path.join(self.dir_path))

    def make_LSTM_model(self, activity_train, resource_train, activity_val, resource_val, train_y, val_y):
        seed_worker = torch.manual_seed(22)
        dataset = torch.utils.data.TensorDataset(activity_train, resource_train)
        dataset = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, drop_last=True, worker_init_fn=seed_worker)
        # checkpoint saver
        checkpoint_saver = CheckpointSaver_adversarial(dirpath=self.dir_path, decreasing=True, top_n=1)
        criterion = nn.BCELoss()
        cls_adv = Model(self.vocab_size, self.embed_size, self.dropout, self.lstm_size, self.max_prefix_length).to(self.device)
        if self.optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(cls_adv.parameters(), lr=0.0001)
        elif self.optimizer_name == 'Nadam':
            optimizer = torch.optim.NAdam(cls_adv.parameters(), lr=0.0001)
        print('training')
        early_stop_patience = 11
        best_auc = 0.5
        lr_reducer = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=False,
                                threshold=0.0001, cooldown=0, min_lr=0)
        epochs = 100
        for epoch in range(epochs):
          print("Epoch: ", epoch)
          for i, (data_act, data_res) in enumerate(dataset, 0): # loop over the data, and jump with step = bptt.
                cls_adv.train()
                data_act = data_act.to(self.device)
                data_res = data_res.to(self.device)
                y_ = cls_adv(data_act,data_res).to(self.device) 
                train_batch = torch.tensor(train_y[i*self.batch_size:(i+1)*self.batch_size], dtype= torch.float32).to(self.device)
                train_batch = train_batch[:, None].to(self.device)
                train_loss = criterion(y_,train_batch).to(self.device)
                train_loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                optimizer.zero_grad()
          with torch.no_grad():
                cls_adv.eval()
                print('testing')
                pred = cls_adv(activity_val, resource_val).squeeze(-1).to(self.device)
                pred = pred.cpu()
                validation_auc = roc_auc_score(torch.tensor(val_y , dtype= torch.float32).to(self.device).cpu(), pred)
                lr_reducer.step(validation_auc)
                # Log evaluation metrics to WandB
                print('validation_auc',validation_auc)

                if validation_auc > best_auc:
                      best_auc = validation_auc
          if epoch > early_stop_patience and validation_auc <= best_auc:
            print('best auc', best_auc)
            print('validation_auc', validation_auc)
            print("Early stopping triggered.")
            break
        return cls_adv



class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf

    def __call__(self, model, epoch, metric_val, learning_rate, latent_size, optimizer, batch_size):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + '_' + str(learning_rate) + '_' + str(latent_size) + '_' + str(optimizer) + '_' + str(batch_size) + f'_epoch{epoch}.pt')
        save = metric_val < self.best_metric_val if self.decreasing else metric_val > self.best_metric_val
        if save:
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.log_artifact(f'model-ckpt'+'_' + str(learning_rate) + '_' + str(latent_size) + '_' + str(optimizer) + '_' + str(batch_size) + f'-epoch-{epoch}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths) > self.top_n:
            self.cleanup()

    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)

    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]


class CheckpointSaver_adversarial:
    def __init__(self, dirpath, decreasing=True, top_n=1):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf

    def __call__(self, model, epoch, metric_val, learning_rate, latent_size, optimizer, batch_size):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + '_' + str(learning_rate) + '_' + str(latent_size) + '_' + str(optimizer) + '_' + str(batch_size) + f'_epoch{epoch}.pt')
        save = metric_val < self.best_metric_val if self.decreasing else metric_val > self.best_metric_val
        if save:
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths) > self.top_n:
            self.cleanup()

    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]
