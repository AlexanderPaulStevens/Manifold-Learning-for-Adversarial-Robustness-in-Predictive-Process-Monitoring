# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 19:47:56 2023

@author: u0138175
"""
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import torch.nn as nn
import torch
from settings import global_setting
seed = global_setting['seed']

class MakeModel:
    """preprocessing for the machine learning models."""

    def __init__(self, cls_method, args):
        self.cls_method = cls_method
        self.args = args
    
    def save_original_model(self,filename, dt_train_named, train_y):
        if self.cls_method == 'LR':
            cls = LogisticRegression(C=2**self.args['C'], solver='saga', penalty="l1", n_jobs=-1, random_state=seed, max_iter=10000)
        elif self.cls_method =="XGB":
            cls = xgb.XGBClassifier(objective='binary:logistic',
                                                n_estimators=500,
                                                learning_rate= self.args['learning_rate'],
                                                subsample=self.args['subsample'],
                                                max_depth=int(self.args['max_depth']),
                                                colsample_bytree=self.args['colsample_bytree'],
                                                min_child_weight=int(self.args['min_child_weight']),
                                                n_jobs=-1,
                                                seed=seed)

        elif self.cls_method == "RF":
            cls = RandomForestClassifier(n_estimators=500,
                                          max_features=self.args['max_features'],
                                          n_jobs=-1,
                                          random_state=seed)
        print('model fitting')
        cls.fit(dt_train_named, train_y)
        # save the model to disk
        pickle.dump(cls, open(filename, 'wb'))
        
    def fit_model(self,dt_train_named, train_y):
        if self.cls_method == 'LR':
            cls = LogisticRegression(C=2**self.args['C'], solver='saga', penalty="l1", n_jobs=-1, random_state=seed)
        elif self.cls_method =="XGB":
            cls = xgb.XGBClassifier(objective='binary:logistic',
                                                    n_estimators=500,
                                                    learning_rate= self.args['learning_rate'],
                                                    subsample=self.args['subsample'],
                                                    max_depth=int(self.args['max_depth']),
                                                    colsample_bytree=self.args['colsample_bytree'],
                                                    min_child_weight=int(self.args['min_child_weight']),
                                                    n_jobs=-1,
                                                    seed=seed)

        elif self.cls_method == "RF":
            cls = RandomForestClassifier(n_estimators=500,
                                              max_features=self.args['max_features'],
                                              n_jobs=-1,
                                              random_state=seed)
        print('model fitting')
        cls.fit(dt_train_named, train_y)
        return cls

    def model_maker(self,adversarial_named, train_y):
        cls = self.fit_model(adversarial_named, train_y)
        return cls