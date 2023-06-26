# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:58:20 2023

@author: u0138175
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion
import EncoderFactory
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_string_dtype
from collections import OrderedDict
# torch packages
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable
from torchvision.utils import make_grid
from sklearn.preprocessing import MinMaxScaler
from itertools import chain
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc

class DataCreation:
    """preprocessing for the machine learning models."""

    def __init__(self, dataset_manager, dataset_name, max_prefix_length, cls_method=None, cls_encoding=None):
        self.dataset_manager = dataset_manager
        self.dataset_name = dataset_name
        self.cls_method = cls_method
        self.cls_encoding = cls_encoding
        self.train_ratio = 0.8
        self.max_prefix_length = max_prefix_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        """
        self.dt_train_prefixes = dt_train_prefixes
        self.dt_test_prefixes = dt_test_prefixes
        self.cls_encoder_args = cls_encoder_args
        self.cols = cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        """

    def create_case_lengths(self, data, case_id_col, activity_col):
        data['case_length'] = data.groupby(case_id_col)[activity_col].transform(len)
        return data

    def prefix_test(self,train, test, cat_cols, cols, min_length, max_length):
        dt_train_prefixes = self.dataset_manager.generate_prefix_data(train, min_length, max_length)
        dt_test_prefixes = self.dataset_manager.generate_prefix_data(test, min_length, max_length)
        train_cat_cols, test_cat_cols, _ = self.prepare_inputs(dt_train_prefixes.loc[:, cat_cols], dt_test_prefixes.loc[:, cat_cols])
        dt_train_prefixes[cat_cols] = train_cat_cols
        dt_test_prefixes[cat_cols] = test_cat_cols
        dt_train_prefixes = dt_train_prefixes[cols].copy()
        dt_test_prefixes = dt_test_prefixes[cols].copy()
        test_y = self.dataset_manager.get_label_numeric(dt_test_prefixes)
        train_y = self.dataset_manager.get_label_numeric(dt_train_prefixes)
        return dt_train_prefixes, dt_test_prefixes, test_y, train_y
    
    def groupby_caseID(self, data, cols, col):
        groups = data[cols].groupby('Case ID', as_index=True)
        case_ids = groups.groups.keys()
        ans = [torch.tensor(list(y[col])) for _, y in groups]
        label_lists = [y['label'].iloc[0] for _, y in groups]
        return ans, label_lists, list(case_ids)


    def groupby_pad(self,prefixes, cols, activity_col, resource_col):
        ans_act, label, cases = self.groupby_caseID(prefixes, cols, activity_col)
        ans_res, _, _ = self.groupby_caseID(prefixes, cols, resource_col)
        ######ACTIVITY AND RESOURCE COL########
        activity = self.pad_data(ans_act)
        resource = self.pad_data(ans_res)
        activity = activity.to(self.device)
        resource = resource.to(self.device)
        return activity, resource, label, cases

    def get_label_numeric_adversarial(self, adversarial_prefixes):
        adversarial_prefixes.loc[(adversarial_prefixes['label'] == 1), 'label'] = 'deviant'
        adversarial_prefixes.loc[(adversarial_prefixes['label'] == 0), 'label'] = 'regular'
        label = self.dataset_manager.get_label_numeric(adversarial_prefixes)
        adversarial_prefixes.loc[(adversarial_prefixes['label'] == 'deviant'), 'label'] = 1
        adversarial_prefixes.loc[(adversarial_prefixes['label'] == 'regular'), 'label'] = 0
        return label

    def prepare_inputs(self, X_train, X_test):
        global ce
        ce = ColumnEncoder()
        X_train, X_test = X_train.astype(str), X_test.astype(str)
        X_train_enc = ce.fit_transform(X_train)
        X_test_enc = ce.transform(X_test)
        return X_train_enc, X_test_enc, ce

    def pad_data(self, data):
        data[0] = nn.ConstantPad1d((0, self.max_prefix_length - data[0].shape[0]), 0)(data[0])
        padding = pad_sequence(data, batch_first=True, padding_value=0)
        return padding

    def create_index(self, log_df, column):
        """Creates an idx for a categorical attribute.
        Args:
            log_df: dataframe.
            column: column name.
        Returns:
            index of a categorical attribute pairs.
        """
        temp_list = temp_list = log_df[log_df[column] != 'none'][[column]].values.tolist()  # remove all 'none' values from the index
        subsec_set = {(x[0]) for x in temp_list}
        subsec_set = sorted(list(subsec_set))
        alias = dict()
        for i, _ in enumerate(subsec_set):
            alias[subsec_set[i]] = i
        # reorder by the index value
        alias = {k: v for k, v in sorted(alias.items(), key=lambda item: item[1])}
        return alias

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='int')[y]

    def create_indexes(self, i, data):
        dyn_index = self.create_index(data, i)
        index_dyn = {v: k for k, v in dyn_index.items()}
        dyn_weights = self.to_categorical(sorted(index_dyn.keys()), len(dyn_index))
        no_cols = len(data.groupby([i]))
        return dyn_weights,  dyn_index, index_dyn, no_cols

    def transform_data_train(self, dt_train, y_train, encoding_dict, cls_encoder_args, attack_cols):
        # feature combiner and columns
        methods = encoding_dict[self.cls_encoding]
        feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(
            method, case_id_col=cls_encoder_args['case_id_col'], static_cat_cols=list(), static_num_cols=list(), dynamic_cat_cols=attack_cols,
            dynamic_num_cols=list(), fillna=False, max_events=None, activity_col=attack_cols[0], resource_col=attack_cols[1], timestamp_col=None,
            scale_model=None)) for method in methods])
        feature_combiner.fit(dt_train, y_train)

        # transform train dataset and add the column names back to the dataframe
        train_named = feature_combiner.transform(dt_train)
        train_named = pd.DataFrame(train_named)
        names = feature_combiner.get_feature_names_out()
        train_named.columns = names
        scaler = MinMaxScaler()
        train_named_scaled = scaler.fit_transform(train_named)
        train_named = pd.DataFrame(train_named_scaled, columns=train_named.columns)
        return feature_combiner, scaler, train_named

    def transform_data_test(self, feature_combiner, scaler, dt_test):
        # transform test dataset
        test_named = feature_combiner.transform(dt_test)
        test_named = pd.DataFrame(test_named)
        names = feature_combiner.get_feature_names_out()
        test_named.columns = names
        test_named_scaled = scaler.transform(test_named)
        test_named = pd.DataFrame(test_named_scaled, columns=test_named.columns)

        return test_named
    
    def Find_Optimal_Cutoff(self,target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------     
        list type, with optimal cutoff value
            
        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

        return list(roc_t['threshold']) 

# https://towardsdatascience.com/using-neural-networks-with-embedding-layers-to-encode-high-cardinality-categorical-variables-c1b872033ba2


class ColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = None
        self.maps = dict()

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            # encode value x of col via dict entry self.maps[col][x]+1 if present, otherwise 0
            X_copy.loc[:, col] = X_copy.loc[:, col].apply(lambda x: self.maps[col].get(x, -1)+1)
        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            values = list(self.maps[col].keys())
            # find value in ordered list and map out of range values to None
            X_copy.loc[:, col] = [values[i-1] if 0 < i <= len(values) else None for i in X_copy[col]]
        return X_copy

    def fit(self, X, y=None):
        # only apply to string type columns
        self.columns = [col for col in X.columns if is_string_dtype(X[col])]
        for col in self.columns:
            self.maps[col] = OrderedDict({value: num for num, value in enumerate(sorted(set(X[col])))})
        return self
