# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:44:24 2023

@author: u0138175
"""
import os
import pickle
from settings import global_setting, model_setting, training_setting


class Args:
    """preprocessing for the machine learning models."""

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def extract_args(self, data, dataset_manager):
        cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                            'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                            'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                            'fillna': True}
        # determine min and max (truncated) prefix lengths
        min_prefix_length = 1
        #max_prefix_length = maxlen = int(train['event_nr'].max())
        if "traffic_fines" in self.dataset_name:
            max_prefix_length = 10
        elif "bpic2017" in self.dataset_name:
            max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
        else:
            max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
        if "bpic2012" in self.dataset_name:
            min_prefix_length = 15
            max_prefix_length = 20

        if "production" in self.dataset_name:
            min_prefix_length = 1
            max_prefix_length = 15

        elif "bpic2015" in self.dataset_name:
            max_prefix_length = 20

        activity_col = [x for x in cls_encoder_args['dynamic_cat_cols'] if 'Activity' in x][0]
        if self.dataset_name in ['bpic2017_accepted', 'bpic2017_cancelled', 'bpic2017_refused', 'bpic2015_1_f2', 'bpic2015_2_f2', 'bpic2015_3_f2', 'bpic2015_4_f2', 'bpic2015_5_f2']:
            resource_col = 'org:resource'
        else:
            resource_col = 'Resource'
        return cls_encoder_args, min_prefix_length, max_prefix_length, activity_col, resource_col

    def params_args(self, optimal_params_filename):
        with open(optimal_params_filename, "rb") as fin:
            args = pickle.load(fin)
        return args
