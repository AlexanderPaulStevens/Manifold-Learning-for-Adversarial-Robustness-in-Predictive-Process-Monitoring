# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 17:33:28 2023

@author: u0138175
"""
import time
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.linear_model import LogisticRegression

random_state = 22


class ModelCreation:
    def __init__(self, cls_method, args, max_prefix_length, prefix_lengths, nr_events_all):
        self.cls_method = cls_method
        self.args = args
        self.max_prefix_length = max_prefix_length
        self.prefix_lengths = prefix_lengths
        self.nr_events_all = nr_events_all
        """
        self.dt_train_prefixes = dt_train_prefixes
        self.dt_test_prefixes = dt_test_prefixes
        self.cls_encoder_args = cls_encoder_args
        self.maxlen = maxlen
        self.cols = cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        """

    def cls_predictions(self, dt_train_named, dt_train_named2, dt_train_named3, dt_test_named, dt_test_named2, dt_test_named3, train_y, test_y_all):
        preds_all = []
        preds_all2 = []
        preds_all3 = []
        array_of_distances1 = 0
        array_of_distances2 = 0
        if self.cls_method == 'LR':
            cls1 = LogisticRegression(C=2**self.args['C'], solver='saga', penalty="l1", n_jobs=-1, random_state=random_state)
            cls1.fit(dt_train_named, train_y)
            coefmodel = pd.DataFrame({'coefficients': cls1.coef_.T.tolist(), 'variable': dt_test_named.columns})
            coefficients1 = abs(np.array(coefmodel['coefficients'].apply(pd.Series).stack().reset_index(drop=True)))
            preds_pos_label_idx = np.where(cls1.classes_ == 1)[0][0]
            pred = cls1.predict_proba(dt_test_named)[
                :, preds_pos_label_idx]
            preds_all.extend(pred)

            # first attack
            cls2 = LogisticRegression(
                C=2**self.args['C'], solver='saga', penalty="l1", n_jobs=-1, random_state=random_state)
            cls2.fit(dt_train_named2, train_y)
            coefmodel = pd.DataFrame({'coefficients': cls2.coef_.T.tolist(), 'variable': dt_test_named2.columns})
            coefficients2 = abs(np.array(coefmodel['coefficients'].apply(pd.Series).stack().reset_index(drop=True)))
            pred2 = cls2.predict_proba(dt_test_named2)[
                :, preds_pos_label_idx]
            preds_all2.extend(pred2)

            # second attack
            cls3 = LogisticRegression(C=2**self.args['C'], solver='saga', penalty="l1", n_jobs=-1, random_state=random_state)
            cls3.fit(dt_train_named3, train_y)
            coefmodel = pd.DataFrame({'coefficients': cls3.coef_.T.tolist(), 'variable': dt_test_named3.columns})
            coefficients3 = abs(np.array(coefmodel['coefficients'].apply(pd.Series).stack().reset_index(drop=True)))
            pred3 = cls3.predict_proba(dt_test_named3)[:, preds_pos_label_idx]
            preds_all3.extend(pred3)

            # attack1
            array_of_distances1 = self.calculate_distance(preds_all, preds_all2, coefficients1, coefficients2, dt_test_named, dt_test_named2, test_y_all)

            # attack2
            array_of_distances2 = self.calculate_distance(preds_all, preds_all3, coefficients1, coefficients3, dt_test_named, dt_test_named2, test_y_all)

        elif self.cls_method == "RF":
            cls1 = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=random_state)
            cls1.fit(dt_train_named, train_y)
            cls2 = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=random_state)
            cls2.fit(dt_train_named2, train_y)

            cls3 = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=random_state)
            cls3.fit(dt_train_named3, train_y)

            # predictions
            preds_pos_label_idx = np.where(cls1.classes_ == 1)[0][0]
            pred = cls1.predict_proba(dt_test_named)[:, preds_pos_label_idx]
            preds_all.extend(pred)

            pred2 = cls2.predict_proba(dt_test_named2)[:, preds_pos_label_idx]
            preds_all2.extend(pred2)

            pred3 = cls3.predict_proba(dt_test_named3)[:, preds_pos_label_idx]
            preds_all3.extend(pred3)

            shap_values1 = shapley(cls1, dt_test_named)
            shap_values2 = shapley(cls2, dt_test_named2)
            shap_values3 = shapley(cls3, dt_test_named3)

            # first attack
            array_of_distances1 = self.calculate_distance(preds_all, preds_all2, shap_values1, shap_values2)

            # second attack
            array_of_distances2 = self.calculate_distance(preds_all, preds_all3, shap_values1, shap_values3)
        return pred, pred2, pred3, array_of_distances1, array_of_distances2

    def calculate_distance(self, preds, preds2, coef_a, coef_b, dt_test_named, dt_test_named2, test_y_all):
        print('attack')
        array_of_distances = np.zeros((self.max_prefix_length, 5))
        array_of_distances[:, 2] = self.prefix_lengths
        tic = time.perf_counter()
        explanations1 = None
        explanations2 = None
        if self.cls_method == 'LR':
            explanations1 = np.array(dt_test_named)*coef_a
            explanations2 = np.array(dt_test_named2)*coef_b

        elif self.cls_method == 'RF':
            explanations1 = coef_a
            explanations2 = coef_b
        # plus one for if the sum of the rows is 0
        row_sums = explanations1.sum(axis=1)+1
        norm_explanation1 = explanations1 / row_sums[:, np.newaxis]
        row_sums = explanations2.sum(axis=1)+1
        norm_explanation2 = explanations2 / row_sums[:, np.newaxis]

        for ii in range(0, len(self.nr_events_all)):
            nr_event = self.nr_events_all[ii]-1
            dist_temp = distance.euclidean(norm_explanation1[ii], norm_explanation2[ii])
            if round(preds[ii]) == round(preds2[ii]):
                array_of_distances[nr_event, 0] += dist_temp
            else:
                array_of_distances[nr_event, 1] += dist_temp
                if round(preds2[ii]) != test_y_all[ii]:
                    array_of_distances[nr_event, 4] += 1
                else:
                    array_of_distances[nr_event, 3] += 1
        toc = time.perf_counter()
        print(f"adversarial attack in {toc - tic:0.4f} seconds")
        return array_of_distances
