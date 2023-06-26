# import packages
import pandas as pd
import numpy as np
import os
import EncoderFactory
from DatasetManager import DatasetManager
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import hyperopt
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
from hyperopt.pyll.base import scope
from sklearn.preprocessing import MinMaxScaler 
from util.DataCreation import DataCreation
from util.Arguments import Args
from settings import global_setting
os.chdir('G:\My Drive\CurrentWork\Manifold\AdversarialRobustnessGeneralization')
# parameters
seed = global_setting['seed']
train_ratio = global_setting['train_ratio']
path = global_setting['params_dir']
n_splits = global_setting['n_splits']
max_evals = global_setting['max_evals']

# create params directory
if not os.path.exists(os.path.join(path)):
    os.makedirs(os.path.join(path))

encoding_dict = {
    "agg": ["agg"]}
encoding = []
for k, v in encoding_dict.items():
    encoding.append(k)
dataset_ref_to_datasets = {
    # "production": ["production"],
    "bpic2015": ["bpic2015_%s_f2" % (municipality) for municipality in range(1, 2)],
    # "bpic2012": ["bpic2012_cancelled"]
     #, "bpic2012_cancelled", "bpic2012_declined","bpic2012_accepted"],
    # "hospital_billing": ["hospital_billing_%s"%suffix for suffix in [2,3]],
    # "traffic_fines": ["traffic_fines_%s" % formula for formula in range(1, 2)],
    # "bpic2017": ["bpic2017_accepted","bpic2017_cancelled","bpic2017_refused"],
    # "sepsis_cases": ["sepsis_cases_2","sepsis_cases_4"],
    # "bpic2011": ["bpic2011_f%s"%formula for formula in range(2,4)],
}

datasets = []
for k, v in dataset_ref_to_datasets.items():
    datasets.extend(v)

# classifiers dictionary
classifier_ref_to_classifiers = {
    "LRmodels": ["LR"],
    "MLmodels": ["RF", "XGB"]
}

classifiers = []
for k, v in classifier_ref_to_classifiers.items():
    classifiers.extend(v)


def create_and_evaluate_model(args):
    global trial_nr
    trial_nr += 1
    score = 0
    for cv_iter in range(n_splits):
        dt_test_prefixes = dt_prefixes[cv_iter]
        dt_train_prefixes = pd.DataFrame()
        for cv_train_iter in range(n_splits):
            if cv_train_iter != cv_iter:
                dt_train_prefixes = pd.concat([dt_train_prefixes, dt_prefixes[cv_train_iter]], axis=0)

        preds_all = []
        test_y_all = []
        test_y = dataset_manager.get_label_numeric(dt_test_prefixes)
        train_y = dataset_manager.get_label_numeric(dt_train_prefixes)
        test_y_all.extend(test_y)

        dt_train_prefixes = dt_train_prefixes[cols].copy()
        feature_combiner, scaler, dt_train_named = datacreator.transform_data_train(dt_train_prefixes, train_y, encoding_dict, cls_encoder_args, cat_cols)
        dt_test_named = datacreator.transform_data_test(feature_combiner, scaler, dt_test_prefixes)
        dt_train_named_original, dt_test_named_original = dt_train_named.copy(), dt_test_named.copy()

        cls = None
        if cls_method == "LR":
            cls = LogisticRegression(C=2**args['C'], solver='saga', penalty="l1", n_jobs=-1, random_state=seed)

        elif cls_method == "XGB":
            cls = xgb.XGBClassifier(objective='binary:logistic',
                                    n_estimators=500,
                                    learning_rate=args['learning_rate'],
                                    subsample=args['subsample'],
                                    max_depth=int(args['max_depth']),
                                    colsample_bytree=args['colsample_bytree'],
                                    min_child_weight=int(args['min_child_weight']),
                                    n_jobs=-1,
                                    seed=seed)
        elif cls_method == "RF":
            cls = RandomForestClassifier(n_estimators=500,
                                         max_features=args['max_features'],
                                         n_jobs=-1,
                                         random_state=seed)
        cls.fit(dt_train_named, train_y)
        preds_pos_label_idx = np.where(cls.classes_ == 1)[0][0]
        pred = cls.predict_proba(dt_test_named)[:, preds_pos_label_idx]
        preds_all.extend(pred)

        score += roc_auc_score(test_y_all, preds_all)
        for k, v in args.items():
            fout_all.write("%s;%s;%s;%s;%s;%s\n" % (trial_nr, dataset_name,
                                                    cls_method, k, v, score / n_splits))
        fout_all.write("%s;%s;%s;%s\n" % (trial_nr, dataset_name, cls_method, 0))
    fout_all.flush()
    return {'loss': -score / n_splits, 'status': STATUS_OK, 'model': cls}


# print dataset name
for dataset_name in datasets:
    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()
    if "bpic2012" in dataset_name:
        data = data[data['Activity'].str.startswith('W')]
    arguments = Args(dataset_name)
    for cls_method in classifiers:
        for cls_encoding in encoding:
            print('Dataset:', dataset_name)
            print('Classifier', cls_method)
            cls_encoder_args, min_prefix_length, max_prefix_length, activity_col, resource_col = arguments.extract_args(data, dataset_manager)
            datacreator = DataCreation(dataset_manager, dataset_name, cls_method, cls_encoding)
            print('prefix length', min_prefix_length, 'until', max_prefix_length)
            # split into training and test
            train, _ = dataset_manager.split_data_strict(data, train_ratio, split="temporal")
            cat_cols = [activity_col, resource_col]
            cols = [cat_cols[0], cat_cols[1], cls_encoder_args['case_id_col'], 'label', 'event_nr']

            # prepare chunks for CV
            dt_prefixes = []
            class_ratios = []
            for train_chunk, test_chunk in dataset_manager.get_stratified_split_generator(train, n_splits=n_splits):
                class_ratios.append(dataset_manager.get_class_ratio(train_chunk))
                # generate data where each prefix is a separate instance
                dt_prefixes.append(dataset_manager.generate_prefix_data(test_chunk,
                                                                        min_prefix_length, max_prefix_length))
            del train

            # set up search space
            space = {}

            if cls_method == "LR":
                space = {'C': hp.uniform('C', -15, 15)}

            elif cls_method == "RF":
                space = {'max_features': hp.uniform('max_features', 0, 1)}

            elif cls_method == "XGB":
                space = {'learning_rate': hp.uniform("learning_rate", 0, 1),
                         'subsample': hp.uniform("subsample", 0.5, 1),
                         'max_depth': scope.int(hp.quniform('max_depth', 4, 30, 1)),
                         'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 1),
                         'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))}

            # optimize parameters
            trial_nr = 0
            trials = Trials()
            fout_all = open(os.path.join(path, "param_optim_all_trials_%s_%s.csv" % (cls_method, dataset_name)), "w")
            fout_all.write("%s;%s;%s;%s;%s;%s;%s\n" % ("iter", "dataset", "cls", "method", "param", "value",
                                                       "score"))
            best = fmin(create_and_evaluate_model, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
            fout_all.close()

            # write the best parameters
            best_params = hyperopt.space_eval(space, best)
            outfile = os.path.join(path, "optimal_params_%s_%s.pickle" % (cls_method, dataset_name))
            # write to file
            with open(outfile, "wb") as fout:
                pickle.dump(best_params, fout)
