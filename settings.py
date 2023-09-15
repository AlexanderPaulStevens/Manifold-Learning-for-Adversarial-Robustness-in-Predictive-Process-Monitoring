global_setting = {
    "train_ratio": 0.8,
    "train_val_ratio": 0.8,
    "seed": 42,
    "params_dir_VAE":   'params_dir_VAE',
    "params_dir":       'params_dir',
    "params_dir_DL":    'params_dir_DL',
    "results_dir":      'results_dir',
    "manifolds":        'manifolds',
    "best_manifolds":   'best_manifolds',
    "best_LSTMs":       'best_LSTMs',
    "models":           'models',
    "n_splits": 3,
    "max_evals": 15,
}

model_setting = {
    "lstm_layer": 1
}

training_setting = {
    "epochs": 100,
    "bptt": 60,
    "clip": 0.25,
    "train_losses": [],
    "test_losses": [],
    "patience": 10,
    "embed_size": 32
}
