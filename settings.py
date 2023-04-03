global_setting = {
    "train_ratio": 0.8,
    "train_val_ratio": 0.8,
    "seed": 42,
    "params_dir_VAE": r'C:/Users/u0138175/Google Drive/CurrentWork/Manifold/AdversarialRobustnessGeneralization/params_dir_VAE',
    "params_dir": r'C:/Users/u0138175/Google Drive/CurrentWork/Manifold/AdversarialRobustnessGeneralization/params_dir',
    "params_dir_DL": r'C:/Users/u0138175/Google Drive/CurrentWork/Manifold/AdversarialRobustnessGeneralization/params_dir_DL',
    "results_dir": r'C:/Users/u0138175/Google Drive/CurrentWork/Manifold/AdversarialRobustnessGeneralization/results_dir',
    "manifolds": r'C:/Users/u0138175/Google Drive/CurrentWork/Manifold/AdversarialRobustnessGeneralization/manifolds',
    "best_manifolds": r'C:/Users/u0138175/Google Drive/CurrentWork/Manifold/AdversarialRobustnessGeneralization/best_manifolds',
    "best_LSTMs": r'C:/Users/u0138175/Google Drive/CurrentWork/Manifold/AdversarialRobustnessGeneralization/best_LSTMs',
    "models": r'C:/Users/u0138175/Google Drive/CurrentWork/Manifold/AdversarialRobustnessGeneralization/models',
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
    "embed_size": 16
}
