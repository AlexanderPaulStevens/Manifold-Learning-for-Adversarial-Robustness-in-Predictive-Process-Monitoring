# The code and architecture in this file stems from: https://github.com/Khamies/LSTM-Variational-AutoEncoder
# I have only added this file to my own GitHub, so that interested people can easily reproduce this.
# I thank the authors from their valuable work
global_setting = {
    "train_ratio": 0.8,
    "train_val_ratio": 0.8,
    "seed": 42,
    "params_dir_VAE": 'AdversarialRobustnessGeneralization/params_dir_VAE',
    "params_dir": 'AdversarialRobustnessGeneralization/params_dir',
    "params_dir_DL": 'AdversarialRobustnessGeneralization/params_dir_DL',
    "results_dir": 'AdversarialRobustnessGeneralization/results_dir',
    "manifolds": 'AdversarialRobustnessGeneralization/manifolds',
    "best_manifolds": 'AdversarialRobustnessGeneralization/best_manifolds',
    "best_LSTMs": 'AdversarialRobustnessGeneralization/best_LSTMs',
    "models": 'AdversarialRobustnessGeneralization/models',
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
