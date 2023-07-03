# Manifold Learning to Improve Robustness and Generalization in Process Outcome Prediction

Complementary code to reproduce the work of *Manifold Learning for Adversarial Robustness in Process Outcome Prediction*

This following file contains the methodological pipeline for how adversarial train data is generated for adversarial training purposes, and how adversarial examples for the test data consist of the incorrectly predicted prefixes and the adversarial prefixes.

<p align="center">
<img width="700" alt="methodological pipeline" src="https://github.com/AlexanderPaulStevens/Manifold-Learning-for-Adversarial-Robustness-in-Predictive-Process-Monitoring/assets/75080516/23a0ef5c-56d5-4c4a-bdae-414584243386">
</p>

First, we perform an out-of-time train/test split. A classifier is trained on the train prefixes. In the second step 2, we determine the correctly predicted train prefixes and the correctly predicted test prefixes. In the third step 3 , we generate adversarial examples for correctly predicted train and test prefixes. The fourth step 4 shows that the adversarial train data consists of 50% original train prefixes and 50% adversarial examples, which means that we generate adversarial examples from the correctly predicted prefixes until we have the same amount as the original prefixes. The test prefixes, on the other hand, consist of both the misclassified test prefixes and the crafted adversarial examples. In the final step 5, an adversarial classifier is built with the adversarial train data.

The model architecture of the Long Short-Term Memory (LSTM) neural network is given in the following figure. A class-specific manifold is learnt for each label of the dataset.

<p align="center">
<img width="500" alt="VAE" src="https://github.com/AlexanderPaulStevens/Manifold-Learning-for-Adversarial-Robustness-in-Predictive-Process-Monitoring/assets/75080516/3c7a8cbf-1c82-414f-886c-3834c0f0f563">
</p>

### Table of Contents

- **[Introduction](#Introduction)**
- **[Files](#Files)**
- **[**Experiment**](#Experiment)**
- **[**Adversarial Experiment**](#AdversarialExperiment)**
- **[**Appendix**](#Appendix)**
- **[Acknowledgements](#Acknowledgements)**
- **[License](#License)** 

### Introduction

In this paper, we use manifold learning to generate natural adversarial examples that are restricted within the range of data that the model is trained on, i.e. projected to its class-specific manifold. The experimental results suggest that, by learning from these on-manifold adversarial examples,  we can create models that remain accurate on new, unseen data, while being robust against (worst-case) adversarial scenarios. This GitHub repository contains all the code necessary to reproduce the work.

### Files

#### Preprocessing files

The preprocessing and hyperoptimalisation are derivative work based on the code provided by [Outcome-Oriented Predictive Process Monitoring: Review and Benchmark](https://github.com/irhete/predictive-monitoring-benchmark).
We would like to thank the authors for the high quality code that allowed to fastly reproduce the provided work.
- dataset_confs.py
- DatasetManager.py
- EncoderFactory.py

#### PDF
The folder PDF contains the high-resolution figures (PDF format) that have been used in the paper

### Experiment 
To define the optimal models (and save them locally)

*Logistic Regression (LR), Random Forest (RF) and XGBoost (XGB)*
- experiment_ML.py

*Long Short-Term Memory (LSTM) neural networks*
- experiment_DL.py

*Variational Autoencoder (VAE)*
- experiment_VAE.py

### Adversarial Experiment
The adversarial example generation and adversarial training algorithm

*Logistic Regression (LR), Random Forest (RF) XGBoost (XGB)*
- adversarial_experiment.py

*Long Short-Term Memory (LSTM) neural networks*
- adversarial_experiment_LSTM.py

<p align="center">
<img width="700" alt="adversarial example generation" src="https://github.com/AlexanderPaulStevens/Manifold-Learning-for-Adversarial-Robustness-in-Predictive-Process-Monitoring/assets/75080516/3a7b8d66-9c8c-4fac-a92f-a0d10c2afc2d">
</p>

### Appendix

![BPIC20152Merged](https://github.com/AlexanderPaulStevens/Manifold-Learning-for-Adversarial-Robustness-in-Predictive-Process-Monitoring/assets/75080516/35b5ecfc-507e-4c18-a695-cd7dd34b40c1)


### Acknowledgements

- This work has used the GitHub repository from:

- https://github.com/irhete/predictive-monitoring-benchmark
- https://github.com/Khamies/LSTM-Variational-AutoEncoder

We again thank the authors for their valuable and easily reproducible work.
