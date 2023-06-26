# Manifold Learning to Improve Robustness and Generalization in Process Outcome Prediction

Complementary code to reproduce the work of *Manifold Learning for Adversarial Robustness in Process Outcome Prediction*

<!DOCTYPE html>
<html>
  <head>
    <title>Title of the document</title>
    <style>
      .circle {
        border-radius: 50%;
        width: 34px;
        height: 34px;
        padding: 10px;
        background: #fff;
        border: 3px solid #000;
        color: #000;
        text-align: center;
        font: 32px Arial, sans-serif;
      }
    </style>
  </head>
  <body>
    <div class="circle">1</div>
  </body>
</html>

This following file contains the methodological pipeline for how adversarial train data is generated for adversarial training purposes, and how adversarial examples for the test data consist of the incorrectly predicted prefixes and the adversarial prefixes.

<img width="1351" alt="methodological pipeline" src="https://github.com/AlexanderPaulStevens/Manifold-Learning-for-Adversarial-Robustness-in-Predictive-Process-Monitoring/assets/75080516/23a0ef5c-56d5-4c4a-bdae-414584243386">

First, indicated by <i class="fa-solid fa-circle-1"></i> in Figure 1, we perform an out-of-time train/test split. 
A classifier is trained on the train prefixes.
In the second step 2 , we determine the correctly predicted
train prefixes and the correctly predicted test prefixes. Note
that in the case of a black-box attack, we have access to
the predicted output of these prefixes, allowing us to try to
alter the predictions. For example, for the test prefixes, we
can see that prefix 5 was incorrectly predicted, and prefixes 6
and 7 correctly. In the third step 3 , we generate adversarial
examples for correctly predicted train and test prefixes. The
fourth step 4 shows that the adversarial train data consists
of 50% original train prefixes and 50% adversarial examples,
which means that we generate adversarial examples from the
correctly predicted prefixes until we have the same amount
as the original prefixes. The test prefixes, on the other hand,
consist of both the misclassified test prefixes and the crafted
adversarial examples. Therefore, it is crucial to consider that
these incorrectly predicted test prefixes, originally misclassified,
can represent edge cases, making it challenging for any
classifier to accurately predict their results. In the final step
5 , an adversarial classifier is built on the adversarial train
data.

### Table of Contents

- **[Introduction](#Introduction)**
- **[Files](#Files)**
- **[**Experiment**](#Experiment)**
- **[**AdversarialExperiment**](#AdversarialExperiment)**
- **[Acknowledgements](#Acknowledgements)**
- **[License](#License)** 

### Introduction

In the related paper, we use of manifold learning to generate natural adversarial examples that are restricted within the range of data that the model is trained on, i.e. adversarial examples that are restricted to its class-specific manifold. We suggests that learning from these on-manifold adversarial examples breaks the trade-off between adversarial robustness and generalization, showing that there exist models that remain accurate on new, unseen data, while being robust against worst-case adversarial treats. This GitHub repository contains all the code necessary to reproduce the work.

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

*Long Short-Term Memory (lSTM) neural networks*
- experiment_DL.py

*Variational Autoencoder (VAE)*
- experiment_VAE.py

### AdversarialExperiment
The adversarial example generation and adversarial training algorithm

*Logistic Regression (LR), Random Forest (RF) XGBoost (XGB)*
- adversarial_experiment.py

*Long Short-Term Memory (LSTM) neural networks*
- adversarial_experiment_LSTM.py

![adversarial example generation](https://github.com/AlexanderPaulStevens/Manifold-Learning-for-Adversarial-Robustness-in-Predictive-Process-Monitoring/assets/75080516/4617049a-2a59-4d1f-bcee-e18938430cee)


### Acknowledgements

- This work has used the GitHub repository from:

- https://github.com/irhete/predictive-monitoring-benchmark
- https://github.com/Khamies/LSTM-Variational-AutoEncoder

We again thank the authors for their valuable and easily reproducible work.
