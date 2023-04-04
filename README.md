# Manifold Learning to Improve Robustness and Generalization in Process Outcome Prediction

Complementary code to reproduce the work of *Manifold Learning to Improve Robustness and Generalization in Process Outcome Prediction*

![On-manifold training](https://user-images.githubusercontent.com/75080516/229730705-bc4970fa-1f52-4fc5-ba20-51e5a6e89b6a.PNG)

This preview file contains the adversarial training architecture, and how regular adversarial training deviates from on-manifold adversarial training. More information is given below.

### Table of Contents

- **[Introduction](#Introduction)**
- **[Files](#Files)**
- **[**Experiment**](#Experiment)**
- **[Adversarial Example Generation and Training](#Adversarial Example Generation and Training)**
- **[Acknowledgements](#Adversarial Example Generation and Training)**
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
*Logistic Regression (LR) and Random Forest (RF)*
- experiment_ML.py
*Variational Autoencoder (VAE)*
- experiment_VAE.py

### Adversarial Example Generation and Training
The adversarial example generation and adversarial training algorithm
*Logistic Regression (LR) and Random Forest (RF)*
- adversarial_experiment.py

### Acknowledgements

- This work has used the GitHub repository from:

- https://github.com/irhete/predictive-monitoring-benchmark
- https://github.com/Khamies/LSTM-Variational-AutoEncoder

We again thank the authors for their valuable and easily reproducible work.

### License 

https://github.com/AlexanderPaulStevens/ManifoldLearningPPM/blob/main/LICENSE
