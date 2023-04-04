# Manifold Learning to Improve Robustness and Generalization in Process Outcome Prediction

Complementary code to reproduce the work of *Manifold Learning to Improve Robustness and Generalization in Process Outcome Prediction*

![On-manifold training](https://user-images.githubusercontent.com/75080516/229730705-bc4970fa-1f52-4fc5-ba20-51e5a6e89b6a.PNG)

This preview file contains the adversarial training architecture, and how regular adversarial training deviates from on-manifold adversarial training. More information is given below.

### Table of Contents

- **[Introduction](#Introduction)**
- **[Setup](#Setup)**
- [**Run the code**](#Run-the-code)
- **[Training](#Training)**
- **[Inference](#Inference)**
- **[Play with the model](#Play-with-the-model)**
- **[Connect with me](#Connect-with-me)**
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

### Run the code

- To train the model, run: `python main.py`

- To train the model with specific arguments, run: `python main.py --batch_size=64`. The following command-line arguments are available:
  - Batch size: `--batch_size`
  - bptt: `--bptt`
  - Learning rate: `--lr`
  - Embedding size: `--embed_size`
  - Hidden size: `--hidden_size`
  - Latent size: `--latent_size`

### Training

The model is trained on 30 epochs using Adam as an optimizer with a learning rate 0.001. Here are the results from training the LSTM-VAE model:

- **KL Loss**

  <img src="./media/kl.jpg" align="center" height="300" width="500" >

- **Reconstruction loss**

  <img src="./media/reco.jpg" align="center" height="300" width="500" >

- **KL loss vs Reconstruction loss**

  <img src="./media/kl_reco.jpg" align="center" height="300" width="500" >

- **ELBO loss**

  <img src="./media/elbo.jpg" align="center" height="300" width="500" >



### Inference

#### 1. Sample Generation

Here are generated samples from the model. We randomly sampled two latent codes z from standard Gaussian distributions, and specify "like" as the start of the sentence (sos), then we feed them to the decoder. The following are the generated sentences:

- **like other countries such as alex powers a former wang marketer** 

- **like design and artists have been <unk> by how many <unk>**

#### 2. Interpolation

The "President" word has been used as the start of the sentences. We randomly generated two sentences and interpolated between them.

- Sentence 1: **President bush veto power changes meant to be a great number**
- Sentence 2: **President bush veto power opposed to the president of the house**

```markdown
 *bush veto power opposed to the president of the house
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president of the house.
 bush veto power opposed to the president ' s council.
 bush veto power opposed to the president ' s council.
 bush veto power opposed to the president ' s council.
 bush veto power opposed to the president ' s council.
 bush veto power opposed to the president ' s council.
 bush veto power that kind of <unk> of natural gas.
 bush veto power changes to keep the <unk> and that.
 bush veto power changes to keep the <unk> and that.
 bush veto power changes that is in a telephone to.
 bush veto power changes that is in a telephone to.
 bush veto power changes meant to be a great number.
 *bush veto power changes meant to be a great number
```

### Acknowledgement

- This work has used the GitHub repository from:

- https://github.com/irhete/predictive-monitoring-benchmark
- https://github.com/Khamies/LSTM-Variational-AutoEncoder

We again thank the authors for their valuable and easily reproducible work.

### License 

https://github.com/AlexanderPaulStevens/ManifoldLearningPPM/blob/main/LICENSE
