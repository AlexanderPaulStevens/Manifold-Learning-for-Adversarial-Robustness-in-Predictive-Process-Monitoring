# The code and architecture in this file stems from: https://github.com/Khamies/LSTM-Variational-AutoEncoder
# I have added this file to my own GitHub, so that interested people can see how the LSTM-VAE looks like.
import os
import torch
import wandb
import logging
import numpy as np
logging.getLogger().setLevel(logging.INFO)

#from data.ptb import PTB


class LSTM_VAE(torch.nn.Module):

  def __init__(self, vocab_size, embed_size, hidden_size, latent_size, num_layers=1):
    super(LSTM_VAE, self).__init__()

    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Variables
    self.num_layers = num_layers
    self.lstm_factor = num_layers
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.latent_size = latent_size
    self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # For dictionary lookups 
    #self.dictionary = PTB(data_dir="./data", split="train", create_data= False, max_sequence_length= 60)
  
    # X: bsz * seq_len * vocab_size 
    # Embedding
    self.embed_act = torch.nn.Embedding(num_embeddings= self.vocab_size[0],embedding_dim= self.embed_size)
    self.embed_res = torch.nn.Embedding(num_embeddings= self.vocab_size[1],embedding_dim= self.embed_size)

    #    X: bsz * seq_len * vocab_size 
    #    X: bsz * seq_len * embed_size

    # Encoder Part
    self.encoder_lstm = torch.nn.LSTM(input_size= 2*self.embed_size,hidden_size= self.hidden_size, batch_first=True, num_layers= self.num_layers)
    self.mean = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.latent_size)
    self.log_variance = torch.nn.Linear(in_features= self.hidden_size * self.lstm_factor, out_features= self.latent_size)

    # Decoder Part                                   
    self.init_hidden_decoder = torch.nn.Linear(in_features= self.latent_size, out_features= self.hidden_size * self.lstm_factor)
    self.decoder_lstm = torch.nn.LSTM(input_size= 2*self.embed_size, hidden_size= self.hidden_size, batch_first = True, num_layers = self.num_layers)
    
    self.output_act = torch.nn.Linear(in_features=self.hidden_size * self.lstm_factor, out_features= self.vocab_size[0])
    self.output_res =torch.nn.Linear(in_features=self.hidden_size * self.lstm_factor, out_features= self.vocab_size[1])
    self.log_softmax_act = torch.nn.LogSoftmax(dim=2)
    self.log_softmax_res = torch.nn.LogSoftmax(dim=2)

  def init_hidden(self, batch_size):
    hidden_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    state_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
    return (hidden_cell, state_cell)

  def encoder(self, packed_x_embed,total_padding_length, hidden_encoder):

    # pad the packed input.
    packed_output_encoder, hidden_encoder = self.encoder_lstm(packed_x_embed, hidden_encoder)
    output_encoder, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output_encoder, batch_first=True, total_length= total_padding_length)
    # Extimate the mean and the variance of q(z|x)
    mean = self.mean(hidden_encoder[0])
    log_var = self.log_variance(hidden_encoder[0])
    std = torch.exp(0.5 * log_var)   # e^(0.5 log_var) = var^0.5
    
    # Generate a unit gaussian noise.
    batch_size = output_encoder.size(0)
    seq_len = output_encoder.size(1)
    noise = torch.randn(batch_size, self.latent_size).to(self.device)
    
    z = noise * std + mean

    return z, mean, log_var, hidden_encoder


  def decoder(self, z, packed_x_embed, total_padding_length=None):

    hidden_decoder = self.init_hidden_decoder(z)
    hidden_decoder = (hidden_decoder, hidden_decoder)
    

    # pad the packed input.
    packed_output_decoder, hidden_decoder = self.decoder_lstm(packed_x_embed,hidden_decoder) 
    output_decoder, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output_decoder, batch_first=True, total_length= total_padding_length)


    x_hat_act = self.output_act(output_decoder)
    x_hat_act = self.log_softmax_res(x_hat_act)

    x_hat_res = self.output_res(output_decoder)
    x_hat_res = self.log_softmax_res(x_hat_res)

    return x_hat_act, x_hat_res
    
  def forward(self, x_act, x_res,sentences_length,hidden_encoder):
    
    """
      x : bsz * seq_len
    
      hidden_encoder: ( num_lstm_layers * bsz * hidden_size, num_lstm_layers * bsz * hidden_size)

    """
    # Get Embeddings
    #encoder
    x_act_embed_enc  = self.embed_act(x_act)
    maximum_padding_length = x_act_embed_enc.size(1)
    x_res_embed_enc = self.embed_res(x_res)
    x_embed_enc = torch.cat([x_act_embed_enc, x_res_embed_enc], dim=2)
    """
    #decoder
    x_act_dec = torch.roll(x_act, shifts=1, dims=1)
    x_act_dec[0][0]=0
    x_act_embed_dec = self.embed_act(x_act_dec)
    
    x_res_dec = torch.roll(x_res, shifts=1, dims=1)
    x_res_dec[0][0]=0
    x_res_embed_dec = self.embed_res(x_res_dec)
    x_embed_dec = torch.cat([x_act_embed_dec, x_res_embed_dec], dim=2)
    """
    # Packing the input
    packed_x_embed_enc = torch.nn.utils.rnn.pack_padded_sequence(input= x_embed_enc, lengths= sentences_length, batch_first=True, enforce_sorted=False)
    # Encoder
    z, mean, log_var, hidden_encoder = self.encoder(packed_x_embed_enc, maximum_padding_length, hidden_encoder)

    # Decoder
    x_hat_act, x_hat_res = self.decoder(z, packed_x_embed_enc, maximum_padding_length)
    
    return x_hat_act, x_hat_res, mean, log_var, z, hidden_encoder

class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, model, epoch, metric_val, learning_rate, latent_size, optimizer, batch_size):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ +'_'+ str(learning_rate) +'_'+ str(latent_size) +'_'+ str(optimizer) +'_'+ str(batch_size) + f'_epoch{epoch}.pt')
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save: 
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.log_artifact(f'model-ckpt'+'_'+ str(learning_rate) +'_'+ str(latent_size) +'_'+ str(optimizer) +'_'+ str(batch_size) + f'-epoch-{epoch}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths)>self.top_n: 
            self.cleanup()
    
    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)        
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]
