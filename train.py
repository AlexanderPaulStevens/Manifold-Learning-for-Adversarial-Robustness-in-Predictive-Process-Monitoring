import torch
import time
import os 

class Trainer:

    def __init__(self, train_loader, test_loader, model, loss, optimizer) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.interval = 200

    def get_batch(self,batch):
        sentences_length = torch.tensor([batch.shape[1]]*batch.shape[0])
        return sentences_length

    def train(self, train_losses, epoch, batch_size, clip) -> list:  
        # Initialization of RNN hidden, and cell states.
        states = self.model.init_hidden(batch_size)
        for i, (data_act, data_res) in enumerate(self.train_loader, 0): # loop over the data, and jump with step = bptt.
            # get the labels
            sentences_length = self.get_batch(data_act)
            data_act = data_act.to(self.device)
            data_res = data_res.to(self.device)
            x_hat_param_act, x_hat_param_res, mu, log_var, z, states = self.model(data_act,data_res, sentences_length, states)
            # detach hidden states
            states = states[0].detach(), states[1].detach()

            # compute the loss
            mloss, KL_loss, recon_loss = self.loss(mu = mu, log_var = log_var, z = z, x_hat_param_act = x_hat_param_act, x_hat_param_res = x_hat_param_res, x_act = data_act,x_res = data_res)

            mloss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.optimizer.step()
            self.optimizer.zero_grad()

            mloss_detach = mloss.detach()
            KL_detach = KL_loss.detach()
            recon_detach = recon_loss.detach()

            train_losses.append((mloss_detach.item(), KL_detach.item(), recon_detach.item()))
            if i % self.interval == 0 and i > 0:
                print('what')
                print('| epoch {:3d} | elbo_loss {:5.6f} | kl_loss {:5.6f} | recons_loss {:5.6f} '.format(
                    epoch, mloss_detach.item(), KL_detach.item(), recon_detach.item()))     
        return train_losses

    def test(self, test_losses, epoch, batch_size) -> list:

        with torch.no_grad():

            states = self.model.init_hidden(batch_size) 

            for i, (data_act, data_res) in enumerate(self.test_loader, 0): # loop over the data, and jump with step = bptt.
                # get the labels
                sentences_length = self.get_batch(data_act)
                data_act = data_act.to(self.device)
                data_res = data_res.to(self.device)
                x_hat_param_act, x_hat_param_res, mu, log_var, z, states = self.model(data_act,data_res, sentences_length, states)

                # detach hidden states
                states = states[0].detach(), states[1].detach()

                # compute the loss
                mloss, KL_loss, recon_loss = self.loss(mu = mu, log_var = log_var, z = z, x_hat_param_act = x_hat_param_act, x_hat_param_res = x_hat_param_res, x_act = data_act,x_res = data_res)

                test_losses.append((mloss.item() , KL_loss.item(), recon_loss.item()))

                # Statistics.
                # if batch_num % 20 ==0:
                #   print('| epoch {:3d} | elbo_loss {:5.6f} | kl_loss {:5.6f} | recons_loss {:5.6f} '.format(
                #         epoch, mloss.item(), KL_loss.item(), recon_loss.item()))
                
            return test_losses