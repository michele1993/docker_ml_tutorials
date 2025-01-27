import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np


class AE_NN(nn.Module):

    def __init__(self, input_s, ln_rate=1e-4, h_size=50, bottleneck_s=15):
        """ Implement a convolutional autoencoder to mimic cortex, trained to reconstruct images """

        super().__init__()

        self.bottleneck_s = bottleneck_s
        self.h_size = h_size
        self.input_s = input_s

        self.encoder = nn.Sequential(
            nn.Linear(input_s, self.h_size), 
            nn.ReLU(),
            nn.Linear(self.h_size, self.h_size), 
            nn.ReLU(),
        )

        self.l1 = nn.Linear(self.h_size, bottleneck_s) # bottleneck

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_s, self.h_size), 
            nn.ReLU(),
            nn.Linear(self.h_size, self.h_size), 
            nn.ReLU(),
            nn.Linear(self.h_size, input_s)
        )
                
        # Define optimizer
        self.optimizer = opt.Adam(self.parameters(),ln_rate)
    
    def forward(self,x):
        """ 
        NN encoder modelling IT cells, pass input through bottleneck for reconstruction
            (i.e., IT features are not trained through unsupervised learning)
            Args:
                x: the input to be reconstructed
            Returns: 
                x_pred: reconstructed input (e.g., image)
                IT_features: the (latent) representation build by the CN encoder by unsupervised learning (Not the bottleneck) 
        """
        
        IT_features = self.encoder(x.view(-1,self.input_s))

        ## -------- Unsupervised learning (to train IT features only) ----------
        bottleneck = self.l1(IT_features)
        x_pred = self.decoder(bottleneck)

        # return last layer representation
        return x_pred, IT_features.detach() #bottleneck.detach()

    def update(self, x_predictions, x_targets):
        """ 
        update the newtork based on mean squared loss on target images
        Args:
            predictions: (predicted) reconstructed images
            targets: target images
        """
        self.optimizer.zero_grad()

        # Train auto-encoder with MSE
        reconstruction_loss = nn.functional.mse_loss(x_predictions,x_targets.view(-1,self.input_s))
        loss = reconstruction_loss 
        loss.backward()
        self.optimizer.step()
        return reconstruction_loss
