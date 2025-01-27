import torch
import numpy as np
from NN_AE_TrainingLoop import AENN_TrainingLoop
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from get_mnist import get_data
from utils import setup_logger
import logging
 

## ---- Set seed for reproducibility purposes
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
save_file = True

# Select correct device
if torch.cuda.is_available():
    dev='cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): ## for MAC GPU usage
    dev='mps'
else:
    dev='cpu'
    
setup_logger()


# Training variables
epocs = 3
batch_s = 64
ln_rate = 1e-3


# Get data organised in batches 
training_data, test_data = get_data(batch_s=batch_s)


# Initialise training loop
trainingloop = AENN_TrainingLoop(training_data=training_data, test_data=training_data, ln_rate=ln_rate, device=dev) 
                            
                            
for e in range(epocs):
    trainingloop.train(e)

## ---- Save model ------
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,'models')

# Create directory if it did't exist before
os.makedirs(file_dir, exist_ok=True)
model_dir = os.path.join(file_dir,f'mnist_vae_model.pt')
if save_file:
    torch.save(trainingloop.ae.state_dict(),model_dir)
## -------------------------
