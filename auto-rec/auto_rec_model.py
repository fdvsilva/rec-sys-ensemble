import sys
sys.path.insert(0, '/Users/utxeee/Desktop/deep-learning-com-perfis/code/pyTorch/utils')
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

# -----------------------------------------------------------------------------
# Model definition for AutoRec
# -----------------------------------------------------------------------------

class AutoRecModel(nn.Module):

    def __init__(self, D_in, H):
        super(AutoRecModel, self).__init__()
        self.encoder = nn.Linear(D_in, H)
        self.decoder = nn.Linear(H, D_in)
        # tie weights
        self.decoder.weight.data = self.encoder.weight.data.transpose(0,1)
        self.register_buffer('input', torch.zeros(D_in))

    def forward(self, x):
       #hidden_output = F.sigmoid(self.l1(x))
       #ratings_pred = F.sigmoid(self.l2(hidden_output))
       self.input = x.clone()
       #print("size:" + str(x.shape[0]))
       #print("input:" + str(self.input))
       hidden_output = self.encoder(x)
       ratings_pred = self.decoder(hidden_output)
       return ratings_pred
