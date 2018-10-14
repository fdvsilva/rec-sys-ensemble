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
        #self.init_weights()
        self.register_buffer('input', torch.zeros(D_in))

    def forward(self, x):
       # As the batch size is 1, we fetch the first and only element
       # of the batch
       self.input = x.clone()[0]
       hidden_output = F.tanh(self.encoder(x))
       ratings_pred = F.tanh(self.decoder(hidden_output))
       #print('input: {}'.format(x[0]))
       #print('hidden: {}'.format(hidden_output[0]))
       #print('output: {}'.format(ratings_pred[0]))
       return ratings_pred

    # -----------------------------------------------------------------------------
    # Init weights auxiliary function
    # -----------------------------------------------------------------------------
    def init_weights(self):
        nn.init.xavier_uniform_(self.encoder.weight)


# -----------------------------------------------------------------------------
# Model Playground
# -----------------------------------------------------------------------------
#model = AutoRecModel(3, 2)

#utils.print_parameters(model)
