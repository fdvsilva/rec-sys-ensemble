import sys
sys.path.insert(0, '../utils')
sys.path.insert(0, '/Users/utxeee/Desktop/deep-learning-com-perfis/code/pyTorch/data-manipulation')
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import auto_rec_model as arm
import preprocessed_data_loader as pdl
import utils


# -----------------------------------------------------------------------------
# Force custom modules reloading otherwise changes in custom modules after
# loading will not be taken into account herein!
# -----------------------------------------------------------------------------
utils.reload_modules([arm,pdl])


# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
D_in = D_out = 71 # number of activities
H = 20
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4


# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
model = arm.AutoRecModel(D_in, H)


# -----------------------------------------------------------------------------
# Loss initialization
# -----------------------------------------------------------------------------
loss = nn.MSELoss()

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())

loss.register_forward_hook(printnorm)

out = net(input)

# -----------------------------------------------------------------------------
# Optimizer initialization
# -----------------------------------------------------------------------------
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)


# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
for epoch in range(NUM_EPOCHS):
    for batch_idx, (target, input) in enumerate(pdl.get_data_loader()):
        print(batch_idx, target, input)
