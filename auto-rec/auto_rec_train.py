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
