import sys
sys.path.insert(0, '../utils')
sys.path.insert(0, '/Users/utxeee/Desktop/deep-learning-com-perfis/code/pyTorch/data-manipulation')
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import auto_rec_model as arm
import preprocessed_data_loader as pdl
import utils
import math

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
# Train for a single epoch
# -----------------------------------------------------------------------------
def train_epoch(epoch, model, data_loader, optimizer):
    #model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0]))


# -----------------------------------------------------------------------------
# Validate a single epoch
# -----------------------------------------------------------------------------
def validate_epoch(model, data_loader):
    #model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, (index, target_rating) in enumerate(data_loader):
            output = model(data)
            test_loss += pow(target_rating - output[index], 2)

    test_loss /= len(data_loader.dataset)
    test_loss = math.sqrt(test_loss)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))


# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
for epoch in range(NUM_EPOCHS):
    train_epoch(epoch, model, data_loader, optimizer)
    validate_epoch(model, data_loader)
