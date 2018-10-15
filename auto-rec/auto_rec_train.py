import sys
sys.path.insert(0, '../utils')
sys.path.insert(0, '/Users/utxeee/Desktop/deep-learning-com-perfis/code/pyTorch/data-manipulation')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import auto_rec_model as arm
import auto_rec_loss_function as arl
import data_loader as dl
import utils
import math

# -----------------------------------------------------------------------------
# Force custom modules reloading otherwise changes in custom modules after
# loading will not be taken into account herein!
# -----------------------------------------------------------------------------
utils.reload_modules([arm,dl,arl,utils])


# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
D_in = D_out = 71 # number of activities
H = 20
NUM_EPOCHS = 1
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY =  0 #0.5

# -----------------------------------------------------------------------------
# Model initialization
# -----------------------------------------------------------------------------
model = arm.AutoRecModel(D_in, H)


# -----------------------------------------------------------------------------
# Loss initialization
# -----------------------------------------------------------------------------
autorec_loss = arl.AutoRecLoss()


# -----------------------------------------------------------------------------
# Optimizer initialization
# -----------------------------------------------------------------------------
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


# -----------------------------------------------------------------------------
# Zero gradients for weights whose input is missing while doing back
# propagation for the encoder part of the network
# -----------------------------------------------------------------------------
def zero_encoder_weights_grads(grad):
    column_index = 0
    grad_clone = grad.clone()
    #print('grad_clone: {} | grad_shape: {}'.format(grad_clone, grad.shape))
    #print('input: {}'.format(model.state_dict().get("input")))
    for rating in model.state_dict().get("input"):
        #print('rating: {}'.format(rating))
        if rating < 0:
            #grad_clone[column_index, :] = 0
            grad_clone[:, column_index] = 0
        column_index += 1
    print(grad_clone)
    return grad_clone


# -----------------------------------------------------------------------------
# Add an hook to function responsible for calculating the gradients of the
# weights associated with the encoder part of the network.
# -----------------------------------------------------------------------------
def add_encoder_grad_hook(encoder_hook):
    for name, param in model.named_parameters():
       if name == "encoder.weight":
           encoder_handle = param.register_hook(encoder_hook)
    return encoder_handle


# -----------------------------------------------------------------------------
# Train for a single epoch
# -----------------------------------------------------------------------------
def train_epoch(epoch, model, data_loader, optimizer, loss_function):
    #model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        # print('data: {} | type: {}'.format(data[0], data[0].type()))
        optimizer.zero_grad()
        output = model(data)
        #loss = torch.sqrt(F.mse_loss(output, target))
        loss = loss_function(output, target)
        #encoder_hook_handle = add_encoder_grad_hook(zero_encoder_weights_grads)
        loss.backward()
        optimizer.step()
        #encoder_hook_handle.remove()
        #print('Loss: {}'.format(loss.item()))
        batch_idx_off_by_1 = batch_idx + 1
        if (batch_idx_off_by_1 % len(data_loader.dataset) == 0) or (batch_idx_off_by_1 * len(data) == len(data_loader.dataset)):
            # print('Output: {} | Target: {}'.format(output, target))
            # print('Loss: {}'.format(loss.item()))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx_off_by_1 * len(data), len(data_loader.dataset),
                100. * batch_idx_off_by_1 / len(data_loader), loss.item()))


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

#model = 0
for epoch in range(1000):
    train_epoch(epoch + 1, model, dl.get_train_data(), optimizer, autorec_loss)
    # validate_epoch(model, data_loader)


'''
with torch.no_grad():
    for batch_idx, (target, input) in enumerate(dl.get_train_data()):
        #output = model(target)
        #print(target[0][input[0].item()-1])
        print(target)
        print(model(target))
'''
