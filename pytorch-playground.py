import sys
sys.path.insert(0, 'auto-rec')
sys.path.insert(0, 'utils')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim # Optimization package
import torch.optim as optim
import auto_rec_model as arm
import utils


#def init():
utils.reload_modules([utils, arm])


INPUT_SIZE = 5
HIDDEN_SIZE = 3
autorec = arm.AutoRecModel(INPUT_SIZE, HIDDEN_SIZE)
#utils.print_parameters(autorec)
input = torch.tensor([1.,-1.,3,-1,5], requires_grad=False)
loss_function = nn.MSELoss()
optimizer = optim.SGD(autorec.parameters(), lr=1e-1)


def add_grad_hooks(encoder_hook, decoder_hook):
    for name, param in autorec.named_parameters():
       if name == "encoder.weight":
           encoder_handle = param.register_hook(encoder_hook)
       if name == "decoder.weight":
           decoder_handle = param.register_hook(decoder_hook)
    return encoder_handle, decoder_handle


def zero_weights(grad, network='ENCODER'):
    column_index = 0
    grad_clone = grad.clone()
    for rating in autorec.state_dict().get("input"):
        if rating < 0:
            if network == 'ENCODER':
                grad_clone[:,column_index] = 0
            else:
                grad_clone[column_index, :] = 0
        column_index += 1
    print(grad_clone)
    return grad_clone

def zero_encoder_weights(grad):
    return zero_weights(grad)

def zero_decoder_weights(grad):
    return zero_weights(grad, 'DECODER')


for i in range(1):

    autorec.zero_grad()
    pred_y = autorec(input)
    loss = loss_function(pred_y, input)

    pred_y
    encoder_handle, decoder_handle = add_grad_hooks(zero_encoder_weights, zero_decoder_weights)

    loss.backward()

    optimizer.step()
    encoder_handle.remove()
    decoder_handle.remove()

    print(utils.print_parameters_gradient(autorec))


print("Model's state dictionary")
for param in autorec.state_dict():
    print(param, autorec.state_dict()[param].size())

list(range(5))
