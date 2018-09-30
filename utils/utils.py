import torch.nn as nn

# -----------------------------------------------------------------------------
# General utility functions
# -----------------------------------------------------------------------------

# Force custom modules reloading otherwise changes in custom modules after
# loading will not be taken into account herein!
def reload_modules(modules):
    for module in modules:
        reload(module)


# -----------------------------------------------------------------------------
# Utility functions for pyTorch
# -----------------------------------------------------------------------------

# Visual representation of the model
def print_model(model):
    print(model)

# Visual representation of the parameters
def print_parameters(model):
    return [(param, param.shape) for param in model.parameters()]

# Visual representation of the parameters gradient
def print_parameters_gradient(model):
    return [(param, param.grad) for param in model.parameters()]
