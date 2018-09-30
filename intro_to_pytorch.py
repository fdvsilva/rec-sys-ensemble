import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim # Optimization package


# -----------------------------------------------------------------------------
# Let's make some random data
# -----------------------------------------------------------------------------

NUM_DATA_POINTS = 100

# create a an array: [0, 1, 2... NUM_DATA_POINTS]
x = np.arange(NUM_DATA_POINTS)

# create some noise to add to a truly straight line
noise = np.random.uniform(-5,5, size=(NUM_DATA_POINTS))

y = -0.5 * x + 2 + noise

plt.plot(x, y, "bs")


# -----------------------------------------------------------------------------
# Data setup:
# -----------------------------------------------------------------------------

test_data_amount     = int(0.2 * NUM_DATA_POINTS)
training_data_amount = NUM_DATA_POINTS - test_data_amount

all_data_indices = [i for i in range(NUM_DATA_POINTS)]

# get the indices of 20% of the data for testing
x_test  = np.random.choice(all_data_indices, test_data_amount, replace=False)

# get the indices of the rest for training
x_train = list(set(all_data_indices).difference(set(x_test)))

y_test  = [y[i] for i in x_test]
y_train = [y[i] for i in x_train]


# -----------------------------------------------------------------------------
# Enter: PyTorch
# -----------------------------------------------------------------------------


# an n-dimensional array : array (1d) --> matrix (2d) --> tensor (3d) --> tensor (4d) ...)
x_train_tensor = torch.Tensor(x_train)
x_test_tensor = torch.Tensor(x_test)

# create a line with 1 slope (i.e. weight) and one intercept (i.e. bias)
line_model = nn.Linear(1, 1)

# Reshape our tensor otherwise a dimension conflict will be raised
x_test_tensor_reshaped = x_test_tensor.view(-1, 1)

line_model(x_test_reshaped)


# -----------------------------------------------------------------------------
# Enter: PyTorch + Optimization
# -----------------------------------------------------------------------------


# Number of times to adjust our model
NUMBER_OF_EPOCHS = 30000
# How quickly our model "learns"
LEARNING_RATE = 1e-2
# The loss function we use. In this case, Mean Squared Error (since it's Linear Regression)
loss_function = nn.MSELoss()
# The optimizer: Stochastic Gradient Descent
optimizer = optim.Adagrad(line_model.parameters(), lr=LEARNING_RATE)

# making our labels into a form usable by PyTorch
y_train_var = torch.Tensor(y_train).view(-1, 1)

for epoch in range(NUMBER_OF_EPOCHS):
    line_model.zero_grad()
    # get our output from the model
    output = line_model(x_train_tensor.view(-1, 1))
    # calculate the loss
    loss = loss_function(output, y_train_var)
    # calculate all the partial derivatives wrt to the loss function
    loss.backward()
    # add or subtract a portion of the derivatives from each weight / bias
    optimizer.step()

real_y = [-0.5 * x + 2 for x in list(x_train)]
predicted_y = list(line_model(x_train_tensor.view(-1, 1)))

plt.plot(list(x_train), real_y, 'r-', list(x_train), predicted_y, 'bs')


# -----------------------------------------------------------------------------
# Enter: PyTorch + Optimization + Neural Networks
# -----------------------------------------------------------------------------


x_train_nn = Variable(torch.Tensor([[i] for i in range(100)]), requires_grad=False)
y_train_nn = Variable(torch.Tensor([[i] for i in range(100)]), requires_grad=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# we made a blueprint above for our neural network, now we initialize one.
net = Net()

# Red: The identity function: y = x (we're trying to approximate it)
# Blue: What our neural network does now
plt.plot(list(x_train_nn), list(x_train_nn), 'r-', list(x_train_nn), list(net(x_train_nn)), 'bs')

NUMBER_OF_EPOCHS = 1000
LEARNING_RATE = 0.1
loss_function = nn.MSELoss()
optimizer = optim.Adagrad(net.parameters(), lr=LEARNING_RATE)

for epoch in range(NUMBER_OF_EPOCHS):
    net.zero_grad()
    output = net(x_train_nn)
    loss = loss_function(output, y_train_nn)
    loss.backward()
    optimizer.step()

list(net.parameters())

plt.plot(list(x_train_nn), list(x_train_nn), 'r-', list(x_train_nn), list(net(x_train_nn)), 'bs')
