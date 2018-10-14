import sys
sys.path.insert(0, '/Users/utxeee/Desktop/deep-learning-com-perfis/code/pyTorch/utils')
import raw_data_loader as rdl
import seaborn as sns
import utils
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Force custom modules reloading otherwise changes in custom modules after
# loading will not be taken into account herein!
# -----------------------------------------------------------------------------
utils.reload_modules([rdl, utils])


# -----------------------------------------------------------------------------
# Raw data fecthing
# -----------------------------------------------------------------------------
users = rdl.get_users() # Fetch users
activities = rdl.get_activities() # Read activities file
likes = rdl.get_likes() # Read likes file


# -----------------------------------------------------------------------------
# Display a plot with every pair (x_var, y_var) found in data
# -----------------------------------------------------------------------------
def display_pair_plot(x_var, y_var, data_frame):
    sns.pairplot(data_frame, x_vars=[x_var], y_vars=[y_var], size=5, aspect=1)


# -----------------------------------------------------------------------------
# Display plot for points (user_id, interestlevel)
# -----------------------------------------------------------------------------
display_pair_plot('user_id', 'interestlevel', likes)


# -----------------------------------------------------------------------------
# Display an histogram for the number of likes per activity
# -----------------------------------------------------------------------------
def display_likes_per_activity_histogram():
    sns.catplot(data=rdl.get_likes(), y = 'atividade_id', color='#3498db', kind="count", height=15,aspect=0.5)


# -----------------------------------------------------------------------------
# Playground
# -----------------------------------------------------------------------------
