import sys
sys.path.insert(0, '/Users/utxeee/Desktop/deep-learning-com-perfis/code/pyTorch/utils')
import raw_data_loader as rdl
import seaborn as sns
import utils

# -----------------------------------------------------------------------------
# Force custom modules reloading otherwise changes in custom modules after
# loading will not be taken into account herein!
# -----------------------------------------------------------------------------
utils.reload_modules([rdl])


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
