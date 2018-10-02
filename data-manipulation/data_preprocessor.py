import sys
sys.path.insert(0, '/Users/utxeee/Desktop/deep-learning-com-perfis/code/pyTorch/utils')
import numpy as np
import raw_data_loader as rdl
import data_splitting as ds
import utils

# -----------------------------------------------------------------------------
# Force custom modules reloading otherwise changes in custom modules after
# loading will not be taken into account herein!
# -----------------------------------------------------------------------------
utils.reload_modules([rdl, ds])


# -----------------------------------------------------------------------------
# Center ratings around 0
# -----------------------------------------------------------------------------
def mean_centered_ratings(ratings_matrix):
    total_users = ratings_matrix.shape[0]
    total_activities = ratings_matrix.shape[0]
    for user_id in range(total_users):
        user_row = ratings_matrix[user_id];
        user_ratings = user_row[user_row > -1]
        if (user_ratings.size > 0):
            user_mean = user_ratings.mean()
            for i in np.argwhere(user_row > ds.get_empty_rating()).flatten():
                ratings_matrix[user_id][i] -= user_mean
    return ratings_matrix


# -----------------------------------------------------------------------------
# Scale ratings to the interval [-1,1]
# -----------------------------------------------------------------------------
def scale_ratings(ratings_matrix):
    max = ratings_matrix.max()
    min = ratings_matrix[ratings_matrix > ds.get_empty_rating()].min() #FIXME: Make this more efficient
    for row_number in range(ratings_matrix.shape[0]):
        for index in np.argwhere(ratings_matrix[row_number] > ds.get_empty_rating()).flatten():
            ratings_matrix[row_number][index] = 2*((ratings_matrix[row_number][index] - min)/(max-min))-1
    return ratings_matrix


# -----------------------------------------------------------------------------
# Builds the ratings matrix with centered and scaled ratings
# -----------------------------------------------------------------------------
def build_preprocessed_ratings_matrix():
    train_matrix, _, _ = ds.build_matrices()
    centered_train_matrix = mean_centered_ratings(train_matrix)

    centered_train_matrix[28]

    return scale_ratings(centered_train_matrix)


# -----------------------------------------------------------------------------
# Data preprocessor playground
# -----------------------------------------------------------------------------
