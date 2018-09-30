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
# Builds the ratings matrix both from users and activities.
#
# The returned value is an MxN matrix R where:
#   M   : number of users;
#   N   : number of activities;
#   r_mn: rating of the user m to the activity n.
# -----------------------------------------------------------------------------
def build_ratings_matrix():
    rows_number = users.shape[0]
    columns_number = activities.shape[0]
    ratings_matrix = np.full((rows_number, columns_number), get_empty_rating(), dtype="float")
    for user_id in range(rows_number):
        for _, row in get_likes_from_user(user_id).iterrows():
            activity_row_number = get_row_number_by_activity_id(row['atividade_id'])
            ratings_matrix[user_id][activity_row_number] = row['interestlevel']
    return ratings_matrix;


# -----------------------------------------------------------------------------
# Center ratings around 0
# -----------------------------------------------------------------------------
def mean_centered_ratings(ratings_matrix):
    total_users = users.shape[0]
    total_activities = activities.shape[0]
    for user_id in range(total_users):
        user_row = ratings_matrix[user_id];
        user_ratings = user_row[user_row > -1]
        if (user_ratings.size > 0):
            user_mean = user_ratings.mean()
            for i in np.argwhere(user_row > get_empty_rating()).flatten():
                ratings_matrix[user_id][i] -= user_mean
    return ratings_matrix


# -----------------------------------------------------------------------------
# Scale ratings to the interval [-1,1]
# -----------------------------------------------------------------------------
def scale_ratings(ratings_matrix):
    max = ratings_matrix.max()
    min = ratings_matrix[ratings_matrix > get_empty_rating()].min() #FIXME: Make this more efficient
    for row_number in range(ratings_matrix.shape[0]):
        for index in np.argwhere(ratings_matrix[row_number] > get_empty_rating()).flatten():
            ratings_matrix[row_number][index] = 2*((ratings_matrix[row_number][index] - min)/(max-min))-1
    return ratings_matrix


# -----------------------------------------------------------------------------
# Builds the ratings matrix with centered and scaled ratings
# -----------------------------------------------------------------------------
def build_preprocessed_ratings_matrix():
    return scale_ratings(build_ratings_matrix())


# -----------------------------------------------------------------------------
# Data preprocessor playground
# -----------------------------------------------------------------------------

build_ratings_matrix()[5]

scale_ratings(mean_centered_ratings(build_ratings_matrix()))[5]

build_preprocessed_ratings_matrix()[2]


likes['interestlevel'].max()
