import sys
sys.path.insert(0, '/Users/utxeee/Desktop/deep-learning-com-perfis/code/pyTorch/utils')
import numpy as np
import raw_data_loader as rdl
#import data_splitting as ds
import utils
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# Force custom modules reloading otherwise changes in custom modules after
# loading will not be taken into account herein!
# -----------------------------------------------------------------------------
utils.reload_modules([rdl, utils])


# -----------------------------------------------------------------------------
# Scale rating to range [lower_bound, upper_bound] (via third-party libraries)
#
# Remark: Tipically, we scale ratings to [1.10] range because we want to get rid
#         of 0 stars ratings as that rating will be allocated to missing ratings
#         and besides that a range between [1,10] is more meaningful than [0,100]
# -----------------------------------------------------------------------------
def scale_ratings(lower_bound, upper_bound, ratings_matrix):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(lower_bound, upper_bound))
    reshaped_ratings_matrix = np.array(ratings_matrix).reshape(-1,1)
    #return zip(reshaped_ratings, min_max_scaler.fit_transform(reshaped_ratings))
    return min_max_scaler.fit_transform(reshaped_ratings_matrix).reshape(ratings_matrix.shape)


# -----------------------------------------------------------------------------
# Center ratings around 0 (via third-party libraries)
# -----------------------------------------------------------------------------
def zero_center_ratings(ratings_matrix):
    #trans_ratings_matrix = np.transpose(ratings_matrix)
    reshaped_ratings_matrix = np.array(ratings_matrix).reshape(-1,1)
    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=False)
    scaler.fit(reshaped_ratings_matrix)
    return scaler.transform(reshaped_ratings_matrix).reshape(ratings_matrix.shape)


# -----------------------------------------------------------------------------
# Center ratings around 0 (from scratch)
#
# DEPRECATED: See zero_center_ratings
# -----------------------------------------------------------------------------
def mean_centered_ratings_DEPRECATED(ratings_matrix):
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
# Scale ratings to the interval [-1,1] (from scratch)
#
# DEPRECATED: See scale_ratings
# -----------------------------------------------------------------------------
def scale_ratings_DEPRECATED(ratings_matrix):
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
