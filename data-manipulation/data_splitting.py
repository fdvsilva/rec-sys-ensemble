import sys
sys.path.insert(0, '/Users/utxeee/Desktop/deep-learning-com-perfis/code/pyTorch/utils')
import numpy as np
from numpy import array
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import raw_data_loader as rdl
import data_preprocessor as dp
import utils

# -----------------------------------------------------------------------------
# Force custom modules reloading otherwise changes in custom modules after
# loading will not be taken into account herein!
# -----------------------------------------------------------------------------
utils.reload_modules([rdl, dp, utils])


# -----------------------------------------------------------------------------
# Data splitting percentages
# -----------------------------------------------------------------------------
test_split = 0.1
validation_split = 0.1


# -----------------------------------------------------------------------------
# Data Shuffling
# -----------------------------------------------------------------------------
data_shuffling = True;
data_shuffling_seed = 32

# -----------------------------------------------------------------------------
# Raw data fetching
# -----------------------------------------------------------------------------
users = rdl.get_users() # Fetch users
activities = rdl.get_activities() # Read activities file
likes = rdl.get_likes() # Read likes file

# -----------------------------------------------------------------------------
# Gets empty rating "value"
# -----------------------------------------------------------------------------
def get_empty_rating():
    return np.nan
    #return -likes['interestlevel'].max()

# -----------------------------------------------------------------------------
# Fetch all the rows from likes file for user whose id is `user_id`
# -----------------------------------------------------------------------------
def get_likes_from_user(user_id):
    return likes.loc[likes['user_id'] == user_id]


# -----------------------------------------------------------------------------
# Gets the row number in the rating matrix attached to the `activity_id`
# -----------------------------------------------------------------------------
def get_row_number_by_activity_id(activity_id):
    return activity_id - 1;


# -----------------------------------------------------------------------------
# Fetch the ratings
#
# The returned value is a list where each element is a triplet (U,A,R)
# as follows:
#   U   : User id;
#   A   : Activity row number;
#   R   : Rating of the user U to the activity A.
# -----------------------------------------------------------------------------
def fetch_ratings():
    ratings = []
    rows_number = users.shape[0]
    columns_number = activities.shape[0]
    for user_id in range(rows_number):
        for _, row in get_likes_from_user(user_id).iterrows():
            activity_row_number = get_row_number_by_activity_id(row['atividade_id'])
            ratings.append((user_id, activity_row_number, row['interestlevel']))
    return ratings;


# -----------------------------------------------------------------------------
# Split data by test, validation and train clusters based on the number of
# ratings which is given by `dataset_size`
#
# Each cluster is a list where each element is a triplet (U,A,R)
# as follows:
#   U   : User id;
#   A   : Activity row number;
#   R   : Rating of the user U to the activity A.
# -----------------------------------------------------------------------------
def split_data(dataset_size):
    indices = list(range(dataset_size))
    test_size = int(np.floor(test_split * dataset_size))
    validation_size = int(np.floor(validation_split * (dataset_size - test_size)))
    if data_shuffling:
        np.random.seed(data_shuffling_seed)
        np.random.shuffle(indices)
    test_indices = indices[:test_size]
    validation_indices = indices[test_size:test_size+validation_size]
    train_indices = indices[test_size+validation_size:]
    return test_indices, validation_indices, train_indices


# -----------------------------------------------------------------------------
# Fills a rating matrix based on the `ratings` found in `indices`
#
# The returned value is a MxN matrix featuring the following properties:
#   M   : number of users;
#   N   : number of activities;
#   r_mn: rating of the user m to the activity n.
# -----------------------------------------------------------------------------
def fill_matrix(indices, ratings):
    rows_number = users.shape[0]
    columns_number = activities.shape[0]
    matrix = np.full((rows_number, columns_number), get_empty_rating(), dtype="float")
    for index in indices:
        user_id, activity_id, rating = ratings[index]
        matrix[user_id][activity_id] = rating
    return matrix


# -----------------------------------------------------------------------------
# Builds training, validation and testing matrix.
#
# The returned value is a triplet (TR, VAL, TE) where each element is a MxN
# matrix featuring the following properties:
#   M   : number of users;
#   N   : number of activities;
#   r_mn: rating of the user m to the activity n.
# -----------------------------------------------------------------------------
def build_matrices():
    ratings = fetch_ratings()
    test_indices, validation_indices, train_indices = split_data(len(ratings))

    test_matrix = fill_matrix(test_indices, ratings) # Build test rating matrix
    validation_matrix = fill_matrix(validation_indices, ratings) # Build validation rating matrix
    train_matrix = fill_matrix(train_indices, ratings) # Build train rating matrix

    return train_matrix, validation_matrix, test_matrix


# -----------------------------------------------------------------------------
# Builds and returns train matrix already 0 centered and scaled to [-1,1]
# -----------------------------------------------------------------------------
def get_train_matrix():
    train_matrix, _, _ = build_matrices()
    # We are slicing the returned value because we want to discard the first
    # row of the ratings matrix as it is assigned to a control user and not
    # to a real user
    return dp.scale_ratings(-1, 1, dp.zero_center_ratings(train_matrix))[1:]
    #return train_matrix


# -----------------------------------------------------------------------------
# Builds and return validation ratings
# -----------------------------------------------------------------------------
def get_validation_ratings():
    validation_ratings = []
    ratings = fetch_ratings()
    _, validation_indices, _ = split_data(len(ratings))
    for index in validation_indices:
        validation_ratings.append(ratings[index])
    return validation_ratings


# -----------------------------------------------------------------------------
# Data preprocessor playground
# -----------------------------------------------------------------------------
get_train_matrix()[0]

'''
def compare(index):
    matrix = get_train_matrix()
    return (matrix[index], dp.zero_center_ratings(a)[index], dp.scale_ratings(-1, 1, matrix)[index])

dp.zero_center_ratings(get_train_matrix())

compare(1)

get_train_matrix()[1]
'''
