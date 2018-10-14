import sys
sys.path.insert(0, '/Users/utxeee/Desktop/deep-learning-com-perfis/code/pyTorch/utils')
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import data_preprocessor as dp
import data_splitting as ds
import utils

# -----------------------------------------------------------------------------
# Force custom modules reloading otherwise changes in custom modules after
# loading will not be taken into account herein!
# -----------------------------------------------------------------------------
utils.reload_modules([dp, ds, utils])


# -----------------------------------------------------------------------------
# Class: TrainDataset
#
# Description: Dataloader for train dataset
# -----------------------------------------------------------------------------
class TrainDataset(Dataset):
    def __init__(self):
        #ratings_matrix = dp.build_preprocessed_ratings_matrix()
        train_matrix = ds.get_train_matrix()
        self.len = train_matrix.shape[0]
        self.train_matrix = torch.from_numpy(train_matrix)
        # By default the train matrix tensor is a double after converting it
        # from numpy but the model is expecting a float, hence, we cast the
        # type here!
        self.train_matrix = self.train_matrix.type(torch.FloatTensor)

    def __getitem__(self, index):
        cloned_item = self.train_matrix[index].clone()
        cloned_item[torch.isnan(cloned_item)] = 0
        return cloned_item, self.train_matrix[index]
        #return self.train_matrix[index], self.train_matrix[index]

    def fill_missing_ratings(item):
        cloned_item = item.clone()
        cloned_item[torch.isnan(cloned_item)] = 0
        return cloned_item


    def __len__(self):
        return self.len


# -----------------------------------------------------------------------------
# Class: ValidationDataset
#
# Description: Dataloader for train matrix dataset
# -----------------------------------------------------------------------------
class ValidationDataset(Dataset):
    def __init__(self):
        train_matrix = ds.get_train_matrix()
        validation_ratings = ds.get_validation_ratings()
        self.len = len(validation_ratings)
        self.validation_ratings = validation_ratings
        self.train_matrix = torch.from_numpy(train_matrix)
        # By default the train matrix tensor is a double after converting it
        # from numpy but the model is expecting a float, hence, we cast the
        # type here!
        self.train_matrix = self.train_matrix.type(torch.FloatTensor)

    def __getitem__(self, index):
        user_id, activity_id, rating = self.validation_ratings[index]
        return self.train_matrix[user_id], (activity_id, rating)


    def __len__(self):
        return self.len

# -----------------------------------------------------------------------------
# Builds a dataloader for train data
# -----------------------------------------------------------------------------
def get_train_data():
    return DataLoader(TrainDataset(), batch_size=1, shuffle=True, num_workers=1)


# -----------------------------------------------------------------------------
# Builds a dataloader for validation data
# -----------------------------------------------------------------------------
def get_validation_data():
    return DataLoader(ValidationDataset(), batch_size=1, shuffle=True, num_workers=1)


# -----------------------------------------------------------------------------
# Data loader playground
# -----------------------------------------------------------------------------

#len(get_validation_data().dataset)

#for batch_idx, (data, target) in enumerate(get_train_data()):
    #changed = data[0][torch.isnan(data[0])] = 0

#    print(target, data)
