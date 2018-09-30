import sys
sys.path.insert(0, '/Users/utxeee/Desktop/deep-learning-com-perfis/code/pyTorch/utils')
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import data_preprocessor as dp
import utils

# -----------------------------------------------------------------------------
# Force custom modules reloading otherwise changes in custom modules after
# loading will not be taken into account herein!
# -----------------------------------------------------------------------------
utils.reload_modules([dp])


# -----------------------------------------------------------------------------
# Class: RatingsMatrixDataset
#
# Description: Dataloader for rating matrix dataset
# -----------------------------------------------------------------------------
class RatingsMatrixDataset(Dataset):
    def __init__(self):
        ratings_matrix = dp.build_preprocessed_ratings_matrix()
        self.len = ratings_matrix.shape[0]
        self.x_data = torch.from_numpy(ratings_matrix)

    def __getitem__(self, index):
        return self.x_data[index], self.x_data[index]


    def __len__(self):
        return self.len


# -----------------------------------------------------------------------------
# Instantiating the dataset and passing to the dataloader
# -----------------------------------------------------------------------------
def get_data_loader():
    return DataLoader(RatingsMatrixDataset(), batch_size=1, shuffle=True, num_workers=1)
