from common.datasets import SteerDataset
import os

FOLDERPATH = './sim_train'
dset = SteerDataset(os.path.abspath(FOLDERPATH))
