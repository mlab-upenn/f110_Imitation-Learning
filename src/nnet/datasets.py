import os, torch, cv2, pickle
import numpy as np
from nnet.Data_Utils import Data_Utils
from torch.utils.data import Dataset
from torchvision import transforms, utils
import pandas as pd

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class SteerDataset(Dataset):
    """
    Returns img, steering_angle pairs CVS
    """
    def __init__(self, datapath, transforms=None):
        """
        datapath: source of final processed & augmented data
        """
        super(SteerDataset, self).__init__()
        self.datapath = datapath
        self.dutils = Data_Utils()
        self.df = self.dutils.get_df(self.datapath)
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns dictionary {"img": Tensor, C x H x W, "angle":float 1-Tensor}
        """
        cv_img, df_row = self.dutils.df_data_fromidx(self.datapath, self.df, idx)

        if self.transforms:
            cv_img = self.transforms(cv_img)

        ts_angle = torch.Tensor([df_row[1]]).float()
        ts_img = torch.from_numpy(cv_img).permute(2, 0, 1).float() #size (C x H x W)

        data_dict = {"img":ts_img, "angle":ts_angle}
        return data_dict

class SteerDataset_ONLINE(Dataset):
    """
    Very Similar to SteerDataset, but process pkl files
    """
    def __init__(self, pkl_path, transforms=None):
        """
        pkl_path: source of pkl file to read from 
        """
        super(SteerDataset_ONLINE, self).__init__()
        data_in = open(pkl_path, 'rb')
        self.data_array = pickle.load(data_in)

    def __len__(self):
        return len(self.data_array)
    
    def __getitem__(self, idx):
        """
        Returns dictionary {"img":Tensor, C X H x W, "angle":float 1-Tensor}
        """
        data_dict = self.data_array[idx]
        cv_img = data_dict.get("img")
        ts_angle = torch.Tensor([data_dict["steer"]["steering_angle"]]).float()
        ts_img = torch.from_numpy(cv_img).permute(2, 0, 1).float()
        new_data_dict = {"img":ts_img, "angle":ts_angle}
        return new_data_dict