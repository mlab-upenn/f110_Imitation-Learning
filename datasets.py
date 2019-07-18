import os, torch, cv2
import numpy as np
from Data_Utils import Data_Utils
from torch.utils.data import Dataset
from torchvision import transforms, utils
from Data_Utils import Data_Utils
import pandas as pd

class SteerDataset(Dataset):
    """
    Returns img, steering_angle pairs
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

        ts_angle = torch.Tensor([df_row[1]])
        ts_img = torch.from_numpy(cv_img).permute(2, 0, 1).float() #size (C x H x W)

        data_dict = {"img":ts_img, "angle":ts_angle}
        return data_dict