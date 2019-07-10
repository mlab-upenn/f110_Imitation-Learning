import os, torch, json, cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from data_utils import Data_Utils
import pandas as pd

class SteerDataset(Dataset):
    """
    Steer Dataset: Returns cropped image from dashcam + steering angle
    """
    def __init__(self, transforms=None):
        """
        transforms (callable, optional): Optional transforms applied on samples
        """
        super(SteerDataset, self).__init__()
        self.abs_path = json.load(open("params.txt"))["abs_path"] 
        self.steer_df = pd.read_csv(self.abs_path + "/data.csv")
        self.transforms = transforms
        self.dutils = Data_Utils()
    
    def __len__(self):
        return len(self.steer_df)
    
    def __getitem__(self, idx):
        """
        Returns tuple (cropped_image(Tensor, C x H x W), steering angle (float 1x1 tensor))
        """
        img_name, angle = self.steer_df.iloc[idx, 0], self.steer_df.iloc[idx, 1]
        cv_img = cv2.imread(self.abs_path + '/' + img_name)
        angle = np.array([angle])
        angle = torch.from_numpy(angle).float()
        #cropping image + turning to tensor
        img_tensor = self.dutils.cv2_to_croppedtensor(cv_img)
        if self.transforms:
            img_tensor = self.transforms(img_tensor)

        return (img_tensor, angle)
