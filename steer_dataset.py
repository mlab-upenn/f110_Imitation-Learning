import os, torch, json, cv2
from torch.utils.data import Dataset
from torchvision import transforms, utils
import pandas as pd

class SteerDataset(Dataset):
    """
    Steer Dataset: Returns cropped image from dashcam + steering angle
    """
    def __init__(self, transforms):
        """
        transforms (callable, optional): Optional transforms applied on samples
        """
        super(SteerDataset, self).__init__()
        self.abs_path = json.load(open("params.txt"))["abs_path"] 
        self.steer_df = pd.read_csv(self.abs_path + "/data.csv")
        self.transforms = transforms
    
    def __len__(self):
        return len(self.steer_df)
    
    def __getitem__(self, idx):
        """
        Returns tuple (cropped_image(Tensor, C x H x W), steering angle (float))
        """
        img_name, angle = self.steer_df.iloc[idx, 0], self.steer_df.iloc[idx, 1]
        cv_img = cv2.imread(self.abs_path + '/' + img_name)
        
        #cropping image + turning to tensor
        cv_crop = cv_img[200:, :, :]
        img_tensor = torch.from_nummpy(cv_crop) #size (H x W x C)
        img_tensor = img_tensor.permute(2, 0, 1)#size (C x H x W)

        if self.transforms:
            img_tensor = self.transforms(img_tensor)

        return (img_tensor, float(angle))