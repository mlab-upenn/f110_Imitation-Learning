import os, torch, cv2, pickle
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

# class SteerDataset(Dataset):
#     """
#     Returns img, angle pairs CVS
#     """
#     def __init__(self, datapath, transforms=None):
#         """
#         datapath: source of final processed & augmented data
#         """
#         super(SteerDataset, self).__init__()
#         self.datapath = datapath
#         self.dutils = Data_Utils()
#         self.df = self.dutils.get_df(self.datapath)
#         self.transforms = transforms
    
#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         """
#         Returns dictionary {"img": Tensor, C x H x W, "angle":float 1-Tensor}
#         """
#         cv_img, df_row = self.dutils.df_data_fromidx(self.datapath, self.df, idx)

#         if self.transforms:
#             cv_img = self.transforms(cv_img)

#         ts_angle = torch.Tensor([df_row[1]]).float()
#         ts_img = torch.from_numpy(cv_img).permute(2, 0, 1).float() #size (C x H x W)

#         data_dict = {"img":ts_img, "angle":ts_angle}
#         return data_dict

# class SteerDataset_ONLINE(Dataset):
#     """
#     Very Similar to SteerDataset, but process pkl files
#     """
#     def __init__(self, pkl_path, transforms=None):
#         """
#         pkl_path: source of pkl file to read from 
#         """
#         super(SteerDataset_ONLINE, self).__init__()
#         data_in = open(pkl_path, 'rb')
#         self.transforms = transforms
#         self.data_array = pickle.load(data_in)

#     def __len__(self):
#         return len(self.data_array)
    
#     def __getitem__(self, idx):
#         """
#         Returns dictionary {"img":Tensor, C X H x W, "angle":float 1-Tensor}
#         """
#         data_dict = self.data_array[idx]
#         if self.transforms:
#             data_dict = self.transforms(data_dict)
#         cv_img = data_dict.get("img")
#         ts_angle = torch.Tensor([data_dict["steer"]["angle"]]).float()
#         ts_img = torch.from_numpy(cv_img).permute(2, 0, 1).float()
#         new_data_dict = {"img":ts_img, "angle":ts_angle}
#         return new_data_dict

class SteerDataset(Dataset):
    def __init__(self, folderpath, transforms=None):
        """
        folderpath: String of directory containing pkl files
        Each pkl should be a dictionary:
        {
            "obs": observation,
            "action": action
        }
        """
        super(SteerDataset, self).__init__()
        pkl_list = os.listdir(folderpath)
        self.obs_array = []
        self.action_array = []
        for pkl_name in pkl_list:
            with open(os.path.join(folderpath, pkl_name), 'rb') as f:
                pkl_dict = pickle.load(f)
                self.obs_array.append(pkl_dict["obs"])
                self.action_array.append(pkl_dict["action"])

    def __len__(self):
        return len(self.obs_array)

    def __getitem__(self, idx):
        """
        Returns tuple (img- C x H x W Tensor, angle-float 1-Tensor)
        """
        obs = self.obs_array[idx]
        action = self.action_array[idx]
        cv_img = obs.get("img")[0]
        cv2.waitKey(0)
        ts_angle = torch.Tensor([action.get("angle")]).float()
        ts_img = torch.from_numpy(cv_img).permute(2, 0, 1).float()
        return (ts_img, ts_angle)