import os, torch, cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bashplotlib.histogram import plot_hist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import json

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

class Data_Utils(object):
    """
    Fun & Useful functions for dealing with Steer Data
    """
    def __init__(self):
        self.abs_path = json.load(open("params.txt"))["abs_path"]
        csv_file_path = self.abs_path + "/data.csv"
        self.steer_df = pd.read_csv(csv_file_path)

    def show_steer_angle_hist(self, w_matplotlib=False):
        """
        Shows a histogram illustrating distribution of steering angles
        w_matplotlib: boolean to trigger plotting with matplotlib. will otherwise plot to bash with bashplotlib
        """
        angle_column = self.steer_df.iloc[:, 1].values
        num_bins = 20
        if(w_matplotlib):
            #save plot with matplotlib
            plt.hist(angle_column, num_bins, color='green')
            plt.title("Distribution of steering angles (rads)")
            plt.savefig('steer_hist.png')
        else:
            #Plots in bash terminal
            num_bins = 20
            plot_hist(angle_column, num_bins, binwidth=0.01, colour='green', title='Distribution of steering angles (rads)', xlab=True, showSummary=True)
    
    def get_dataloaders(self, batch_size):
        """
        Returns a training & validation dataloader
        """
        steer_dataset = SteerDataset()
        vsplit = 0.2 #80, 20 split

        #idxs for train & valid
        dset_size = len(steer_dataset)
        idxs = list(range(dset_size))
        split = int(np.floor(vsplit * dset_size))
        np.random.shuffle(idxs)
        train_idxs, val_idxs = idxs[split:], idxs[:split]

        #Using SubsetRandomSampler but should ideally sample equally from each steer angle to avoid distributional bias
        train_sampler = SubsetRandomSampler(train_idxs)
        val_sampler = SubsetRandomSampler(val_idxs)

        train_dataloader = DataLoader(steer_dataset, batch_size=batch_size, sampler=train_sampler)
        valid_dataloader = DataLoader(steer_dataset, batch_size=batch_size, sampler=val_sampler)

        return train_dataloader, valid_dataloader
    
    def cv2_to_croppedtensor(self, cv_img):
        #cropping image + turning to tensor
        cv_crop = cv_img[200:, :, :]
        img_tensor = torch.from_numpy(cv_crop).float()#size (H x W x C)
        img_tensor = img_tensor.permute(2, 0, 1)#size (C x H x W)
        return img_tensor


    def combine_csvs(self, folder_list):
        df = pd.concat([pd.read_csv(f+'/data.csv') for f in folder_list])
        path= os.path.join(self.abs_path , 'merged.csv')
        df.to_csv(path)


def main():
    du = Data_Utils()
    du.show_steer_angle_hist()

if __name__ == '__main__':
    main()