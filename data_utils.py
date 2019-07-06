import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bashplotlib.histogram import plot_hist
from steer_dataset import SteerDataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import json

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

def main():
    du = Data_Utils()
    du.show_steer_angle_hist()

if __name__ == '__main__':
    main()