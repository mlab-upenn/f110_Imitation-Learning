import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bashplotlib.histogram import plot_hist
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

def main():
    du = Data_Utils()
    du.show_steer_angle_hist(w_matplotlib=True)

if __name__ == '__main__':
    main()