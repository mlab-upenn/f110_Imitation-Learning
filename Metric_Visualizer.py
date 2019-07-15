import os, cv2, math, sys, json, torch, pdb
from data_utils import Data_Utils
import pandas as pd 
from tensorboardX import SummaryWriter

class Metric_Visualizer(object):
    """
    Visualize metrics in Tensorboard
    """
    def __init__(self, step_id):
        self.params_dict = json.load(open("steps.json"))["params"]
        
        #Create SummaryWriter for tensorboard that updates frequently
        logdir = os.path.join(self.params_dict["abs_path"], self.params_dict["sess"])
        self.writer = SummaryWriter(logdir=os.path.join())
