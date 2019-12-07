import os, random, time, pickle

#torch imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("cannot fully use Trainer without tensorboardX")

__author__ = "Dhruv Karthik <dhruvkar@seas.upenn.edu>"

device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 
