from __future__ import print_function

#Common imports
import cv2, random, os, time
from common.datasets import SteerDataset_ONLINE
from common.models import NVIDIA_ConvNet
from common.steps import session
from common.Trainer import Trainer
import numpy as np

trainer = Trainer()
exp_path = trainer.get_exp_path()
datapath = os.path.join(exp_path, 'data')
pkl_list = os.listdir(datapath)
print(pkl_list)
trainer.view_model(pkl_list)
#trainer.train_model(pkl_list)