from __future__ import print_function
from f110_gym.distributed.exp_server import ExperienceServer
import cv2, random, threading, msgpack, os, time
from common.datasets import SteerDataset_ONLINE
from common.models import NVIDIA_ConvNet
from common.Trainer import Trainer
#torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#nnet & logging imports
from nnet.Online import Online
from nnet.Metric_Visualizer import Metric_Visualizer
try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("cannot fully use Trainer without tensorboardX")

#misc imports
from functools import partial
import msgpack_numpy as m
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

def deserialize_obs():
    def _deser(multipart_msg):
        lidar = msgpack.loads(multipart_msg[0], encoding="utf-8")
        steer = msgpack.unpackb(multipart_msg[1], encoding="utf-8")
        md = msgpack.unpackb(multipart_msg[2])
        cv_img = multipart_msg[3]
        cv_img = np.frombuffer(cv_img, dtype=md[b'dtype'])
        cv_img = cv_img.reshape(md[b'shape'])
        obs_dict = {"img":cv_img, "lidar":lidar, "steer":steer}
        return obs_dict
    return _deser

class SSIL_server(object):
    """
    Serverside Class for Self Supervised Imitation Learning (runs on the server)
    """
    def __init__(self):
        self.onl = Online()
        self.vis = Metric_Visualizer()
        self.trainer = Trainer(sess_type="online")
        self.serv = ExperienceServer(self.ob_callback, deserialize_obs(), 4)
        self.exp_path = self.trainer.get_exp_path()
        self.modelpath = self.trainer.get_model_path()
        self.modelname = self.trainer.get_model_name()
        m.patch()

    def ob_callback(self, obs_array):
        pkl_name = self.onl.save_obsarray_to_pickle(obs_array, os.path.join(self.exp_path, 'data'))

        self.vis.vid_from_pklpath(os.path.join(self.exp_path, 'data', pkl_name), 0, 0, show_steer=True, units='rad', live=True)

        self.trainer.train_model([pkl_name])

        #Send model back
        with open(os.path.join(self.modelpath, "train_" + self.modelname), 'rb') as binary_file:
            model_dump = bytes(binary_file.read())
        return [model_dump]

    def start_serv(self):
        self.serv.start()
        self.serv.join()
    #####TRAIN FUNCTIONS ###################

ssil_serv = SSIL_server()
ssil_serv.start_serv()
