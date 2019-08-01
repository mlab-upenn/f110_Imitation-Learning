from __future__ import print_function
from f110_gym.distributed.exp_server import ExperienceServer
import cv2, random, threading, msgpack, os, time
from common.datasets import SteerDataset_ONLINE
from common.models import NVIDIA_ConvNet
from common.augs import *

#nnet & logging imports
from nnet.Online import Online
from nnet.Metric_Visualizer import Metric_Visualizer

#misc imports
from functools import partial
import msgpack_numpy as m
import numpy as np

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

class SAVE_server(object):
    """
    Serverside Class that Saves, Augments - Very, Exciting (S.A.V.E)
    """
    def __init__(self):
        self.onl = Online()
        self.vis = Metric_Visualizer()
        self.serv = ExperienceServer(self.ob_callback, deserialize_obs(), 4)
        self.exp_path = self.get_exp_path()
        m.patch()
    
    def get_exp_path(self):
        exp_path = os.path.join("/home/dhruvkar/datasets/avfone", "runs", "0", "exp")
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        print("EXPERIENCE PATH:", exp_path)
        return exp_path

    def ob_callback(self, obs_array):
        pkl_name = self.onl.save_obsarray_to_pickle(obs_array, os.path.join(self.exp_path, 'data'))

        self.vis.vid_from_pklpath(os.path.join(self.exp_path, 'data', pkl_name), 0, 0, show_steer=True, units='rad', live=True)

        return [b'Yeet']
    
    def start_serv(self):
        self.serv.start()
        self.serv.join()
    
saves = SAVE_server()
saves.start_serv()