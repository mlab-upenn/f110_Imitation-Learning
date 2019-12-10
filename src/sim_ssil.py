from __future__ import print_function
import airsim
import cv2, sys, os, random
from f110_gym.sim_f110_core import SIM_f110Env
import numpy as np

#Common Imports
from common.datasets import SteerDataset
from common.models import NVIDIA_ConvNet
from oracles.FGM import FGM

#torch imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'


def seed_env():
    seed = 6582
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 

RENDER = False
seed_env()

def main():
    env = SIM_f110Env()
    obs = env.reset()

    #Load Net
    net = NVIDIA_ConvNet().cuda()
    net.load_state_dict(torch.load('sim_net'))
    net = net.cpu()
    while True:
        cv_img = obs["img"][0]
        lidar = obs["lidar"]
        lidar = lidar[..., 0:2]

        if RENDER:
            cv2.imshow("Forward", cv_img)
            env.render_lidar2D(lidar)

        #Pass through network
        ts_img = torch.from_numpy(cv_img).permute(2, 0, 1).float()
        ts_angle = net(ts_img[None])

        #Take Action
        action = {"angle": ts_angle.item() * np.pi/180.0, "speed":0.3}
        obs, _, done, _ = env.step(action)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
        if done:
            print("ISDONE")
            obs = env.reset()  

if __name__ == '__main__':
    main()