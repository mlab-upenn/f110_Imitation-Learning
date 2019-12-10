#!/usr/bin/env python
from __future__ import print_function
import airsim
import cv2, sys, os
from f110_gym.sim_f110_core import SIM_f110Env
from common.utils import cart_to_polar, vis_roslidar, polar_to_rosformat
from oracles.FGM import FGM
import pdb
import numpy as np

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'

def main():
    env = SIM_f110Env()
    obs = env.reset()
    for i in range(10):
        done = False

        obs = env.reset()  
        while not done:
            #display cv_img
            cv_img = obs["img"][0]
            cv2.imshow('latestimg', cv_img)

            #display lidar
            lidar = obs["lidar"]
            lidar = lidar[..., 0:2]
            env.render_lidar2D(lidar)

            action = {"angle":0., "speed":0.3}
            obs, reward, done, info = env.step(action)

            if cv2.waitKey(3) & 0xFF == ord('q'):
                break
        print("DONE - RESETTING")
if __name__ == '__main__':
    main()