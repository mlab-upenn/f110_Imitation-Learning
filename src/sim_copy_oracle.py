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
    angle_min, angle_incr = env.sensor_info.get("angle_min"), env.sensor_info.get("angle_incr")
    fgm = FGM(angle_min, angle_incr)
    obs = env.reset()
    while True:
        #display cv_img
        cv_img = obs["img"][0]
        cv2.imshow('latestimg', cv_img)

        #display lidar
        lidar = obs["lidar"]
        lidar = lidar[..., 0:2]
        env.render_lidar2D(lidar)
        ranges, theta = cart_to_polar(lidar)
        ranges = polar_to_rosformat(angle_min, -1.0 * angle_min, angle_incr, theta, ranges)
        vis_roslidar(ranges, angle_min, angle_incr)

        fgm.act(ranges)
        action = {"angle":fgm.act(ranges), "speed":0.3}
        # action = {"angle":0.0, "speed":0.0}
        # print(action)
        obs, reward, done, info = env.step(action)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
        if done:
            print("ISDONE")
            obs = env.reset()  

if __name__ == '__main__':
    main()