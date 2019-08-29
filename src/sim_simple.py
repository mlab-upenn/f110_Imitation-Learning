#!/usr/bin/env python
from __future__ import print_function
import airsim
import cv2, sys, os
from f110_gym.sim_f110_core import SIM_f110Env
import pdb
import numpy as np

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'


def main():
    env = SIM_f110Env()
    obs = env.reset()
    count = 0
    while True:
        random_action = {"angle":0.0, "speed":1.0}
        obs, reward, done, info = env.step(random_action)
        
        #display cv_img
        cv_img = obs["img"][0]
        cv2.imshow('latestimg', cv_img)

        #print lidar
        lidar = obs["lidar"]
        pdb.set_trace()

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
        if done:
            print("ISDONE")
            obs = env.reset()  

if __name__ == '__main__':
    main()