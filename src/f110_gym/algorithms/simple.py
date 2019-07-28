#!/usr/bin/env python
from __future__ import print_function
from f110_gym.wrappers.imitation_wrapper import make_imitation_env
from f110_gym.f110_core import f110Env
import rospy, cv2
import threading
import numpy as np

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

env = f110Env()
obs = env.reset()
while True:
    random_action = {"angle":0.2, "speed":1.0}
    obs, reward, done, info = env.step(random_action)
    cv_img = obs["img"]
    cv2.imshow('latestimg', cv_img)
    cv2.waitKey(2)
    if done:
        obs = env.reset() 
