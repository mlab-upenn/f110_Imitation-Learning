#!/usr/bin/env python
from __future__ import print_function
from f110_gym.wrappers.imitation_wrapper import make_imitation_env
from f110_gym.f110_core import f110Env
import rospy
import threading
import numpy as np

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

env = f110Env()
obs = env.reset()
for i in range(1000):
    random_action = {"angle":0.2, "speed":1.0}
    obs, reward, done, info = env.step(random_action)
    print(obs.keys())
    if done:
        obs = env.reset() 
