#!/usr/bin/env python
from __future__ import print_function
from f110_gym.wrappers.imitation_wrapper import make_imitation_env
from f110_gym.f110_core import f110env
from distributed.exp_sender import ExperienceSender
import rospy, cv2, random, threading
import numpy as np

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'

def update_nn(reply):
    print(reply)

env = f110env()
obs = env.reset()
obs_array = []
sender = ExperienceSender()
for i in range(1000):
    random_action = {"angle":random.uniform(-0.3, 0.3), "speed":1.0}
    obs, reward, done, info = env.step(random_action)
    obs_array.append(obs)
    if i % 50:
        sender.send_obs(obs_array, env.serial_func(), update_nn, header_dict={'env':f110env})
    if done:
        obs = env.reset() 