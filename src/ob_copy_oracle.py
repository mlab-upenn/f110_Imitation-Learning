#!/usr/bin/env python
from __future__ import print_function

#f110_gym imports
from wrappers.imitation_wrapper import make_imitation_env
from f110_gym.f110_core import f110Env
from f110_gym.distributed.exp_sender import ExperienceSender

#Common Imports
from oracles.FGM import FGM

#Misc
import rospy, cv2, random, threading, os, time
from collections import deque
import numpy as np
modelpath = '/home/nvidia/datasets/avfone/models/'

__author__ = 'Dhruv karthik <dhruvkar@seas.upenn.edu>'

class Copy_Oracle(object):
    """
    Enables expert policy execution for collecting training data
    """
    def __init__(self):
        self.serv_sender = ExperienceSender()
        self.record = False
        self.env = make_imitation_env()
        
        #this is where we store observations for sender
        self.sender_buffer = deque(maxlen=20)

    