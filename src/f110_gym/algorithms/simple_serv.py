#!/usr/bin/env python
from __future__ import print_function
from f110_gym.wrappers.imitation_wrapper import make_imitation_env
from f110_gym.f110_core import f110env
from distributed.exp_sender import ExperienceSender
import rospy, cv2
import threading
import numpy as np

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'

env = f110env()
