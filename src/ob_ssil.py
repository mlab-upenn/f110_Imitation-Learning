#!/usr/bin/env python
from __future__ import print_function
from f110_gym.wrappers.imitation_wrapper import make_imitation_env
from f110_gym.f110_core import f110Env
from f110_gym.distributed.exp_sender import ExperienceSender
import rospy, cv2, random, threading
from oracles.FGM import FGM
import numpy as np

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'

class SSIL_ob(object):
    """
    Class that enables Self-Supervised Imitation Learning (runs ob=on-board F110)
    """
    def __init__(self):
        