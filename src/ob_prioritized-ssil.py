#!/usr/bin/env python
from __future__ import print_function

#f110_gym imports
from wrappers.imitation_wrapper import make_imitation_env
from f110_gym.f110_core import f110Env
from f110_gym.distributed.exp_sender import ExperienceSender

#common imports
from common.f110_repbuf import f110_PrioritizedReplayBuffer
from ob_ssil import SSIL_ob


#Misc Imports
import rospy, cv2, random, threading, torch, os, time, math
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
modelpath = '/home/nvidia/datasets/avfone/models/'

__author__ = 'Dhruv karthik <dhruvkar@seas.upenn.edu>'

class PrioritizedSSIL_ob(SSIL_ob):
    """
    Similar to SSIL_ob but allows replay buffer prioritization
    """
    def __init__(self):
        SSIL_ob.__init__(self)
        self.repbuf = f110_PrioritizedReplayBuffer()

    def run_policy(self):
        """ Uses self.model to run the policy onboard & adds experiences to the prioritized replay buffer """
        env = make_imitation_env()
        obs_dict = env.reset()
        while True:
            action = self.get_action(self.gymobs_to_inputdict(obs_dict))
            next_obs_dict, reward, done, info = env.step(action)
            if info.get("record"):
                self.record = True
                ret_dict = self.oracle.fix(obs_dict)
                self.repbuf.add(ret_dict, action, reward, done)
            else:
                self.record = False

            obs_dict = next_obs_dict
            if done:
                obs_dict = env.reset() 