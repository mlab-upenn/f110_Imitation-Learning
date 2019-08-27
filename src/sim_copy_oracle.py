from __future__ import print_function

#Common & Wrapper Imports
from wrappers.imitation_wrapper import sim_make_imitation_env
from oracles.FGM import FGM

import airsim

#Misc
import cv2, random, threading, os, time
from collections import deque
import numpy as np

__author__ = 'Dhruv karthik <dhruvkar@seas.upenn.edu>'

class SIM_Copy_Oracle(object):
    """
    Copies & executes expert policy on AirSim f110
    """
    def __init__(self):
        self.oracle = FGM()
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)

    def get_action(self, obs_dict):
        """ Gets action from self.oracle returns action_dict for gym"""
        ret_dict = self.oracle.fix(obs_dict)
        act = {"angle":ret_dict["steer"]["angle"], "speed":0.7}
        return act

    def run_policy(self):
        """Uses self.oracle to run the policy onboard"""
        car_controls = airsim.CarControls()

    # def run_policy(self): #     """ Uses self.oracle to run the policy onboard"""
    #     env = make_imitation_env(skip=2)
    #     obs_dict = env.reset()
    #     self.sender_buffer.append(obs_dict)
    #     while True:
    #         action = self.get_action(obs_dict)
    #         nobs_dict, reward, done, info = env.step(action)
    #         if info.get("record"):
    #             self.sender_buffer.append(nobs_dict)
    #         obs_dict = nobs_dict
    #         if done:
    #             obs_dict = env.reset()

def main():
    co = SIM_Copy_Oracle()
    co.run_policy() #run policy on the main thread

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        rospy.signal_shutdown('Done')
        pass
