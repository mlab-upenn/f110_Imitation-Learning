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
    Copies and executes expert policy while frequently sending batches over to the server
    """
    def __init__(self):
        self.serv_sender = ExperienceSender()
        self.record = False
        self.env = make_imitation_env()
        self.oracle = FGM()
        
        #store observations for sender (sending off to server for training)
        self.sender_buffer = deque(maxlen=20)

    def get_action(self, obs_dict):
        """ Gets action from self.oracle returns action_dict for gym"""
        ret_dict = self.oracle.fix(obs_dict)
        return ret_dict

    def run_policy(self):
        """ Uses self.oracle to run the policy onboard"""
        env = make_imitation_env(skip=2)
        obs_dict = env.reset()
        self.sender_buffer.append(obs_dict)
        while True:
            action = self.get_action(obs_dict)
            nobs_dict, reward, done, info = env.step(action)
            if info.get("record"):
                self.sender_buffer.append(nobs_dict)
            obs_dict = nobs_dict
            if done:
                obs_dict = env.reset()
    
    def server_callback(self, reply_dump):
        """When the server returns something, this function will get called from another thread"""
        print("YEET")
        pass

    def send_batches(self):
        """Handles sending batches of experiences sampled from the replay buffer to the Server for training """
        while True:
            if len(self.sender_buffer) > 21:
                obs_array = []
                for i in range(20):
                    #FYI:popping from opposite side of deque is thread-safe
                    obs_array.append(self.sender_buffer.popleft())
                self.serv_sender.send_obs(obs_array, self.env.serialize_obs(), self.server_callback)
            time.sleep(10)

def main():
    co = Copy_Oracle()
    server_thread = threading.Thread(target=co.send_batches)
    server_thread.daemon = True
    server_thread.start() #run recording on another thread
    co.run_policy() #run policy on the main thread
    server_thread.join()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        rospy.signal_shutdown('Done')
        pass