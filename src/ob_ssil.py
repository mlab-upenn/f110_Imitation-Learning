#!/usr/bin/env python
from __future__ import print_function

#f110_gym imports
from f110_gym.wrappers.imitation_wrapper import make_imitation_env
from f110_gym.f110_core import f110Env
from f110_gym.distributed.exp_sender import ExperienceSender

#Common Imports
from f110_repbuf import f110_ReplayBuffer

#Misc
import rospy, cv2, random, threading, torch, os, time
from collections import deque
from nnet.models import NVIDIA_ConvNet
from oracles.FGM import FGM
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
modelpath = '/home/nvidia/datasets/avfone/models/'

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'

class SSIL_ob(object):
    """
    Class that enables Self-Supervised Imitation Learning (runs ob=on-board F110)
    """
    def __init__(self):
        self.model  = NVIDIA_ConvNet().to(device)
        self.model.eval()
        self.oracle = FGM()
        self.serv_sender = ExperienceSender()
        self.repbuf = f110_ReplayBuffer()
        self.env = make_imitation_env()

    def gymobs_to_inputdict(self, obs_dict):
        """ Utility to convert gym observation to an input dictionary into the neural network"""
        input_dict = {}
        cv_img = obs_dict["img"]
        ts_img = torch.from_numpy(cv_img).permute(2, 0, 1).float()
        ts_img = ts_img[None]
        input_dict["img"] = ts_img.to(device)
        return input_dict

    def get_action(self, input_dict):
        """ Gets action from self.model & returns action_dict for gym"""
        out_dict = self.model(input_dict)
        angle_pred = out_dict["angle"].item()
        vel = 1.0
        return {"angle":angle_pred, "speed":1.0}

    def run_policy(self):
        """ Uses self.model to run the policy onboard & adds experiences to the replay buffer """
        env = make_imitation_env()
        obs_dict = env.reset()
        while True:
            action = self.get_action(self.gymobs_to_inputdict(obs_dict))
            next_obs_dict, reward, done, info = env.step(action)
            if info.get("record"):
                obs_dict = self.oracle.fix(obs_dict)
                self.repbuf.add(obs_dict, action, reward, done)
            obs_dict = next_obs_dict
            if done:
                obs_dict = env.reset()
    
    def save_model(self, model_dump):
        if not os.path.exists(modelpath):
            os.makedirs(modelpath)
        f = open(os.path.join(modelpath, 'model'), 'w')
        f.write(model_dump)
        f.close()

    def update_nn(self):
        if os.path.exists(modelpath):
            self.model.load_state_dict(torch.load(modelpath))
        self.model.to(device)
        self.model.to(eval)
        print("LOADED MODEL")
        print("DEVICE:{device}".format(device=device))
        print("MODEL:")
        print(self.model)

    def server_callback(self, reply_dump):
        """When the server returns something, this function will get called from another thread"""
        self.save_model(reply_dump[0])
        self.update_nn()

    def send_batches(self):
        """Handles sending batches of experiences sampled from the replay buffer to the Server for training """
        while True:
            try:
                obs_array, _, _, _,  = self.repbuf.sample()
                sender.send(obs_array, self.env.serialize_obs(), self.server_callback)
            except:
                pass
            time.sleep(2)

def main():
    ssil = SSIL_ob()
    #Launch two threads
    gym_thread = threading.Thread(target=ssil.run_policy)
    server_thread = threading.Thread(target=ssil.send_batches)
    gym_thread.start()
    server_thread.start()
    gym_thread.join()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        rospy.signal_shutdown('Done')
        pass