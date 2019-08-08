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
import rospy, cv2, random, threading, torch, os, time, math, copy
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
    
    def get_repbuf_entry(self, obs_dict, action, next_obs_dict, reward, done, info):
        """ Returns an entry for the replay buffer with specific modifications reqd. by this algorithm"""
        old_steer = obs_dict["steer"]["angle"]
        new_obs_dict = self.oracle.fix(obs_dict)

        #L1 Norm of steering angle to indicate priority
        new_steer = new_obs_dict["steer"]["angle"]
        new_steer = new_steer * 180.0/math.pi
        old_steer = old_steer * 180.0/math.pi
        l1norm = math.fabs(new_steer - old_steer)
        l2norm = math.sqrt((new_steer-old_steer)**2 )
        entry = (obs_dict, action, reward, done, l2norm)
        print("PRIORITY:", l1norm)
        return entry

    def run_policy(self):
        """ Uses self.model to run the policy onboard & adds experiences to the prioritized replay buffer """
        env = make_imitation_env()
        obs_dict = env.reset()
        while True:
            action = self.get_action(self.gymobs_to_inputdict(obs_dict))
            next_obs_dict, reward, done, info = env.step(action)
            if info.get("record"):
                self.record = True
                entry = self.get_repbuf_entry(obs_dict, action, next_obs_dict, reward, done, info)
                self.repbuf.add(*entry)
            else:
                self.record = False

            obs_dict = next_obs_dict
            if done:
                obs_dict = env.reset() 

def main():
    ssil = PrioritizedSSIL_ob()
    server_thread = threading.Thread(target=ssil.send_batches)
    server_thread.daemon = True
    server_thread.start() #run recording on another thread
    ssil.run_policy() #run policy on the main thread
    server_thread.join()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        rospy.signal_shutdown('Done')
        pass
