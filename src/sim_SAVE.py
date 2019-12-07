import airsim
import cv2, sys, os
from f110_gym.sim_f110_core import SIM_f110Env
from common.utils import cart_to_polar, vis_roslidar, polar_to_rosformat
from oracles.FGM import FGM
import pickle
import pdb
import numpy as np

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'

"""
The following file generates training data for an Imitation Learning Algorithm in Simulation.
It uses the FGM (Follow Gap Method) as an Oracle Policy
"""
RENDER = False
FOLDERPATH = './sim_train'
num_saves = 0

def save_data(obs, action):
    global num_saves
    if(num_saves == 0 and not os.path.exists(FOLDERPATH)):
        os.mkdir(FOLDERPATH)
    pkl_dict = {"obs":obs, "action":action}
    filename = f"{FOLDERPATH}/sim_{num_saves}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(pkl_dict, f)
    num_saves+=1
        
def main():
    env = SIM_f110Env()

    #1) Initialize Follow Gap Method Oracle Policy to select our actions
    angle_min, angle_incr = env.sensor_info.get("angle_min"), env.sensor_info.get("angle_incr")
    fgm = FGM(angle_min, angle_incr)

    #2) Begin GYM Training Loop
    obs = env.reset()
    done = False
    while not done:
        cv_img = obs["img"][0]
        lidar = obs["lidar"]
        lidar = lidar[..., 0:2]
        if RENDER:
            print(cv_img.shape)
            cv2.imshow('FrontCamera', cv_img)
            env.render_lidar2D(lidar)

        #a) Convert xyz lidar data to ROS LaserScan message format for FGM
        ranges, theta = cart_to_polar(lidar)
        ranges = polar_to_rosformat(angle_min, -1.0 * angle_min, angle_incr, theta, ranges)

        #b) Use FGM to get action, save it, and step the environment
        action = {"angle":fgm.act(ranges), "speed":0.6}
        save_data(obs, action)
        obs, _, done, _ = env.step(action)

        if cv2.waitKey(3) & 0xFF == ord('q'):
            break
        if done:
            obs = env.reset()  

if __name__ == '__main__':
    main()