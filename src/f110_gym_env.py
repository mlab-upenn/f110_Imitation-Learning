#!/usr/bin/env python
from __future__ import print_function
import os, sys, cv2, math, time
import numpy as np
from steps import session
from collections import deque

#ROS Dependencies
import roslib, rospy
import numpy as np
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from sensor_msgs.msg import Image, LaserScan, Joy
from cv_bridge import CvBridge, CvBridgeError

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class f110_gym_env(object):
    """
    Implements a Gym Environment & neccessary funcs for the F110 Autonomous RC Car(similar structure to gym.wrappers)
    """
    def __init__(self):
        
        #At least need LIDAR, IMG & STEER for everything here to work 
        obs_info = {
            'lidar': {'topic':'/scan', 'type':LaserScan, 'callback':self.lidar_callback},

            'img': {'topic':'/usb_cam/image_raw', 'type':Image, 'callback':self.img_callback},

            'steer':{'topic':'/vesc/low_level/ackermann_cmd_mux/output', 'type':AckermannDriveStamped, 'callback':self.steer_callback}
        }

        self.sublist = self.setup_subs(obs_info)
        
        #one observation is '4' consecutive readings
        self.latest_obs = deque(maxlen=4)         
        self.record = False

    def setup_subs(self, obs_info):
        """
        Initializes subscribers w/ obs_info & returns a list of subscribers
        """
        makesub = lambda subdict : rospy.Subscriber(subdict['topic'], subdict['type'], subdict['callback']) 

        sublist = []
        for topic in obs_info:
            sublist.append(makesub(obs_info[topic]))
        return sublist
    
    def steer_callback(self, data):