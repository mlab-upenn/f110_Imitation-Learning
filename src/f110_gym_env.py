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
        self.obs_info = {
            'lidar': {'topic':'/scan', 'type':LaserScan, 'callback':self.lidar_callback},

            'img': {'topic':'/usb_cam/image_raw', 'type':Image, 'callback':self.img_callback},

            'steer':{'topic':'/vesc/low_level/ackermann_cmd_mux/output', 'type':AckermannDriveStamped, 'callback':self.steer_callback}
        }

        self.sublist = self.setup_subs(self.obs_info)

        #Subscribe to joy (to access train_button) & pubish to ackermann
        self.joy_sub = rospy.Subscriber('/vesc/joy', Joy, self.joy_callback)        
        self.drive_pub = rospy.Publisher("vesc/high_level/ackermann_cmd_mux/input/nav_0", AckermannDriveStamped, queue_size=4) 

        #one observation is '4' consecutive readings
        self.latest_obs = deque(maxlen=4)         
        self.latest_reading_dict = {}
        self.record = False

        #misc
        self.bridge = CvBridge()

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
        if self.record:
            if data.drive.steering_angle > 0.34:
                data.drive.steering_angle = 0.34
            elif data.drive.steering_angle < -0.34:
                data.drive.steering_angle = -0.34

            steer = dict(
                steering_angle = -1.0 * data.drive.steering_angle, 
                steering_angle_velocity = data.drive.steering_angle_velocity,
                speed = data.drive.speed
            )
            self.latest_reading_dict["steer"] = steer

    def lidar_callback(self, data):
        if self.record:
            lidar = dict(
                angle_min = data.angle_min,
                angle_increment = data.angle_increment,
                ranges = data.ranges
            )
            self.latest_reading_dict["lidar"] = lidar 
    
    def set_status_str(self, prefix=''):
        status_str = ''
        if self.record:
            status_str = 'True'
        else:
            status_str = 'False'
        sys.stdout.write(prefix + "curr_recording: %s" % status_str)
        
        sys.stdout.flush()
    
    def is_reading_complete(self):
        #checks if all the readings are present in latest_reading_dict
        base_check = "lidar" in self.latest_reading_dict and "steer" in self.latest_reading_dict and "img" in self.latest_reading_dict
        return base_check

    def base_preprocessing(self, cv_img):
        cv_img = cv2.resize(cv_img, None, fx=0.5, fy=0.5)
        cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv_img

    def update_latest_obs(self):
        self.latest_obs.append(self.latest_reading_dict)
        self.latest_reading_dict = {}

    def img_callback(self, data):
        self.set_status_str(prefix='\r')

        #img_callback adds latest_reading to the self.lates_obs
        if self.is_reading_complete() and self.record:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e) 
            cv_img  = self.base_preprocessing(cv_img)
            self.latest_reading_dict["img"] = cv_img

            self.update_latest_obs()