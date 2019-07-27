#!/usr/bin/env python
from __future__ import print_function
import os, sys, cv2, math, time
import numpy as np
from collections import deque

#ROS Dependencies
import roslib, rospy
import numpy as np
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from sensor_msgs.msg import Image, LaserScan, Joy
from cv_bridge import CvBridge, CvBridgeError

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class f110Env(object):
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
        self.history= deque(maxlen=500) #for reversing during reset

    ############ GYM METHODS ###################################

    def _get_obs(self):
        """
        Returns latest observation (TODO:DECIDE IF I WANT TO RET JUST ONE OR MORE OBS)
        """
        obs_dict = self.latest_obs[-1]
        return obs_dict
        
    def reset(self):
        """
        Reverse until we're not 'tooclose'
        """
        if self.tooclose():
            self.record = False
            self.reverse()
        else:
            self.record = True
        
        #TODO: consider sleeping a few milliseconds?
        return self._get_obs()

    def get_reward(self):
        """
        TODO:Implement reward functionality
        """
        return 0

    def step(self, action):
        """
        Action should be a steer_dict = {"angle":float, "speed":float}
        """
        #execute action
        drive_msg = self.get_drive_msg(action.get("angle"), action.get("speed"), flip_angle=-1.0)
        self.drive_pub.pubish(drive_msg)

        #get reward & check if done & return
        reward = self.get_reward()
        done = self.tooclose()
        info = ''
        return self._get_obs, reward, done, info

    ############ GYM METHODS ###################################

    ############ ROS HANDLING METHODS ###################################
    def setup_subs(self, obs_info):
        """
        Initializes subscribers w/ obs_info & returns a list of subscribers
        """
        makesub = lambda subdict : rospy.Subscriber(subdict['topic'], subdict['type'], subdict['callback']) 

        sublist = []
        for topic in obs_info:
            sublist.append(makesub(obs_info[topic]))
        return sublist

    def add_to_history(self, data):
        if abs(data.drive.steering_angle - 0.05) != 0.0:
            steer_dict = {"angle":data.drive.steering_angle, "speed":data.drive.speed}
            for i in range(40):
                self.history.append(steer_dict) 
    
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

        self.add_to_history(data) #add steering commands to history

    def lidar_callback(self, data):
        if self.record:
            lidar = dict(
                angle_min = data.angle_min,
                angle_increment = data.angle_increment,
                ranges = data.ranges
            )
            self.latest_reading_dict["lidar"] = lidar 
    
    def joy_callback(self, data):
        train_button = data.buttons[1]
        if train_button:
            self.record = True
        else:
            self.record = False

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

    def get_drive_msg(self, angle, speed, flip_angle=1.0):
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "odom" 
        drive_msg.drive.steering_angle = flip_angle * angle
        drive_msg.drive.speed = speed
        return drive_msg

    def reverse(self):
        """
        Uses self.history to back out
        """
        sign = lambda x: (1, -1)[x < 0]
        default_steer_dict = {"angle":0.0, "speed":1.0}
        try:
            steer_dict = self.history.pop()
        except:
            steer_dict = default_steer_dict

        rev_angle = steer_dict["angle"]
        rev_speed = -1.0
        print("REVERSE {rev_angle}".format(rev_angle = rev_angle))
        drive_msg = self.get_drive_msg(rev_angle, rev_speed)
        self.drive_pub.publish(drive_msg)
    
    def tooclose(self):
        """
        Uses self.latest_obs to determine if we are too_close (currently uses LIDAR)
        """
        tc = True
        if len(self.latest_obs) > 0:

            reading = self.latest_obs[-1]

            #Use LIDAR Reading to check if we're too close
            lidar = reading["lidar"]
            ranges = lidar.get("ranges")
            angle_min = lidar.get("angle_min")
            angle_incr = lidar.get("angle_incr")
            rfrac = lambda st, en : ranges[int(st*len(ranges)):int(en*len(ranges))]
            mindist = lambda r, min_range : np.nanmin(r[r != -np.inf]) <= min_range
            #ensure that boundaries are met in each region
            r1 = rfrac(0, 1./4.)
            r2 = rfrac(1./4., 3./4.)
            r3 = rfrac(3./4., 1.) 
            if mindist(r1, 0.4) or mindist(r2, 0.6) or mindist(r3, 0.4):
                tc = True
            else:
                tc = False
        else:
            tc = False

        return tc
    ############ ROS HANDLING METHODS ###################################

    class f110Wrapper(f110Env):
        """
        Wraps the environment to allow a modular transformation of 
        """