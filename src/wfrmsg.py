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

rospy.init_node("Gym_Recorder", anonymous=True)
bridge = CvBridge()
while True:
    img_data = rospy.wait_for_message(
            "/usb_cam/image_raw",
            Image)
    steer_data = rospy.wait_for_message(
            "/vesc/low_level/ackermann_cmd_mux/output",
            AckermannDriveStamped)
    lidar_data = rospy.wait_for_message(
            "/vesc/low_level/ackermann_cmd_mux/output",
            AckermannDriveStamped)

    print(steer_data)
    cv_img = bridge.imgmsg_to_cv2(img_data, "bgr8")
    cv2.imshow('latestimg', cv_img)
    cv2.waitKey(2)
