#!/usr/bin/env python
from __future__ import print_function

import roslib, rospy, cv2, sys, math, time, json
# from std_msgs.msg import String, Header, ColorRGBA
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
# from visualization_msgs.msg import Marker, MarkerArray
# from geometry_msgs.msg import Pose, Point, Quaternion, Vector3, PointStamped
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from rospy_message_converter import json_message_converter
# from common.imagezmq import SerializingContext, SerializingSocket
import zmq, msgpack
import numpy as np

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'
class ExperienceRecorder(object):
    """
    Opens zmq REQ socket & sends 'experiences' over
    ATM: Records Lidar, Camera & Steer (cmd_mux)
    """
    def __init__(self, connect_to='tcp://195.0.0.7:5555', only_record='both'
                ,record_topics = {'lidar_topic':'/scan', 
                                  'camera_topic':'/usb_cam/image_raw',
                                  'steer_topic':'/vesc/low_level/ackermann_cmd_mux'}):
        """
        only_record: 'autonomous', 'joystick', or 'both'
        """
        #important zmq initialization stuff
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect(connect_to)