"""f110zmq: Send f110 data via ZMQ
@author: Dhruv Karthik (github.com/botforge)
"""

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
from common.imagezmq import SerializingContext, SerializingSocket
import zmq
import numpy as np
import msgpack
import msgpack_numpy as m

class f110Sender(object):
    """
    Opens zmq REQ socket & sends batches of image pairs over to host
    """
    def __init__(self, connect_to='tcp://127.0.0.1:5555'):
        """
        sublist:list of rostopics you want sent over
        """
        #important zmq initialization stuff
        self.zmq_context = zmq.SerializingContext()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect(connect_to)

        #convenient lambdas to use later on 
        self.rostojson = lambda x:json_message_converter.convert_ros_message_to_json(x)
        self.jsontoros = lambda topic, x:json_message_converter.convert_json_to_ros_message(topic, x)

        #Syncs to each frame
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.steer_sub = rospy.Subscriber("/vesc/high_level/ackermann_cmd_mux/output", AckermannDriveStamped, self.steer_callback)
        self.cam_sub = rospy.Subscriber("/usb_cam", Image, self.cam_callback)

        #Hooks sends to the camera, so we need the latest of each observation
        self.latest_obs = {}
        m.patch()
        self.bridge = CvBridge()

    def lidar_callback(self, data):
        """
        Alter LaserScan messages to only include relevant data in the json
        """
        lidar = dict(
            angle_min = data.angle_min,
            angle_increment = data.angle_increment,
            ranges = data.ranges
        )
        self.latest_obs['lidar'] = lidar
        
    def steer_callback(self, data):
        steer = dict(
            steering_angle = data.steering_angle, 
            steering_angle_velocity = data.steering_angle_velocity,
            speed = data.speed
        )
        self.latest_obs['steer'] = steer
    
    def cam_callback(self, data):
        "Send the lidar dumps out here"
        try:
            cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv_img = cv2.resize(cv_img, None, fx=0.4, fy=0.4)
        #make msgpack dumps of everything
        lidar_dump = msgpack.dumps(self.latest_obs["lidar"]) 
        steer_dump = msgpack.dumps(self.latest_obs["steer"])
        self.zmq_socket.send(lidar_dump, copy=False | zmq.SNDMORE)
        self.zmq_socket.send(steer_dump, copy=False | zmq.SNDMORE)
        self.zmq_socket.send_array(cv_img, copy=False, track=False)

class f110Server(object):
    """
    Opens a zmq REQ socket & recieves ROS data 
    """
    def __init__(self, open_port='tcp://*:5555'):
        self.zmq_context = SerializingContext()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind(open_port)
    
    def recv_data(self, copy=False):
        lidar = msgpack.unpack(self.zmq_socket.recv())
        steer = msgpack.unpack(self.zmq_socket.recv())
        md, cv_img = self.zmq_socket.recv_array(copy=False)
        print(lidar, steer)
        cv2.imshow('Bastards', cv_img)