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
import zmq, msgpack, threading
import msgpack_numpy as m
import numpy as np

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class ExperienceRecorder(threading.Thread):
    """
    Opens zmq DEALIER socket & sends 'experiences' over
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
        self.zmq_socket = self.zmq_context.socket(zmq.DEALER)
        self.zmq_socket.connect(connect_to)
        id = '0'
        self.zmq_socket.identity = identity.encode('ascii')

        #Sub to each frame
        self.lidar_sub = rospy.Subscriber(record_topics['lidar_topic'], LaserScan, self.lidar_callback)
        self.steer_sub = rospy.Subscriber(record_topics['camera_topic'], AckermannDriveStamped, self.steer_callback)
        self.cam_sub = rospy.Subscriber(record_topics['steer_topic'], Image, self.cam_callback)

        #other stuff
        self.latest_obs = {}
        self.curr_batch = []
        self.framecount = 0
        m.patch()
        self.bridge = CvBridge()

        #Multithreading stuff
        threading.Thread.__init__ (self)
    
    def lidar_callback(self, data):
        lidar = dict(
            angle_min = data.angle_min,
            angle_increment = data.angle_increment,
            ranges = data.ranges
        )
        self.latest_obs['lidar'] = lidar
        
    def steer_callback(self, data):
        steer = dict(
            steering_angle = data.drive.steering_angle, 
	    steering_angle_velocity = data.drive.steering_angle_velocity,
            speed = data.drive.speed
        )
        self.latest_obs['steer'] = steer
    
    def cam_callback(self, data):
        if "lidar" in self.latest_obs and "steer" in self.latest_obs:
            if self.framecount % 10 == 0:
                #Add every 10 full frames to batch
                lidar_dump = msgpack.dumps(self.latest_obs["lidar"])
                steer_dump = msgpack.dumps(self.latest_obs["steer"])
                try:
                    cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
                except CvBridgeError as e:
                    print(e)
                cv_md = dict(
                    dtype=str(cv_img.dtype),
                    shape=cv_img.shape,
                )
                cv_md_dump = msgpack.dumps(cv_md)
                # self.curr_batch += [lidar_dump, steer_dump, cv_md_dump, cv_img]
                self.curr_batch += [b'lidary', b'dumpy', b'steery', b'cvmd', b'cvimg']
                self.latest_obs = {}
                if (len(self.curr_batch) / 4.0 % 20.0) == 0:
                    print(f"Sending out batch {len(self.curr_batch)/4.0}")
                    self.zmq_socket.send_multipart(self.curr_batch, copy=False)
    def run(self):
        poll = zmq.Poller()
        poll.register(self.zmq_socket, zmq.POLLIN)
        while True:
            sockets = dict(poll.poll(1000))
            if self.zmq_socket in sockets:
                msg = self.zmq_socket.recv_multipart()
                print(msg)

def main(args):
    rospy.init_node("ExperienceRecorder", anonymous=True)
    sender = ExperienceRecorder(connect_to="tcp://195.0.0.7:5555")
    sender.start()
    rospy.sleep(0.1)
    rospy.spin()