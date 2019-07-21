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
        self.zmq_context = SerializingContext()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect(connect_to)

        #convenient lambdas to use later on 
        self.rostojson = lambda x:json_message_converter.convert_ros_message_to_json(x)
        self.jsontoros = lambda topic, x:json_message_converter.convert_json_to_ros_message(topic, x)

        #Syncs to each frame
        self.lidar_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.steer_sub = rospy.Subscriber("/vesc/low_level/ackermann_cmd_mux/output", AckermannDriveStamped, self.steer_callback)
        self.cam_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.cam_callback)

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
            steering_angle = data.drive.steering_angle, 
	    steering_angle_velocity = data.drive.steering_angle_velocity,
            speed = data.drive.speed
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
	if "lidar" in self.latest_obs and "steer" in self.latest_obs:
		lidar_dump = msgpack.dumps(self.latest_obs["lidar"]) 
		steer_dump = msgpack.dumps(self.latest_obs["steer"])
		#print(self.latest_obs["lidar"])
		#print('----------------------------')
		#print(self.latest_obs["steer"])
		#lidar_dump = msgpack.dumps('olda')
		self.zmq_socket.send(lidar_dump, copy=False, flags=zmq.SNDMORE)
		self.zmq_socket.send(steer_dump, copy=False, flags=zmq.SNDMORE)
		self.zmq_socket.send_array(cv_img, copy=False, track=False)
		message = self.zmq_socket.recv()
		print("Recv reply")
		self.latest_obs = {}

def main(args):
	rospy.init_node("f110ZMQTest", anonymous=True)
	sender = f110Sender(connect_to="tcp://195.0.0.7:5555")
	rospy.sleep(0.1)
	rospy.spin()

if __name__ == '__main__':
	main(sys.argv)
