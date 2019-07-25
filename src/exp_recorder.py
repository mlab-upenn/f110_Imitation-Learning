#!/usr/bin/env python
from __future__ import print_function

import roslib, rospy, cv2, sys, math, time, json, os
from sensor_msgs.msg import Image, LaserScan, Joy
from cv_bridge import CvBridge, CvBridgeError
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from rospy_message_converter import json_message_converter
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
                                  'steer_topic':'/vesc/low_level/ackermann_cmd_mux/output'}):
        """
        only_record: 'autonomous', 'joystick', or 'both'
        """
        #important zmq initialization stuff
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.DEALER)
        self.zmq_socket.connect(connect_to)
        myid = b'0'
        self.zmq_socket.identity = myid.encode('ascii')

        #Sub to each frame
        self.lidar_sub = rospy.Subscriber(record_topics['lidar_topic'], LaserScan, self.lidar_callback)
        self.steer_sub = rospy.Subscriber(record_topics['steer_topic'], AckermannDriveStamped, self.steer_callback)
        self.cam_sub = rospy.Subscriber(record_topics['camera_topic'], Image, self.cam_callback)
	self.joy_sub = rospy.Subscriber('/vesc/joy', Joy, self.joy_callback)

        #other stuff
        self.latest_obs = {}
        self.curr_batch = []
        self.framecount = 0
	self.batchcount = 0
        m.patch()
        self.bridge = CvBridge()
	self.only_record = only_record
	self.curr_recording = '' #either autonomous, joystick or both

        #Multithreading stuff
	threading.Thread.__init__(self) 

    def joy_callback(self, data): #just sets the bit auton_button = data.buttons[5]
	auton_button = data.buttons[5]
	joy_button = data.buttons[4]
	if joy_button:
	    self.curr_recording = 'joystick'
	elif auton_button:
	    self.curr_recording = 'autonomous'
	else:
	    self.curr_recording = '          '

    def lidar_callback(self, data):
        lidar = dict(
            angle_min = data.angle_min,
            angle_increment = data.angle_increment,
            ranges = data.ranges
        )
        self.latest_obs['lidar'] = lidar
        
    def steer_callback(self, data):
	if self.curr_recording == self.only_record or self.only_record == 'both':
		steer = dict(
		    steering_angle = -1.0 * data.drive.steering_angle, 
		    steering_angle_velocity = data.drive.steering_angle_velocity,
		    speed = data.drive.speed
		)
		self.latest_obs['steer'] = steer
    
    def save_model(self, model_dump):
	modelpath = '/home/nvidia/datasets/avfone/models/'
	if not os.path.exists(modelpath):
	    os.makedirs(modelpath)
	f = open(os.path.join(modelpath, 'model'), 'w')
	f.write(model_dump)
	f.close()
	
    def cam_callback(self, data):
	sys.stdout.write("\rcurr_recording: %s" %self.curr_recording)
	sys.stdout.flush()
        if "lidar" in self.latest_obs and "steer" in self.latest_obs:
            if self.framecount % 10 == 0:
                #Add every 10 full frames to batch
                lidar_dump = msgpack.dumps(self.latest_obs["lidar"])
                steer_dump = msgpack.dumps(self.latest_obs["steer"])
                try:
                    cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
                except CvBridgeError as e:
                    print(e)
		cv_img = cv2.resize(cv_img, None, fx=0.5, fy=0.5)
		cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv_md = dict(
                    dtype=str(cv_img.dtype),
                    shape=cv_img.shape,
                )
                cv_md_dump = msgpack.dumps(cv_md)
                self.curr_batch += [lidar_dump, steer_dump, cv_md_dump, cv_img]
                self.latest_obs = {}
                if (len(self.curr_batch) / 4.0 % 8.0) == 0:
                    sys.stdout.write(" ||| Sending out batch %s" % self.batchcount)
		    batchcount_dump = msgpack.dumps(self.batchcount)
	            self.curr_batch = [batchcount_dump] + self.curr_batch
		    sys.stdout.flush()
                    self.zmq_socket.send_multipart(self.curr_batch, copy=False)
		    self.curr_batch = []
		    self.batchcount+=1
	    self.framecount+=1

    def run(self):
        poll = zmq.Poller()
        poll.register(self.zmq_socket, zmq.POLLIN)
        while True:
            sockets = dict(poll.poll(10000))
            if self.zmq_socket in sockets:
                msg = self.zmq_socket.recv_multipart()
		batchnum = msgpack.loads(msg[0], encoding="utf-8")
		print("\n RECVD NN FOR BATCH %s" % batchnum)
		self.save_model(msg[1])

def main(args):
    rospy.init_node("ExperienceRecorder", anonymous=True)
    sender = ExperienceRecorder(connect_to="tcp://195.0.0.7:5555", only_record='joystick')
    sender.daemon = True
    sender.start()
    rospy.sleep(0.2)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
