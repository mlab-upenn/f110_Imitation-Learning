#!/usr/bin/env python
from __future__ import print_function
import os, sys, cv2, math, time, torch
import numpy as np
from nnet.models import *
from steps import session
from nnet.Data_Utils import Data_Utils

#ROS Dependencies
import roslib, rospy
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from sensor_msgs.msg import Image, LaserScan, Joy
from cv_bridge import CvBridge, CvBridgeError
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class NN_Steer(object):
    """
    Steer the F110 Car using commands generated by a pytorch Neural Network
    """
    def __init__(self, model_name='model'):
        
        self.load_net(model_name=model_name) #updates self.net


        #Setup Subscribers & Publishers
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.steer_sub = rospy.Subscriber('/usb_cam/image_raw', AckermannDriveStamped, self.steer_callback)
        self.cam_sub = rospy.Subscriber('/vesc/low_level/ackermann_cmd_mux/output', Image, self.cam_callback)
        self.steer_pub = rospy.Publishers("vesc/high_level/ackermann_cmd_mux/input/nav_0", AckermannDriveStamped, queue_size=1)

        #At what interval should we sample to get a steering angle
        self.sample_interval = 4
        self.framecount = 0
        self.bridge = CvBridge()
        self.dutils = Data_Utils()
        self.funclist = session["online"]["funclist"]

    def lidar_callback(self, data):
        pass
    
    def steer_callback(self, data):
        pass
    
    def cam_callback(self, data):
        if self.framecount % self.sample_interval == 0:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

            ts_img = self.NN_preprocess(cv_img)

            #use net to get predicted angle
            ts_angle_pred = self.net(ts_img)
            
            #Send to car
            angle_pred = ts_angle_pred.item()
            vel = 1
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = rospy.Time.now()
            drive_msg.header.frame_id = "odom" 
            drive_msg.drive.steering_angle = -1.0 * angle_pred.item()
            drive_msg.drive.speed = vel
            self.steer_pub.publish(drive_msg)

    def NN_preprocess(self, cv_img):
            #basic serverside preprocess
            cv_img = cv2.resize(cv_img, None, fx=0.5, fy=0.5)
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            #external funclist preprocessing
            src_dict = {"img":cv_img}
            new_data_dict = self.dutils.apply_flist(src_dict, self.funclist, w_rosdict=True)
            cv_img = new_data_dict["img"]

            #Dataset preprocessing
            ts_img = torch.from_numpy(cv_img).permute(2, 0, 1).float()
            ts_img = ts_img[None]
            return ts_img

    def load_net(self, model_name='model'):
        modelpath = os.path.join(session["online"]["modelpath"], model_name)
        net = NVIDIA_ConvNet()
        if os.path.exists(modelpath):
            net.load_state_dict(torch.load(modelpath))
        net.to(device)
        net.eval()
        self.net = net
        print("LOADED NETWORK")
        print("DEVICE:{device}".format(device=device))
        print("NETWORK:")
        print(net)