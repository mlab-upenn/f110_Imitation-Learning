#!/usr/bin/env python
import rospy
import tf
from time import gmtime, strftime
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, os
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
import argparse


class DataGenerate(object):
	"""
	@brief      Class for extracting images and corresponding steering commands and combining them into a csv 
	"""

	def __init__(self):
		self.params = json.load(open("params.txt"))
		self.bridge = CvBridge()
		self.count = 0
		self.cv_left_img = np.empty((480, 640, 3))
		self.cv_right_img = np.empty((480, 640, 3))
		self.cv_front_img = np.empty((480, 640, 3))
		csv_file = os.path.join(sself.params['abs_path'],'data.csv')
		self.file = open((csv_file), 'w')
		self.file.write('%s, %s, %s \n'%('Image','Angle'))

	def cameraLeftCallback(self,data):
		self.cv_left_img = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

	def cameraRightCallback(self,data):
		self.cv_right_img = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

	def cameraFrontCallback(self,data):
		self.cv_front_img = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")

	def driveCallBack(self,data):
	"""
	@brief      Upon recieving new drive message store left/right/front with the steering angle and offset  
	"""
		if self.params['left_cam']:
			cv2.imwrite(os.path.join(self.params['abs_path'] "image_left%06i.png" % self.count), self.cv_left_img)
			self.file.write('%s, %f\n'%(("image_left%06i.png" % self.count),-(data.drive.steering_angle + self.params['left_offset'])))
		if self.params['right_cam']:
			cv2.imwrite(os.path.join(self.params['abs_path'] "image_right%06i.png" % self.count), self.cv_right_img)
			self.file.write('%s, %f\n'%(("image_right%06i.png" % self.count),(data.drive.steering_angle + self.params['right_offset'])))

		if self.params['front_cam']:
			cv2.imwrite(os.path.join(self.params['abs_path'] "image_front%06i.png" % self.count), self.cv_front_img)
			self.file.write('%s, %f\n'%(("image_front%06i.png" % self.count),(data.drive.steering_angle)))
		self.count += 1


	def listener(self):
		rospy.init_node('drive_logger', anonymous=True)
		rospy.Subscriber('/vesc/low_level/ackermann_cmd_mux/output', AckermannDriveStamped, self.driveCallBack)
		if self.params['left_cam']:
			rospy.Subscriber(self.params['left_camera'], Image, self.cameraLeftCallback)
		if self.params['right_cam']:
			rospy.Subscriber(self.params['right_camera'], Image, self.cameraRightCallback)
		if self.params['front_cam']:
			rospy.Subscriber(self.params['front_camera'], Image, self.cameraFrontCallback)
		rospy.spin()

if __name__=='__main__':

	dG = DataGenerate()
	dG.listener()