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
import json
import numpy as np
from sensor_msgs.msg import CompressedImage

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
        self.left_csv = ""
        self.right_csv = ""
        self.front_csv = ""
        self.setup()
        print("Waiting for topic to start recording")     

    def open_csv(self, csv_file, folder_name):
        try:
            os.makedirs(folder_name)
        except OSError, e:
                pass  
        if os.path.exists(csv_file):
            return open((csv_file), 'a')
        else: 
            open_file = open((csv_file), 'w')
            open_file.write('%s, %s, %s \n'%('Image','Angle','Speed'))
            return open_file


    def setup(self):
        if self.params['left_cam']:
            self.left_folder = os.path.join(self.params['abs_path'], 'left_folder')
            csv_file = os.path.join(self.params['abs_path'],'left_folder','data_left.csv')
            self.left_csv = self.open_csv(csv_file,self.left_folder)

        if self.params['right_cam']: 
            self.right_folder = os.path.join(self.params['abs_path'], 'right_folder')
            csv_file = os.path.join(self.params['abs_path'],'right_folder','data_right.csv')
            self.right_csv = self.open_csv(csv_file,self.right_folder)              

        if self.params['front_cam']: 
            self.front_folder = os.path.join(self.params['abs_path'], 'front_folder')
            csv_file = os.path.join(self.params['abs_path'],'front_folder','data_front.csv')
            self.front_csv = self.open_csv(csv_file,  self.front_folder)                 
        

    def cameraLeftCallback(self,data):
        self.cv_left_img = self.bridge.compressed_imgmsg_to_cv2(data)

    def cameraRightCallback(self,data):
        self.cv_right_img = self.bridge.compressed_imgmsg_to_cv2(data)

    def cameraFrontCallback(self,data):
        self.cv_front_img = self.bridge.compressed_imgmsg_to_cv2(data)

    def driveCallBack(self,data):
        """
        @brief      Upon recieving new drive message store left/right/front with the steering angle and offset  
        """

        #Ackermann messages give left as +ve and right as -ve. Storing steering angle as negative of that to maintain convention
        # Convention left -ve and right +ve
        steering_angle = - data.drive.steering_angle
        if self.params['left_cam']:
            cv2.imwrite(os.path.join(self.left_folder, "image_left%06i.jpg" % self.count), self.cv_left_img)
            self.left_csv.write('%s, %f, %f\n'%(("image_left%06i.jpg" % self.count),(steering_angle - self.params['left_offset']),data.drive.speed))
        if self.params['right_cam']:
            cv2.imwrite(os.path.join(self.right_folder, "image_right%06i.jpg" % self.count), self.cv_right_img)
            self.right_csv.write('%s, %f, %f\n'%(("image_right%06i.jpg" % self.count),(steering_angle + self.params['right_offset']),data.drive.speed))
        if self.params['front_cam']:
            cv2.imwrite(os.path.join(self.front_folder, "image_front%06i.jpg" % self.count), self.cv_front_img)
            self.front_csv.write('%s, %f, %f\n'%(("image_front%06i.jpg" % self.count),(steering_angle),data.drive.speed))
        self.count += 1


    def listener(self):
        rospy.init_node('drive_logger', anonymous=True)
        rospy.Subscriber('/vesc/low_level/ackermann_cmd_mux/output', AckermannDriveStamped, self.driveCallBack)
        if self.params['left_cam']:
            rospy.Subscriber(self.params['left_camera'], CompressedImage, self.cameraLeftCallback)
        if self.params['right_cam']:
            rospy.Subscriber(self.params['right_camera'], CompressedImage, self.cameraRightCallback)
        if self.params['front_cam']:
            rospy.Subscriber(self.params['front_camera'], CompressedImage, self.cameraFrontCallback)
        rospy.spin()

if __name__=='__main__':

	dG = DataGenerate()
	dG.listener()
