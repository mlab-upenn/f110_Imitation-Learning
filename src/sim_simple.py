#!/usr/bin/env python
from __future__ import print_function
import airsim
import cv2, sys, os
import numpy as np

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'


def main():
    car_controls = airsim.CarControls()



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        rospy.signal_shutdown('Done')
        pass
