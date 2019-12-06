from __future__ import print_function
import numpy as np
import math

def transform_obsarray(obs_array, func):
    ret_array = obs_array.copy()
    for obs_dict in obs_array:
        new_dict = func(obs_dict)
        if new_dict.get("flag", True):
            ret_array.append(new_dict)
    return ret_array

def cart_to_polar(xy):
    """
    Returns range, theta coordinates
    """
    ranges = np.linalg.norm(xy, axis=1)
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    return ranges, theta

def polar_to_cart(theta, r):
    """
    Assumes theta in radians & returns x,y
    """
    x = r*math.cos(theta)
    y = r*math.sin(theta)
    return x,y

def lidar_polar_to_cart(ranges, angle_min, angle_increment):
    """
    Convert a lidar_dict to cartesian & return x_ranges & y_ranges
    """
    x_ranges = []
    y_ranges = []
    for i, r in enumerate(ranges):
        theta = angle_min + i * angle_increment
        x, y = polar_to_cart(theta + math.pi/2, r*100.0)
        x_ranges.append(x)
        y_ranges.append(y)
    return x_ranges, y_ranges