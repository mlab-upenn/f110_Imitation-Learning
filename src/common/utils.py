from __future__ import print_function
import numpy as np
import cv2
import math

def transform_obsarray(obs_array, func):
    ret_array = obs_array.copy()
    for obs_dict in obs_array:
        new_dict = func(obs_dict)
        if new_dict.get("flag", True):
            ret_array.append(new_dict)
    return ret_array

def polar_to_rosformat(angle_min, angle_max, angle_increment, theta, ranges):
    """
    Convert ranges array into ROS ranges format
    """
    sorted_idxs = np.argsort(theta)
    ranges = ranges[sorted_idxs]
    theta = theta[sorted_idxs]

    out_ranges = []
    out_len = (angle_max - angle_min) // angle_increment
    tol = .004363323
    out_ranges = []
    last_idx = -1
    num_nans = 0

    #Fast WAY
    for i in range(int(out_len)):
        curr_theta = angle_min + i * angle_increment
        min_idx = last_idx + 1
        min_diff = abs(theta[min(len(theta)-1, min_idx)] - curr_theta)
        # print(min_idx, min_diff, min_diff <= tol)
        if(min_idx < len(theta) and min_diff <= tol):
            out_ranges.append(ranges[min_idx])
            last_idx += 1
        else:
            out_ranges.append(np.nan)
            num_nans +=1
    # print(f"outlen:{out_len}, fraction of nans:{num_nans/out_len}, range mean:{np.nanmean(np.array(out_ranges))}, range max: {np.nanmax(np.array(out_ranges))}")
    return out_ranges

def lidar_estimate_angleincr(ranges, theta):
    """
    Estimates the angle_incr given ranges & theta
    """
    sorted_idxs = np.argsort(theta)
    ranges = ranges[sorted_idxs]
    theta = theta[sorted_idxs]

    #To estimate angle_increment find difference between all angles & mean
    theta_plusone = theta
    theta = np.roll(theta, 1)
    diff = theta_plusone - theta
    diff = diff[1:]
    return np.mean(diff)

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
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x,y

def lidar_polar_to_cart(ranges, angle_min, angle_increment):
    """
    Convert a lidar_dict to cartesian & return x_ranges & y_ranges
    """
    x_ranges = []
    y_ranges = []
    for i, r in enumerate(ranges):
        if r == np.nan:
            x_ranges.append(1000000)
            y_ranges.append(1000000)
        else:
            theta = angle_min + i * angle_increment
            x, y = polar_to_cart(theta + math.pi/2, r*100.0)
            x_ranges.append(x)
            y_ranges.append(y)
    return x_ranges, y_ranges

def vis_roslidar(ranges, angle_min, angle_increment):
    """
    lidar_dict has the following format:
    {
        'ranges': [float array],
        'angle_min':float,
        'angle_increment':float
    }
    steer_dict has the following format:
    {
        'steering_angle_velocity':float,
        'speed':float,
        'angle':float
    }
    return lidar frame
    """
    #convert lidar data to x,y coordinates
    x_ranges, y_ranges = lidar_polar_to_cart(ranges, angle_min, angle_increment)
    lidar_frame = np.zeros((500, 500, 3)) * 75
    cx = 250
    cy = 450
    rangecheck = lambda x, y: abs(x) < 1000. and abs(y) < 1000.
    for x, y in zip(x_ranges, y_ranges):
        if (rangecheck(x, y)):
            scaled_x = int(cx + x)
            scaled_y = int(cy - y)
            cv2.circle(lidar_frame, (scaled_x, scaled_y), 1, (255, 255, 255), -1)
    cv2.imshow("Reformated", lidar_frame)
    cv2.waitKey(1)