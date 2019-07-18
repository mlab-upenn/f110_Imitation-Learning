import os, json, pdb, cv2, math
import numpy as np
from functools import partial
import pandas as pd
from Data_Utils import Data_Utils
dutils = Data_Utils()

def changeName(dest_dict, prefix):
    curr_name = dest_dict.get("row")[0]
    dest_dict["row"][0] = prefix + curr_name
    return dest_dict

def cannyEdge(args, src_dict):
    assert(len(args) == 2), "Incorrect size argument to cannyEdge"
    dest_dict = src_dict
    src_img = src_dict["img"]
    dest_img = cv2.Canny(src_img, args[0], args[1])
    dest_dict["img"] = dest_img
    return changeName(dest_dict, 'cannyedge')

def filterBadData(args, src_dict):
    assert(len(args) == 0), "Incorrect size argument to filterBadData"
    dest_dict = src_dict
    src_img = src_dict.get("img")

    #stupid check to see if picture is largely black
    if np.mean(src_img) < 5:
        dest_dict["flag"] = False
    return changeName(dest_dict, 'fbd')

def opticalFlow(args, src_dict):
    assert(len(args) == 0), "Incorrect size argument to opticalFlow"
    dest_dict = src_dict
    framelist = dutils.get_last_n_frames(3, src_dict["src_datapath"], src_dict["idx"])
    return changeName(dest_dict, 'opticalFlow')

def cropVertical(args, src_dict):
    assert (len(args) == 2),"Incorrect sized argument to cropVertical"
    cropStart = args[0]
    cropEnd = args[1]
    dest_dict = src_dict
    src_img = src_dict.get("img")
    src_row = src_dict.get("row")
    if cropEnd < 0:
        dest_dict["img"] = src_img[cropStart:, :, :]
    elif cropEnd > 0:
        dest_dict["img"] = src_img[cropStart:cropEnd, :, :]
    else:
        raise Exception('bad args cropvertical')
    return changeName(dest_dict, 'cropV')

def rot90(args, src_dict):
    assert (len(args) == 1),"Incorrect sized argument to rot90"
    direction = args[0]
    dest_dict = src_dict
    src_img = src_dict.get("img")
    src_row = src_dict.get("row")
    if direction == 'clockwise':
        dest_dict["img"] = cv2.rotate(src_img, cv2.ROTATE_90_CLOCKWISE)

    elif direction == 'anticlockwise':
        #TODO: DO ANTICLOCKWISE
        pass
    return changeName(dest_dict, 'rot90')

def radOffset(args, src_dict):
    assert (len(args) == 1), "Incorrect sized argument to radOffset"
    offset = args[0]
    dest_dict = src_dict
    src_row = src_dict.get("row")
    dest_dict["row"][1] = src_row[1] + offset
    return changeName(dest_dict, 'radOff')

def rad2deg(args, src_dict):
    assert(len(args) == 0), "Incorrect sized arguments to rad2deg"
    dest_dict = src_dict
    src_row = src_dict.get("row")
    dest_dict["row"][1] = src_row[1] * 180.0/math.pi
    return changeName(dest_dict, 'todeg')

def flipNonZero(args, src_dict):
    assert(len(args) == 0), "Incorrect sized argumnets to flipNonzero"
    dest_dict = src_dict
    src_img = src_dict.get("img")
    src_row = src_dict.get("row")
    if src_row[1] == 0.0:
        dest_dict["flag"] = False
    else:
        dest_dict["flag"] = True
        dest_dict["img"] = cv2.flip(src_img, 1)
        dest_dict["row"][1] = -1.0 * src_row[1]
    return changeName(dest_dict, 'flipped')

def gaussianSamplingAngle(args, src_dict):
    assert(len(args) == 1), "Incorrect number of arguments to gaussianSamplingAngle"
    #input angle must be in degrees
    sigma = args[0]
    dest_dict = src_dict
    src_row = src_dict.get("row")
    #sample from gaussian (w mean = angle)
    dest_dict["row"][1] = np.random.normal(loc=src_row[1], scale=sigma)
    return changeName(dest_dict, 'gaussangle')

def rescaleImg(args, src_dict):
    assert(len(args) == 1), "Incorrect sized arguments to rescaleImg"
    scale = args[0]
    dest_dict = src_dict
    src_img = src_dict.get("img")
    dest_dict["img"] = cv2.resize(src_img, None, fx=scale, fy=scale)
    return changeName(dest_dict, 'rescaled')