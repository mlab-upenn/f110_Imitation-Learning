import os, json, pdb, cv2, math
import numpy as np
from functools import partial
import pandas as pd

def filterBadData(args, src_dict):
    assert(len(args) == 0), "Incorrect size argument to filterBadData"
    dest_dict = src_dict
    src_img = src_dict.get("img")

    #stupid check to see if picture is largely black
    if np.mean(src_img) < 5:
        dest_dict["flag"] = False
    return dest_dict

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
    dest_dict["img_name"] = "vcrop_" + (src_row[0])
    return dest_dict

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
    dest_dict["img_name"] = "rot90" + (src_row[0])
    return dest_dict

def radOffset(args, src_dict):
    assert (len(args) == 1), "Incorrect sized argument to radOffset"
    offset = args[0]
    dest_dict = src_dict
    src_row = src_dict.get("row")
    dest_dict["row"][1] = src_row[1] + offset
    return dest_dict

def rad2deg(args, src_dict):
    assert(len(args) == 0), "Incorrect sized arguments to rad2deg"
    dest_dict = src_dict
    src_row = src_dict.get("row")
    dest_dict["row"][1] = src_row[1] * 180.0/math.pi
    return dest_dict

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
    dest_dict["img_name"] = "rot90" + (src_row[0])
    return dest_dict
