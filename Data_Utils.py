import os, json, pdb, cv2
from functools import partial
import pandas as pd

def cropVertical(args, img):
    """
    Assumes img is np.array of shape H x W x C
    """
    assert (len(args) == 2) "Incorrect sized argument to cropVertical"
    cropStart = args[0]
    cropEnd = args[1]

    if cropEnd < 0:
        return img[cropStart:, :, :]
    elif cropEnd > 0:
        return img[cropStart:cropEnd, :, :]


class Data_Utils(object):
    """
    Useful functions for moving around & processing steer data
    """
    def __init__(self):
        self.params_dict = json.load(open("steps.json"))["params"]    

    def
    
    def get_partials_list(self, tf_list):
        """
        tf_list: a list of transforms in "F":, "args"[], format
        return a list of partial functions
        """
        parsed_tf_list = []
        for tf in tf_list:
            func = self.tf_partial(tf["F"], tf["args"])
            parsed_tf_list.append(func)

    def preprocess_folder(self, sourcepath, destpath, tf_list):
        parsed_tf_list = self.get_partials_list(tf_list)