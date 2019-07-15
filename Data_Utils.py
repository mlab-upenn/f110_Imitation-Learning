import os, json, pdb, cv2
from functools import partial
import pandas as pd

def cropVertical(args, img, metrics_row):
    assert (len(args) == 2),"Incorrect sized argument to cropVertical"
    cropStart = args[0]
    cropEnd = args[1]
    if cropEnd < 0:
        return img[cropStart:, :, :], metrics_row
    elif cropEnd > 0:
        return img[cropStart:cropEnd, :, :], metrics_row

def radOffset(args, img, metrics_row):
    assert (len(args) == 1), "Incorrect sized argument to radOffset"
    offset = args[0]
    new_metrics_row = metrics_row.copy()
    angle_rad = metrics_row[1]
    new_metrics_row[1] = angle_rad + offset
    return img, new_metrics_row

def rad2deg(args, img, metrics_row):
    assert(len(args) == 0), "Incorrect sized arguments to rad2deg"
    new_metrics_row = metrics_row.copy()
    angle_rad = metrics_row[1]
    new_metrics_row[1] = angle_rad * 180.0/math.pi

class Data_Utils(object):
    """
    Useful functions for moving around & processing steer data
    """
    def __init__(self):
        self.params_dict = json.load(open("steps.json"))["params"]    

    def tf_partial(self, fname, args):
        if fname == 'cropVertical':
            p = partial(cropVertical, args)
        elif fname == 'rad2deg':
            p = partial(rad2deg, args)
        elif fname == 'radOffset':
            p = partial(radOffset)
        else:
            raise Exception('{fname} is not in the list of functions')
        return p

    def get_partials_list(self, tf_list):
        """
        tf_list: a list of transforms in "F":, "args"[], format
        return a list of partial functions
        """
        parsed_tf_list = []
        for tf in tf_list:
            func = self.tf_partial(tf["F"], tf["args"])
            parsed_tf_list.append(func)
        return parsed_tf_list

    def preprocess_folder(self, sourcepath, destpath, tf_list):
        parsed_tf_list = self.get_partials_list(tf_list)

        csvpath = os.path.join(sourcepath, "data.csv")