import os, json, pdb, cv2, math
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
    return img, new_metrics_row

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
            p = partial(radOffset, args)
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
    
    def apply_tfs(self, old_img, old_row, tfs):
        new_img = old_img
        new_row = old_row
        for tf in tfs:
            new_img, new_row = tf(new_img, new_row)
        return new_img, new_row

    def preprocess_folder(self, sourcepath, destpath, tf_list):
        "TODO: Account for when destpath already exists"
        tfs = self.get_partials_list(tf_list)

        #make directories to destpath
        os.makedirs(destpath)

        #set up new csv file/dataframe
        csvpath = os.path.join(sourcepath, "data.csv")
        old_df = pd.read_csv(csvpath)
        col_names = old_df.columns.values
        new_df = pd.DataFrame(columns=col_names)

        for i in range(len(old_df)):
            #get old_image and old_row
            old_row = old_df.iloc[i]
            img_name = old_row[0]
            img_path = os.path.join(sourcepath, img_name)
            old_img = cv2.imread(img_path)

            #apply operations to get new_img and new_row
            new_img, new_row = self.apply_tfs(old_img, old_row, tfs)

            #write image and append to new_df
            new_img_path = os.path.join(destpath, img_name)
            cv2.imwrite(new_img_path, new_img)
            new_df.append(new_row)
        
        newcsvpath = os.path.join(sourcepath, "data.csv")
        new_df.to_csv(newcsvpath)