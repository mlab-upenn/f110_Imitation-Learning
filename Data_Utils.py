import os, json, pdb, cv2, math
import numpy as np
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

def rot90(args, img, metrics_row):
    assert (len(args) == 1),"Incorrect sized argument to rot90"
    direction = args[0]
    if direction == 'clockwise':
        new_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    elif direction == 'anticlockwise':
        #TODO: DO ANTICLOCKWISE
        pass

    return new_img, metrics_row

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

def flipNonZero(args, img, metrics_row):
    assert(len(args) == 0), "Incorrect sized argumnets to flipNonzero"
    new_metrics_row = metrics_row.copy()
    angle = metrics_row[1]
    if angle == 0.0:
        new_metrics_row[1] = -1.0 * angle
        new_img = cv2.flip(img, 1)
        return new_img, new_metrics_row
    else:
        return img, metrics_row

class Data_Utils(object):
    """
    UPDATE THIS WHEN I GET IT LMAO
    """
    def __init__(self):
        pass
    
    def get_dest_datapath(self, dest_datadir, folder, op):
        """
        Returns absolute path for destination data, takes care of naming
        folder: name of folder being moved
        op: if 'aug' don't screw with naming
        """
        if op == 'aug':
            return os.path.join(dest_datadir, folder)
        else:
            new_folder = folder + str(len(os.listdir(dest_datadir)))
            return os.path.join(dest_datadir, new_folder)

    def get_df(self, datapath):
        csvpath = os.path.join(datapath, 'data.csv')
        df = pd.read_csv(csvpath)
        return df

    def df_data_fromidx(self, datapath, df, idx):
        row = df.iloc[idx]
        img_name = row[0]
        img_path = os.path.join(datapath, img_name)
        img = cv2.imread(img_path)
        return img, row

    def MOVE(self, src_datadir, folder, dest_datadir, flist=[], maxlen=-1, op='mv'):
        """
        MOVE takes src_datadir/folder and moves to dest_datadir & applies flist functions to it
        src_datadir: abs path to dir containing src data
        dest_datadir: abs path to dest dir
        folder: name of folder in src_datadir to MOVE
        flist: list of json-formatted functions (see steps.json)
        maxlen: the number of entries to move
        op: if 'aug', augment current dataset instead of creating a whole new one & moving it elsewhere (IF SO, SRC_DATADIR MUST = DEST_DATADIR)
        """
        assert(op=='aug' and src_datadir != dest_datadir), f"MOVE Error: If op={op}, src_datadir = dest_datadir"

        if not os.path.exists(dest_datadir):
            os.makedirs(dest_datadir)

        src_datapath = os.path.join(src_datadir, folder)
        dest_datapath = self.get_dest_datapath(dest_datadir, folder, op)

        #make new dataframes
        src_df = self.get_df(src_datapath)
        dest_df = pd.DataFrame(columns=src_df.columns.values)

        #iterate through dataframe
        maxlen = max(len(src_df), maxlen)

        for i in range(maxlen):
            src_img, src_row = self.df_data_fromidx(src_datapath, src_df, i)
            

    def tf_partial(self, fname, args):
        if fname == 'cropVertical':
            p = partial(cropVertical, args)
        elif fname == 'rad2deg':
            p = partial(rad2deg, args)
        elif fname == 'radOffset':
            p = partial(radOffset, args)
        elif fname == 'rot90':
            p = partial(rot90, args)
        elif fname == 'flipNonZero':
            p = partial(flipNonZero, args)
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
            new_df = new_df.append(new_row)
        
        newcsvpath = os.path.join(destpath, "data.csv")
        new_df.to_csv(newcsvpath, index_label=False, index=False)


    def augment_folder(self, sourcepath, tf_list):
        "TODO: Account for when destpath already exists"
        tfs = self.get_partials_list(tf_list)

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
            new_img_path = os.path.join(sourcepath, 'aug_' + img_name)
            cv2.imwrite(new_img_path, new_img)
            new_df = new_df.append(new_row)
        
        new_df.append(old_df)
        newcsvpath = os.path.join(sourcepath, "data.csv")
        new_df.to_csv(newcsvpath, index_label=False, index=False)

    
    def create_preview_folder(self, sourcepath, destpath):
        #make directories to destpath
        os.makedirs(destpath)

        #set up new csv file/dataframe
        csvpath = os.path.join(sourcepath, "data.csv")
        old_df = pd.read_csv(csvpath)
        col_names = old_df.columns.values
        new_df = pd.DataFrame(columns=col_names)

        for i in range(50):
            #get old_image and old_row
            old_row = old_df.iloc[i]
            img_name = old_row[0]
            img_path = os.path.join(sourcepath, img_name)
            old_img = cv2.imread(img_path)

            new_img_path = os.path.join(destpath, img_name)
            cv2.imwrite(new_img_path, old_img)
            new_df = new_df.append(old_row)

        #add max steering angle and minimum steering angle
        angle_column = old_df.iloc[:, 1].values
        max_idx = np.argmax(angle_column)
        min_idx = np.argmin(angle_column)
        max_row = old_df.iloc[max_idx]
        min_row = old_df.iloc[min_idx]
        max_img_name = max_row[0]
        max_img_path = os.path.join(sourcepath, max_img_name)
        max_img = cv2.imread(max_img_path)
        cv2.imwrite(os.path.join(destpath, max_img_name), max_img)
        new_df = new_df.append(max_row)
        min_img_name = min_row[0]
        min_img_path = os.path.join(sourcepath, min_img_name)
        min_img = cv2.imread(min_img_path) 
        cv2.imwrite(os.path.join(destpath, min_img_name), min_img)
        new_df = new_df.append(min_row)
        newcsvpath = os.path.join(destpath, "data.csv")
        new_df.to_csv(newcsvpath, index_label=False, index=False)