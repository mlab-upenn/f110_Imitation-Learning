from __future__ import print_function
import os, json, pdb, cv2, math, random, pickle, msgpack
try:
    import msgpack_numpy as m
except ImportError:
   print("cannot use full functionality of DATA_UTILS without msgpack_numpy")

import numpy as np
from functools import partial
import pandas as pd
import json
try:
	from detection.yolov3 import YoloDetection
except ImportError:
	print("Could not import  YOLO")

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'
class Data_Utils(object):
    """
    Move around & augment data 
    """
    def __init__(self):
        m.patch()
        pass
    
    def get_dest_datapath(self, dest_datadir, folder, op):
        """
        Returns absolute path for destination data, takes care of naming
        folder: name of folder being moved
        op: if 'aug' don't screw with naming
        """
        if op == 'aug':
            return folder, os.path.join(dest_datadir, folder)
        elif op == 'mv':
            new_folder = str(len(os.listdir(dest_datadir))) + '_' +  folder 
        elif 'prefix' in op:
            strlist = op.split('|')
            assert(len(strlist) == 2), "Incorrect formatting for op = {op}"
            new_folder = strlist[1] + folder
        elif 'choosename' in op:
            #you need to take care that there are no name conflicts here
            if os.path.exists(os.path.join(dest_datadir, op)):
                print("WARNING: FOLDER {op} already exists in PATH:{dest_datadir}".format(op=op, dest_datadir=dest_datadir))

            strlist = op.split('|')
            assert(len(strlist) == 2), "Incorrect formatting for op = {op}".format(op=op)
            new_folder = strlist[1]
        return new_folder, os.path.join(dest_datadir, new_folder)

    def get_df(self, datapath):
        csvpath = os.path.join(datapath, 'data.csv')
        df = pd.read_csv(csvpath)
        return df
    
    def get_interesting_idxs(self, dpath, num_idxs):
        df = self.get_df(dpath)
        interesting_idxs = []
        #currently only considers the angle
        angle_column = df.iloc[:, 1].values
        max_idx = np.argmax(angle_column)
        min_idx = np.argmin(angle_column)
        interesting_idxs += [max_idx, min_idx]
        #stupid, but just randomly sample the other stuff
        dfsize = len(df)
        interesting_idxs += random.sample(range(1, dfsize), num_idxs - 2)
        return interesting_idxs

    def df_data_fromidx(self, datapath, df, idx):
        row = df.iloc[idx]
        img_name = row[0]
        img_path = os.path.join(datapath, img_name)
        img = cv2.imread(img_path)
        return img, row
    
    def get_finaldf(self, src_df, dest_df, op, new_folder, dest_datapath):
        """
        Creates final dataframe
        Will overwrite unless op has 'combine' in it 
        """
        final_df = dest_df
        if op == 'aug':
            final_df = dest_df.append(src_df)
        elif 'combine' in op:
            #check if dest_datapath already has a data.csv
            try:
                curr_df = self.get_df(dest_datapath)
            except:
                curr_df = None
            if curr_df is not None:
                final_df = final_df.append(curr_df)
        return final_df

    def get_last_n_frames(self, n, datapath, idx):
        """
        Retrieve last n frames from data.csv in "datapath"
        If less than n frames available, will return as many as possible
        """
        df = self.get_df(datapath)
        src_img, src_row = self.df_data_fromidx(datapath, df, idx)
        n = min(idx, n)
        start_idx = idx - n
        datalist = [self.df_data_fromidx(datapath, df, x)[0] for x in range(start_idx, idx)]
        pdb.set_trace()
        return datalist

    def MOVE(self, src_datadir, folder, dest_datadir, flist=[], preview=False, op='mv'):
        """
        MOVE takes src_datadir/folder and moves to dest_datadir & applies flist functions to it
        also returns the new_folder name for use in dlist
        src_datadir: abs path to dir containing src data
        dest_datadir: abs path to dest dir
        folder: name of folder in src_datadir to MOVE
        flist: list of json-formatted functions (see steps.json)
        preview: if true, shows fewer entries
        op: if 'aug', augment current dataset instead of creating a whole new one & moving it elsewhere (IF SO, SRC_DATADIR MUST = DEST_DATADIR)
        """
        assert((op =='aug' and src_datadir == dest_datadir) or (op != 'aug')), "MOVE Error: If op={op}, src_datadir = dest_datadir".format(op=op, src_datadir=src_datadir)

        if not os.path.exists(dest_datadir):
            os.makedirs(dest_datadir)

        src_datapath = os.path.join(src_datadir, folder)
        new_folder, dest_datapath = self.get_dest_datapath(dest_datadir, folder, op)

        if not os.path.exists(dest_datapath):
            os.makedirs(dest_datapath)

        #make new dataframes
        src_df = self.get_df(src_datapath)
        dest_df = pd.DataFrame(columns=src_df.columns.values)

        #iterate through dataframe
        maxlen = self.get_interesting_idxs(src_datapath, 20) if preview else range(len(src_df))
        for i in maxlen:
            #Apply flist, get output
            src_img, src_row = self.df_data_fromidx(src_datapath, src_df, i)
            src_dict = {"img":src_img, "row":src_row, "src_datapath":src_datapath, "idx":i}
            dest_dict = self.apply_flist(src_dict, flist)
            #continue adding data if flag is true
            flag = dest_dict.get("flag", True)
            if flag:
                dest_row = dest_dict.get("row")
                dest_img = dest_dict.get("img")
                dest_img_name = dest_row[0]
                #TODO: Check dest_dict in a B_VER-esque way

                #Accordingly alter dataframe & write img
                dest_df = dest_df.append(dest_row)
                cv2.imwrite(os.path.join(dest_datapath, dest_img_name), dest_img)

        #write df
        final_df = self.get_finaldf(src_df, dest_df, op, new_folder, dest_datapath)
        final_df.to_csv(os.path.join(dest_datapath, 'data.csv'), index=False)
        return new_folder

    def _get_image_size(self, dpath):
        df = self.get_df(dpath)
        img_name_0 = df.iloc[0, 0]
        framepath = os.path.join(dpath, img_name_0)
        frame_0 = cv2.imread(framepath)
        h, w, c = frame_0.shape
        return h, w
    
    def polar_to_cart(self, theta, r):
        """
        Assumes theta in radians & returns x,y
        """
        x = r*math.cos(theta)
        y = r*math.sin(theta)
        return x,y

    def lidar_polar_to_cart(self, lidar_dict):
        """
        Convert a lidar_dict to cartesian & return x_ranges & y_ranges
        """
        ranges = lidar_dict.get("ranges")
        angle_min = lidar_dict.get("angle_min")
        angle_incr = lidar_dict.get("angle_increment")
        x_ranges = []
        y_ranges = []
        for i, r in enumerate(ranges):
            theta = angle_min + i * angle_incr
            x, y = self.polar_to_cart(theta + math.pi/2, r*100.0)
            x_ranges.append(x)
            y_ranges.append(y)
        return x_ranges, y_ranges

    def apply_flist(self, src_dict, flist, w_rosdict=False):
        """
        Apply a list of functions and return a dict
        src_dict: dictionary representing all source variables
        flist: partial formatted function list 
        TODO:CONSIDER if first False should end it 
        """
        dest_dict = src_dict
        for json_func in flist:
            partial_func = json_func
            if w_rosdict:
                dest_dict = partial_func(None, ros_dict=src_dict)
            else:
                dest_dict = partial_func(dest_dict)
        return dest_dict

    def initYolo(self,model_dict):
        modelDir = os.path.dirname(YoloDetection.__file__)
        paramsFile = os.path.join(modelDir,model_dict.get("paramsFile"))
        jsonFile = open(paramsFile)
        params = json.load(jsonFile)
        params = params["Yolo_params"]
        cfg_path = os.path.join(modelDir,params["cfg"])
        data_cfg = os.path.join(modelDir,params["data-cfg"])
        weights =  os.path.join(modelDir,params["weights"])
        return YoloDetection.YoloDetection(cfg_path,data_cfg,weights)

    def initDetection(self,model_dict):
        if (model_dict.get("detectType") == "yolo"):
            return self.initYolo(model_dict)

    def DETECT(self, src_datadir, folder, dest_datadir, model_dict,flist=[], preview=False):
        
        if not os.path.exists(dest_datadir):
            os.makedirs(dest_datadir)

        src_datapath = os.path.join(src_datadir, folder)
        new_folder, dest_datapath = self.get_dest_datapath(dest_datadir, folder, op="aug")

        if not os.path.exists(dest_datapath):
            os.makedirs(dest_datapath)
        #make new dataframes
        src_df = self.get_df(src_datapath)
        dest_df = pd.DataFrame(columns=src_df.columns.values)
        model = self.initDetection(model_dict)
        maxlen = self.get_interesting_idxs(src_datapath, 20) if preview else range(len(src_df))

        for i in maxlen:
            #Apply flist, get output
            src_img, src_row = self.df_data_fromidx(src_datapath, src_df, i)
            points = model.predict(src_img)
            src_dict = {"img":src_img, "row":src_row, "src_datapath":src_datapath, "idx":i, "points":points}
            dest_dict = self.apply_flist(src_dict, flist)

            flag = dest_dict.get("flag", True)
            if flag:
                dest_row = dest_dict.get("row")
                dest_img = dest_dict.get("img")
                dest_img_name = dest_row[0]

        #write df
        final_df = self.get_finaldf(src_df, dest_df, "aug", new_folder, dest_datapath,)
        final_df.to_csv(os.path.join(dest_datapath, 'data.csv'), index=False)
        return new_folder
