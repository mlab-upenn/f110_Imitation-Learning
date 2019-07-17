import os, json, pdb, cv2, math
import numpy as np
from functools import partial
import pandas as pd
from func_utils import *

class Data_Utils(object):
    """
    UPDATE WHEN I GET THIS LMAO
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
            return folder, os.path.join(dest_datadir, folder)
        elif op == 'mv':
            new_folder = str(len(os.listdir(dest_datadir))) + '_' +  folder 
        elif 'prefix' in op:
            #you need to take care that there are no name conflicts here
            strlist = op.split('|')
            assert(len(strlist) == 2), "Incorrect formatting for op = {op}"
            new_folder = strlist[1] + folder
        elif 'choosename' in op:
            #you need to take care that there are no name conflicts here
            strlist = op.split('|')
            assert(len(strlist) == 2), "Incorrect formatting for op = {op}"
            new_folder = strlist[1]
        return new_folder, os.path.join(dest_datadir, new_folder)

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


    def MOVE(self, src_datadir, folder, dest_datadir, flist=[], preview=False, op='mv'):
        """
        MOVE takes src_datadir/folder and moves to dest_datadir & applies flist functions to it
        also returns the new_folder name for use in dlist
        src_datadir: abs path to dir containing src data
        dest_datadir: abs path to dest dir
        folder: name of folder in src_datadir to MOVE
        flist: list of json-formatted functions (see steps.json)
        preview:preview shows fewer entries
        op: if 'aug', augment current dataset instead of creating a whole new one & moving it elsewhere (IF SO, SRC_DATADIR MUST = DEST_DATADIR)
        """
        assert((op =='aug' and src_datadir == dest_datadir) or (op != 'aug')), f"MOVE Error: If op={op}, src_datadir = dest_datadir"

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
        maxlen = 20 if preview else len(src_df)
        for i in range(maxlen):
            #Apply flist, get output
            src_img, src_row = self.df_data_fromidx(src_datapath, src_df, i)
            src_dict = {"img":src_img, "row":src_row}
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

    def get_partial_func(self, json_func):
        """
        json_func: json formatted function
        Functions are in func_utils.py
        """
        fname, args = json_func["F"], json_func["args"]

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
        elif fname == 'filterBadData':
            p = partial(filterBadData, args)
        elif fname == 'rescaleImg':
            p = partial(rescaleImg, args)
        else:
            raise Exception('{fname} is not in the list of functions')
        return p

    def _get_image_size(self, dpath):
        df = self.get_df(dpath)
        img_name_0 = df.iloc[0, 0]
        framepath = os.path.join(dpath, img_name_0)
        frame_0 = cv2.imread(framepath)
        h, w, c = frame_0.shape
        return h, w

    def apply_flist(self, src_dict, flist):
        """
        Apply a list of functions and return a dict
        src_dict: dictionary representing all source variables
        flist: json formatted function list (see steps.json)
        TODO:CONSIDER if first False should end it 
        """
        dest_dict = src_dict
        for json_func in flist:
            partial_func = self.get_partial_func(json_func)
            dest_dict = partial_func(dest_dict)
        return dest_dict