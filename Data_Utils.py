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

    if cropEnd < 0:
        dest_dict["img"] = src_img[cropStart:, :, :]
    elif cropEnd > 0:
        dest_dict["img"] = src_img[cropStart:cropEnd, :, :]
    else:
        raise Exception('bad args cropvertical')
    return dest_dict

def rot90(args, src_dict):
    assert (len(args) == 1),"Incorrect sized argument to rot90"
    direction = args[0]
    dest_dict = src_dict
    src_img = src_dict.get("img")

    if direction == 'clockwise':
        dest_dict["img"] = cv2.rotate(src_img, cv2.ROTATE_90_CLOCKWISE)

    elif direction == 'anticlockwise':
        #TODO: DO ANTICLOCKWISE
        pass
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
    return dest_dict

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
    
    def get_finaldf(self, src_df, dest_df, op):
        final_df = dest_df
        if op == 'aug':
            final_df = dest_df.append(src_df)
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
        assert((op =='aug' and src_datadir != dest_datadir) or (op != 'aug')), f"MOVE Error: If op={op}, src_datadir = dest_datadir"

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
        maxlen = 100 if preview else len(src_df)
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
                dest_img_name = dest_dict.get("img_name", dest_row[0])

                #TODO: Check dest_dict in a B_VER-esque way

                #Accordingly alter dataframe & write img
                dest_df = dest_df.append(dest_row)
                cv2.imwrite(os.path.join(dest_datapath, dest_img_name), dest_img)

        #write df
        final_df = self.get_finaldf(src_df, dest_df, op)
        final_df.to_csv(os.path.join(dest_datapath, 'data.csv'), index=False)
        return new_folder

    def get_partial_func(self, json_func):
        """
        json_func: json formatted function
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
        else:
            raise Exception('{fname} is not in the list of functions')
        return p

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