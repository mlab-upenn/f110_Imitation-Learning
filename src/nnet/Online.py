import os, json, pdb, cv2, math, random, pickle, msgpack, sys
import msgpack_numpy as m
import numpy as np
from nnet.Data_Utils import Data_Utils
from steps import session
from nnet.oracles.FGM import FGM
from nnet.Metric_Visualizer import Metric_Visualizer

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class Online(object):
    """
    Useful functions for Online Learning
    TODO:
    1.Read path of online pickle files
    2.Fix the online readings 
    """
    def __init__(self):
        self.vis = Metric_Visualizer()
        self.oracle = FGM()
        self.dutils = Data_Utils()
        self.seen_pkls = []

    def pickledump(self, dump_array, dump_path, replace=True):
        if replace:
            if os.path.exists(dump_path):
                os.remove(dump_path)
        dump_out = open(dump_path, "wb")
        pickle.dump(dump_array, dump_out)

    def save_obsarray_to_pickle(self, obs_array, dest_dir):
        """ Saves deserialized dump to a pkl file in self.online_sess_dir
        """
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        dump_array = obs_array
        pkl_name = 'batch' + str(len(os.listdir(dest_dir)))
        dump_path = os.path.join(dest_dir, pkl_name)
        self.pickledump(dump_array, dump_path)
        return pkl_name

    def save_obsdump_to_pickle(self, fullmsg, dest_dir):
        """
        Saves serialized dump to a pkl file in self.online_sess_dir
        """
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        dump_array = []
        for i in range(len(fullmsg)):
            if i%4 == 0:
                lidar = msgpack.loads(fullmsg[i], encoding="utf-8")
                steer = msgpack.unpackb(fullmsg[i+1], encoding="utf-8")
                md = msgpack.unpackb(fullmsg[i+2])
                cv_img = fullmsg[i+3]
                cv_img = np.frombuffer(cv_img, dtype=md[b'dtype'])
                cv_img = cv_img.reshape(md[b'shape'])
                dump_array.append({"img":cv_img, "lidar":lidar, "steer":steer})
        pkl_name = 'batch' + str(len(os.listdir(dest_dir)))
        dump_path = os.path.join(dest_dir, pkl_name)
        self.pickledump(dump_array, dump_path)
        return pkl_name

    def fix_steering(self, src_dir, pkl_name, dest_dir):
        """
        Use an Oracle/Expert Policy to fix steering angles
        src_dir:abs path to current dir containing pkl file
        pkl_name: name of pkl file to fix
        dest_dir:final directory (will change pkl filename accordingly)
        """
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
    
        dump_array = []
        data_in = open(os.path.join(src_dir, pkl_name), 'rb')
        data_array = pickle.load(data_in)
        for i, data_dict in enumerate(data_array):
            try:
                new_data_dict = self.oracle.fix(data_dict)
            except Exception as e:
                print(e)
                new_data_dict["flag"] = False
            if new_data_dict.get("flag", True):
                dump_array.append(new_data_dict)
        new_pkl_name = 'proc_' + pkl_name
        dump_path = os.path.join(dest_dir, new_pkl_name)
        self.pickledump(dump_array, dump_path)
        return new_pkl_name
        
    def apply_funcs(self, src_dir, pkl_name, dest_dir, funclist):
        """
        Apply a series of functions and RET new pkl name
        """
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        pkl_files = os.listdir(src_dir)
        data_in = open(os.path.join(src_dir, pkl_name), 'rb')
        data_array = pickle.load(data_in)
        dump_array = []
        for i, data_dict in enumerate(data_array):
            new_data_dict = self.dutils.apply_flist(data_dict, funclist, w_rosdict=True)
            if new_data_dict.get("flag", True):
                dump_array.append(new_data_dict)
        new_pkl_name = pkl_name
        dump_path = os.path.join(dest_dir, new_pkl_name)
        self.pickledump(dump_array, dump_path)
        self.seen_pkls.append(new_pkl_name)
        return new_pkl_name