import os, json, pdb, cv2, math, random, pickle, msgpack
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
        self.seen_pkls = []

    def save_batch_to_pickle(self, fullmsg, dest_dir):
        """
        Saves batch to a pkl file in self.online_sess_dir
        """
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
        dump_path = os.path.join(dest_dir, 'batch'+str(len(os.listdir(dest_dir))))
        dump_out = open(dump_path, "wb")
        pickle.dump(dump_array, dump_out)

    def fix_steering(self, dest_dir):
        """
        Use an Oracle/Expert Policy to fix steering angles
        """
        pkl_files = os.listdir(dest_dir)
        for pkl in pkl_files:
            print(os.path.join(dest_dir, pkl))
            data_in = open(os.path.join(dest_dir, pkl), 'rb')
            data_array = pickle.load(data_in)
            for i, data_dict in enumerate(data_array):
                frame = self.vis.frame_from_datadict(data_dict)
                cv2.imshow("OG_FRAME", frame)
                cv2.waitKey(3)
                new_data_dict = self.oracle.fix(data_dict)
                frame = self.vis.frame_from_datadict(data_dict)
                cv2.imshow("NEW FRAME", frame)
                cv2.waitKey(0)
                self.seen_pkls.append(pkl)