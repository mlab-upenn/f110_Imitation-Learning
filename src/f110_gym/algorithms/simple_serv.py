from __future__ import print_function
from distributed.exp_server import ExperienceServer
import cv2, random, threading, msgpack
import msgpack_numpy as m
import numpy as np

__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'

def deserialize_obs():
    def _deser(multipart_msg):
        lidar = msgpack.loads(multipart_msg[0], encoding="utf-8")
        steer = msgpack.unpackb(multipart_msg[1], encoding="utf-8")
        md = msgpack.unpackb(multipart_msg[2])
        cv_img = multipart_msg[3]
        cv_img = np.frombuffer(cv_img, dtype=md[b'dtype'])
        cv_img = cv_img.reshape(md[b'shape'])
        obs_dict = {"img":cv_img, "lidar":lidar, "steer":steer}
        return obs_dict
    return _deser

def show_msg(obs_array):
    print(len(obs_array))
    randomstuff= {'NN':'dingus'}
    dump = msgpack.dumps(randomstuff)
    return dump

serv = ExperienceServer(show_msg, deserialize_obs(), 4)
serv.start()
serv.join()