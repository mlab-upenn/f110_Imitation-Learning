import cv2, msgpack, os, time, threading, zmq
import numpy as np
import msgpack_numpy as m

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class ExperienceServer(threading.Thread):
    def __init__(self, open_port='tcp://*:5555', ):
        
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.ROUTER)
        self.zmq_socket.bind(open_port)
        m.patch()
        threading.Thread.__init__(self)