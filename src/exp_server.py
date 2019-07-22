import time
import cv2
import zmq
import msgpack, threading
import msgpack_numpy as m
__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class ExperienceServer(threading.Thread):
    def __init__(self, open_port='tcp://*:5555'):
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.ROUTER)
        self.zmq_socket.bind(open_port)
        threading.Thread.__init__(self)

    def run(self):
        while True:
            ident, msg = self.zmq_socket.recv_multipart()
            print('IDENT:', ident)
            print('MSG:', msg)
            time.sleep(20)
            msg = b'NN for experience %s' % (msg)
            #Include where i'm sending also as a multipart
            self.zmq_socket.send_multipart([ident,msg, b'Hello', b'mister', b'lioa'])
 