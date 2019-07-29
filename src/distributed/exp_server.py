import cv2, msgpack, os, time, threading, zmq
import numpy as np
import msgpack_numpy as m

__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class ExperienceServer(threading.Thread):
    def __init__(self, recv_callback, open_port='tcp://*:5555'):
        
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.ROUTER)
        self.zmq_socket.bind(open_port)
        self.recv_callback = recv_callback
        m.patch()
        
        threading.Thread.__init__(self)

    def run(self):
        while True:
            multipart_msg = self.zmq_socket.recv_multipart()
            header_dict = msgpack.loads(multipart_msg[1], encoding="utf-8")
            print('RECVD BATCH:', header_dict.get("batchnum"))
            dump_msg = self.recv_callback(multipart_msg[2:])
            dump_array = [multipart_msg[0], multipart_msg[1], dump_msg]
            self.zmq_socket.send_multipart(dump_array)