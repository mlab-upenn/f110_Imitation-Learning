import time
import zmq
import msgpack
import msgpack_numpy
from common.imagezmq import SerializingContext
class f110Server(object):
    """
    Opens a zmq REQ socket & recieves ROS data 
    """
    def __init__(self, open_port='tcp://*:5555'):
        self.zmq_context = SerializingContext()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind(open_port)
    
    def recv_data(self, copy=False):
        # lidar = msgpack.unpack(self.zmq_socket.recv())
        # steer = msgpack.unpack(self.zmq_socket.recv())
        md, cv_img = self.zmq_socket.recv_array(copy=False)
        # print(lidar, steer)
        cv2.imshow('Bastards', cv_img)
        self.zmq_socket.send(b"World")

        
server = f110Server()
while True:
    server.recv_data()
