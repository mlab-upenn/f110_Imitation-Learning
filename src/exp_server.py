import time, cv2, zmq, msgpack, threading
import numpy as np
import msgpack_numpy as m
__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class ExperienceServer(threading.Thread):
    def __init__(self, open_port='tcp://*:5555'):
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.ROUTER)
        self.zmq_socket.bind(open_port)
        threading.Thread.__init__(self)

    def show_msg(self, fullmsg):
        for i in range(len(fullmsg)):
            if i%4 == 0:
                lidar = msgpack.unpackb(fullmsg[i])
                print(lidar.keys())
                steer = msgpack.unpackb(fullmsg[i+1])
                print(steer.keys())
                md = msgpack.unpackb(fullmsg[i + 2])
                print(md.keys())
                cv_img = fullmsg[i+3]
                print('---------------------')
                cv_img = np.frombuffer(cv_img, dtype=md[b'dtype'])
                cv_img = cv_img.reshape(md[b'shape'])
                print(cv_img.shape)
                cv2.imshow('BatchImages', cv_img)
                cv2.waitKey(0)
        
    def run(self):
        while True:
            fullmsg = self.zmq_socket.recv_multipart()
            print('IDENT:', fullmsg[0])
            self.show_msg(fullmsg[1:])
            msg = b'NN for experience %s' % (fullmsg[0])
            
            #Include where i'm sending also as a multipart
            self.zmq_socket.send_multipart([fullmsg[0],msg, b'Hello', b'mister', b'lioa'])

server = ExperienceServer()
server.start()