import time, cv2, zmq, msgpack, threading, os
from nnet.Metric_Visualizer import Metric_Visualizer
from nnet.Data_Utils import Data_Utils
from steps import session
import numpy as np
import msgpack_numpy as m
__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class ExperienceServer(threading.Thread):
    def __init__(self, open_port='tcp://*:5555'):
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.ROUTER)
        self.zmq_socket.bind(open_port)
        threading.Thread.__init__(self)

        #Visualizer & Data Utils
        self.vis = Metric_Visualizer()
        self.dutils = Data_Utils()

        #Session to save experiences
        self.exp_path = os.path.join(session["params"]["abs_path"], session["params"]["sess_root"], str(session["dagger"]["sess_id"]), "exp")
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        print("EXPERIENCE PATH:", self.exp_path)

    def show_msg(self, fullmsg):
        for i in range(len(fullmsg)):
            if i%4 == 0:
                # lidar = msgpack.unpackb(fullmsg[i])
                lidar = msgpack.loads(fullmsg[i], encoding="utf-8")
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
            self.dutils.save_batch_to_pickle(fullmsg[1:], os.path.join(self.exp_path))
            self.vis.vid_from_pkl(self.exp_path, 0, 0, show_steer=True, units='rad')
            msg = b'NN for experience %s' % (fullmsg[0])
            
            #Include where i'm sending also as a multipart
            self.zmq_socket.send_multipart([fullmsg[0],msg, b'Hello', b'mister', b'lioa'])

server = ExperienceServer()
server.start()