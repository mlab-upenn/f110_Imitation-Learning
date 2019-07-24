import time, cv2, zmq, msgpack, threading, os
from nnet.Online import Online
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
        self.online_learner = Online()

        #Session to save experiences
        self.exp_path = os.path.join(session["params"]["abs_path"], session["params"]["sess_root"], str(session["online"]["sess_id"]), "exp")
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        print("EXPERIENCE PATH:", self.exp_path)

    def show_msg(self, fullmsg):
        for i in range(len(fullmsg)):
            if i%4 == 0:
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
        
    def dostuff(self, fullmsg):
        """
        NEEDS A BETTER NAME - But basically takes fullmsg & does stuff with it, kind of like how Stepper 'does stuff' with the OG data
        """
        self.online_learner.save_batch_to_pickle(fullmsg, self.exp_path)

        #fix steering angles to use follow the gap
        self.online_learner.fix_steering(self.exp_path)
        
        #for now, visualize batches live (do tensorboard stuff soon)
        self.vis.vid_from_online_dir(self.exp_path, 0, 0, show_steer=True, units='rad', live=True)

    def run(self):
        while True:
            fullmsg = self.zmq_socket.recv_multipart()
            print('IDENT:', fullmsg[0])

            #Do stuff with fullmsg & get an nn
            self.dostuff(fullmsg[1:])
            msg = b'NN for experience %s' % (fullmsg[0])
            #Include where i'm sending also as a multipart
            self.zmq_socket.send_multipart([fullmsg[0],msg, b'Hello', b'mister', b'lioa'])

server = ExperienceServer()
server.start()