import time, cv2, zmq, msgpack, threading, os, pdb
from nnet.Online import Online
from nnet.Metric_Visualizer import Metric_Visualizer
from nnet.Data_Utils import Data_Utils
from nnet.train import Trainer
from nnet.datasets import *
from steps import session
import numpy as np
import msgpack_numpy as m
__author__ = 'Dhruv Karthik <dhruvkar@seas.upenn.edu>'

class ExperienceServer(threading.Thread):
    def __init__(self, open_port='tcp://*:5555', debug=False):
        
        if not debug:
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
        self.funclist = session["online"]["funclist"]
        print("EXPERIENCE PATH:", self.exp_path)
        self.debug = debug

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
        
    def dostuff(self, fullmsg, pkl_name=None):
        """
        NEEDS A BETTER NAME - But basically takes fullmsg & does stuff with it, kind of like how Stepper 'does stuff' with the OG data
        """
        if not self.debug:
            pkl_name = self.online_learner.save_batch_to_pickle(fullmsg, os.path.join(self.exp_path, 'raw'))
        
        #fix steering angles to use follow the gap
        pkl_name = self.online_learner.fix_steering(os.path.join(self.exp_path, 'raw'), pkl_name, os.path.join(self.exp_path, 'proc'))
        
        #Apply funclist to each batch
        pkl_name = self.online_learner.apply_funcs(os.path.join(self.exp_path, 'proc'), pkl_name, os.path.join(self.exp_path,'proc'), self.funclist)

        #for now, visualized processed batches live (TODO:Tensorboard Support)
        self.vis.vid_from_pklpath(os.path.join(self.exp_path, 'proc', pkl_name), 0, 0, show_steer=True, units='rad', live=True)

        #Train Model
        trainer = Trainer(online=True, pklpath=os.path.join(self.exp_path, 'proc', pkl_name), train_id=0)

        #Send model back
        modelpath = trainer.get_model_path()
        model_in = open(modelpath, 'rb')
        pdb.set_trace()

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