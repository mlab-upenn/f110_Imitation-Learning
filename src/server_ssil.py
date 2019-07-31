from __future__ import print_function
from f110_gym.distributed.exp_server import ExperienceServer
import cv2, random, threading, msgpack, os, time
from datasets import SteerDataset_ONLINE
from models import NVIDIA_ConvNet

#torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#nnet & logging imports
from nnet.Online import Online
from nnet.Metric_Visualizer import Metric_Visualizer
try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("cannot fully use Trainer without tensorboardX")

#misc imports
from functools import partial
import msgpack_numpy as m
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 

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

class SSIL_server(object):
    """
    Serverside Class for Self Supervised Imitation Learning (runs on the server)
    """
    def __init__(self):
        self.onl = Online()
        self.vis = Metric_Visualizer()
        self.serv = ExperienceServer(self.ob_callback, deserialize_obs(), 4)
        self.exp_path = self.get_exp_path()
        self.modelpath = self.get_model_path()
        m.patch()

        self.train_writer = SummaryWriter(logdir=os.path.join(self.exp_path, 'models'))
        self.train_vis = Metric_Visualizer(writer=self.train_writer)

    def get_exp_path(self):
        exp_path = os.path.join("/home/dhruvkar/datasets/avfone", "runs", "0", "exp")
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        print("EXPERIENCE PATH:", exp_path)
        return exp_path
    
    def get_model_path(self):
        model_path = os.path.join(self.exp_path, "models", "model")
        return model_path

    def ob_callback(self, obs_array):
        pkl_name = self.onl.save_obsarray_to_pickle(obs_array, os.path.join(self.exp_path, 'data'))

        self.vis.vid_from_pklpath(os.path.join(self.exp_path, 'data', pkl_name), 0, 0, show_steer=True, units='rad', live=True)

        self.train_model(os.path.join(self.exp_path, 'data', pkl_name))

        #Send model back
        with open(self.modelpath, 'rb') as binary_file:
            model_dump = bytes(binary_file.read())
        return [model_dump]

    ####### TRAINING FUNCTIONS ##########
    def train_model(self, pkl_path):
        self.seed_env()
        model, dataset, optim, loss_func, num_epochs, bs = self.configure_train(pkl_path)
        dataloader = self.get_dataloader(dataset, bs)
        self.TRAIN(model, num_epochs, optim, loss_func, dataloader)

    def TRAIN(self, model, num_epochs, optim, loss_func, dataloader):
        best_train_loss = float('inf')
        for epoch in range(num_epochs):
            print("Starting epoch: {}".format(epoch))
            train_epoch_loss = self.loss_pass(model, loss_func, dataloader, epoch, optim, op='train')
            print("----------------EPOCH{}STATS:".format(epoch))
            print("TRAIN LOSS:{}".format(train_epoch_loss))
            if best_train_loss > train_epoch_loss:
                best_train_loss = train_epoch_loss
                torch.save(model.state_dict(), os.path.join(self.modelpath, str('model')))
            #potentially do some tensorboard logging
        self.train_writer.close()

    def loss_pass(self, net, loss_func, loader, epoch, optim, op='train'):
        if op == 'valid':
            torch.set_grad_enabled(False)
            
        print("STARTING {op} EPOCH{epoch}".format(op=op, epoch=epoch))
        t0 = time.time()
        total_epoch_loss = 0
        for i, input_dict in enumerate(loader):
            ts_imgbatch, ts_anglebatch = input_dict.get("img"), input_dict.get("angle")
            ts_imgbatch, ts_anglebatch = ts_imgbatch.to(device), ts_anglebatch.to(device)
            input_dict["img"] = ts_imgbatch
            input_dict["angle"] = ts_imgbatch
            #Classic train loop
            optim.zero_grad()
            out_dict = net(input_dict)
            ts_predanglebatch = out_dict["angle"]
            ts_loss = loss_func(ts_predanglebatch, ts_anglebatch)
            if op=='train':
                ts_loss.backward()
                optim.step()
            print("loss:{}".format(ts_loss.item()))
            total_epoch_loss += ts_loss.item() 
            if i % 20 == 0:
                self.vis.visualize_batch(ts_imgbatch, ts_anglebatch, ts_predanglebatch, global_step=epoch)
        if op == 'valid':
            torch.set_grad_enabled(True)
        print("FINISHED {op} EPOCH{epoch}".format(op=op, epoch=epoch))
        print("----{now} seconds----".format(now=time.time()-t0, op=op, epoch=epoch))
        return total_epoch_loss

    def get_dataloader(self, dataset, bs):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        return dataloader

    def seed_env(self):
        seed = 6582
        torch.manual_seed(seed)
        if torch.cuda.is_available:
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed) 

    def configure_train(self, pkl_path):
        self.seed_env()
        model = NVIDIA_ConvNet().to(device)

        #check for existing model to load
        if os.path.exists(self.modelpath):
            model.load_state_dict(torch.load(self.modelpath))

        dataset = SteerDataset_ONLINE(pkl_path)
        lr = 1e-3
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        loss_func = nn.functional.mse_loss
        num_epochs = 10
        bs = 16
        return model, dataset, optim, loss_func, num_epochs, bs
    #####TRAIN FUNCTIONS ###################