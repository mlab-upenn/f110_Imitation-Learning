from __future__ import print_function
import os, random, time, pickle
from common.steps import session

#torch imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

#nnet & logging imports
from nnet.Online import Online
from nnet.Metric_Visualizer import Metric_Visualizer
try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("cannot fully use Trainer without tensorboardX")

device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 

__author__ = "Dhruv Karthik <dhruvkar@seas.upenn.edu>"

class Trainer(object):
    """
    Uses "session" (in steps.py) to perform training
    """
    def __init__(self, sess_type="train"):
        self.config = session.get(sess_type)
        self.exp_path = self.get_exp_path()
        self.modelpath = self.get_model_path()
        self.modelname = self.get_model_name()
        self.train_id = self.config["sess_id"]
        self.train_writer = SummaryWriter(logdir=self.modelpath)
        self.train_vis = Metric_Visualizer(writer=self.train_writer)

    ####### TRAINING FUNCTIONS ##########
    def view_model(self, pkl_list):
        self.seed_env()
        model, noinit_dataset, optim, loss_func, num_epochs, bs = self.configure_train()
        dataset = self.get_dataset(noinit_dataset, pkl_list)
        print("MODEL:")
        print(model)
        if self.hasLinear(model):
            print("GET FC_SHAPE", self.get_fc_shape(dataset, model))

    def train_model(self, pkl_list):
        """
        Train model on a list of pkls
        """
        self.seed_env()
        model, noinit_dataset, optim, loss_func, num_epochs, bs = self.configure_train()
        dataset = self.get_dataset(noinit_dataset, pkl_list)
        train_dataloader, valid_dataloader = self.get_dataloader(dataset, bs)
        self.TRAIN(model, num_epochs, optim, loss_func, train_dataloader, valid_dataloader=valid_dataloader)
    
    def get_model_name(self):
        modelname = self.config["modelname"]
        return modelname
    
    def get_exp_path(self):
        exp_path = os.path.join(self.config["root"], "runs", str(self.config["sess_id"]), "exp")
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        print("EXPERIENCE PATH:", exp_path)
        return exp_path
    
    def get_model_path(self):
        model_path = os.path.join(self.exp_path, "models")
        return model_path

    def TRAIN(self, net, num_epochs, optim, loss_func, train_dataloader, valid_dataloader=None):
        """
        Main training loop over epochs
        """
        metadata = self.get_metadata()
        base_epoch = metadata["base_epoch"]
        print(base_epoch)

        best_train_loss = float('inf')
        best_valid_loss = float('inf')
        for epoch in range(num_epochs):
            print("Starting epoch: {}".format(epoch))
            train_epoch_loss = self.loss_pass(net, loss_func, train_dataloader, epoch, optim, op='train')
            if valid_dataloader:
                valid_epoch_loss = self.loss_pass(net, loss_func, valid_dataloader, epoch, optim, op='valid')
            print("----------------EPOCH{}STATS:".format(epoch))
            print("TRAIN LOSS:{}".format(train_epoch_loss))
            if valid_dataloader:
                print("VALIDATION LOSS:{}".format(valid_epoch_loss))
            print("----------------------------")

            if best_train_loss > train_epoch_loss:
                best_train_loss = train_epoch_loss
                torch.save(net.state_dict(), os.path.join(self.modelpath, 'train_' + self.modelname))

            if valid_dataloader and best_valid_loss > valid_epoch_loss:
                best_valid_loss = valid_epoch_loss
                torch.save(net.state_dict(), os.path.join(self.modelpath, "valid_" + self.modelname))

            self.train_writer.add_scalar('Train Loss', train_epoch_loss, base_epoch + epoch)
            if valid_dataloader:
                self.train_writer.add_scalar('Valid Loss', valid_epoch_loss, base_epoch + epoch)

        self.save_metadata(base_epoch + epoch+1)
        self.train_vis.log_training(self.config, self.train_id, best_train_loss, best_valid_loss)
        traintable = self.train_vis.get_train_table(self.config, self.train_id, best_train_loss, best_valid_loss)
        self.save_traintable(traintable)
        self.train_writer.close()
    
    def save_traintable(self, traintable):
        savepath = os.path.join(self.exp_path, "session_table.md")
        f = open(savepath, "wt")
        f.write(traintable)
        f.close()
    
    def save_metadata(self, epoch, modeltype='train'):
        metadata = {"base_epoch":epoch}
        mp = os.path.join(self.modelpath, modeltype + '_' + self.modelname + '.pkl')
        pickle.dump(metadata, open(mp, "wb"))

    def get_metadata(self, modeltype='train'):
        mp = os.path.join(self.modelpath, modeltype + '_' + self.modelname + '.pkl')
        if os.path.exists(mp):
            metadata = pickle.load(open(mp, "rb"))
        else:
            metadata = {"base_epoch":0}
        return metadata

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
                self.train_vis.visualize_batch(ts_imgbatch, ts_anglebatch, ts_predanglebatch, global_step=epoch)
        if op == 'valid':
            torch.set_grad_enabled(True)
        print("FINISHED {op} EPOCH{epoch}".format(op=op, epoch=epoch))
        print("----{now} seconds----".format(now=time.time()-t0, op=op, epoch=epoch))
        return total_epoch_loss
    
    def get_transforms(self):
        tflist = self.config["transforms"]
        if len(tflist) > 0:
            tf = transforms.Compose(tflist)
        else:
            tf = None
        return tf

    def get_dataset(self, noinit_dataset, pkl_list, relative=True):
        dsetlist = []
        transforms = self.get_transforms()
        for pkl in pkl_list:
            if relative:
                pklpath = os.path.join(self.exp_path, 'data', pkl)
            else:
                pklpath = pkl
            dsetlist.append(noinit_dataset(pklpath, transforms=transforms))
        dataset = ConcatDataset(dsetlist)
        print("DSET SIZE:", len(dataset))
        return dataset

    def get_dataloader(self, dataset, bs):
        vsplit = self.config.get("vsplit")

        if vsplit == 0.0:
            train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            return train_dataloader, None

        dset_size = len(dataset)
        idxs = list(range(dset_size))
        split = int(np.floor(vsplit * dset_size))
        np.random.shuffle(idxs)
        train_idxs, val_idxs = idxs[split:], idxs[:split]

        #Using SubsetRandomSampler but should ideally sample equally from each steer angle to avoid distributional bias
        train_sampler = SubsetRandomSampler(train_idxs)
        val_sampler = SubsetRandomSampler(val_idxs)

        train_dataloader = DataLoader(dataset, batch_size=bs, sampler=train_sampler)
        valid_dataloader = DataLoader(dataset, batch_size=bs, sampler=val_sampler)
        return train_dataloader, valid_dataloader

    def seed_env(self):
        seed = 6582
        torch.manual_seed(seed)
        if torch.cuda.is_available:
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed) 

    def configure_train(self, modeltype='train'):
        self.seed_env()
        model = self.config.get("model")().to(device)

        #check for existing model to load
        if os.path.exists(os.path.join(self.modelpath, modeltype + '_' + self.modelname)):
            print("FOUND EXISTING MODEL")
            model.load_state_dict(torch.load(os.path.join(self.modelpath, modeltype + '_' + self.modelname)))

        dataset = self.config.get("dataset")
        lr = self.config.get("lr")
        optim = self.config.get("optimizer")(model.parameters(), lr=lr)
        loss_func = self.config.get("loss_func")
        num_epochs = self.config.get("num_epochs")
        bs = self.config.get("batch_size")
        return model, dataset, optim, loss_func, num_epochs, bs

        ###CHECKING FC LAYERS
    def hasLinear(self, net):
        """
        Checks a net for linear layers
        """
        for idx, m in enumerate(net.named_modules()):
            if 'fc' in m[0]:
                return True
        return False

    def get_fc_shape(self, dataset, dummy_net):
        """
        Get the fc dimensions of images of a dataset and a model
        dummy_net: Helps us get the tensor shape after all the conv layers
        """
        input_dict = dataset[0]
        input_dict["img"] = input_dict["img"][None].to(device)
        out_dict = dummy_net.only_conv(input_dict)
        out = out_dict["img"]
        out = out.view(1, -1)
        return out.shape[1]
    #####TRAIN FUNCTIONS ################### 