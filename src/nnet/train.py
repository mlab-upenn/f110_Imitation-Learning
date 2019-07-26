import os, torch, logging, random, pdb, json, time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from steps import session
import numpy as np
from nnet.Metric_Visualizer import Metric_Visualizer
from nnet.models import NVIDIA_ConvNet
try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("cannot fully use Trainer without tensorboardX")
device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 
class Trainer(object):
    """
    Handles training & associated functions
    """
    def __init__(self, online=False, pklpath=None, train_id=None):
        #Set random seeds
        seed = 6582
        torch.manual_seed(seed)
        if torch.cuda.is_available:
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed) 
        self.sess_path = None
        self.datapath = None
        self.gconf = None
        self.train_id = train_id
        self.config, dataset, net, optim, loss_func, num_epochs, bs = self.configure_train(online=online, pklpath=pklpath, continue_training=online) #sets sess_path, datapath & gconf
        train_dataloader, valid_dataloader = self.get_dataloaders(dataset, bs, online=online)
        
        #Make Writer & Visualize
        self.writer = SummaryWriter(logdir=self.logpath_prefix)
        self.vis = Metric_Visualizer(self.sess_path, self.writer)
    
        #TRAIN!
        self.TRAIN(net, num_epochs, optim, loss_func, train_dataloader, valid_dataloader)

    def loss_pass(self, net, loss_func, loader, epoch, optim, op='train'):
        """
        Performs one epoch & continually updates the model 
        """
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

    def TRAIN(self, net, num_epochs, optim, loss_func, train_dataloader, valid_dataloader=None):
        """
        Main training loop over epochs
        """
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
                torch.save(net.state_dict(), os.path.join(self.logpath_prefix, str('best_train_model')))

            if valid_dataloader and best_valid_loss > valid_epoch_loss:
                best_valid_loss = valid_epoch_loss
                torch.save(net.state_dict(), os.path.join(self.logpath_prefix, str('best_valid_model')))

            self.writer.add_scalar('Train Loss', train_epoch_loss, epoch)
            if valid_dataloader:
                self.writer.add_scalar('Valid Loss', valid_epoch_loss, epoch)
        self.vis.log_training(self.config, self.train_id, best_train_loss, best_valid_loss)
        self.writer.close()
            
    def get_dataloaders(self, dataset, bs, online=False):
        """
        Get train and valid dataloader based on vsplit parameter defined in steps.session 
        """
        if online:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
            return dataloader, None
        vsplit = self.gconf("vsplit")
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

    def configure_train(self, online=False, pklpath=None, continue_training=False):
        """
        Get initialized parameters for training
        """
        params = session.get("params")
        config = session.get("train")
        self.sess_path = os.path.join(params.get("abs_path"), params.get("sess_root"), str(config.get("sess_id")))
        if online:
            self.datapath = pklpath
        else:
            self.datapath = os.path.join(params.get("abs_path"), params.get("sess_root"), str(config.get("sess_id")), config.get("foldername"))
        print("Datapath", self.datapath)

        logdir = os.path.join(self.sess_path, "models")
        if self.train_id is None:
            self.train_id = len(os.listdir(logdir))  

        #get training parameters from file
        self.gconf = lambda key: config.get(key)
        self.logpath_prefix = os.path.join(logdir, str(self.train_id), )
        model = self.gconf("model")
        dataset = self.gconf("dataset")(self.datapath)
        net = self.make_net(model, dataset, continue_training=continue_training)
        lr = self.gconf("lr")
        optim = self.gconf("optimizer")(net.parameters(), lr=lr)
        loss_func = self.gconf("loss_func")
        num_epochs = self.gconf("num_epochs")
        bs = self.gconf("batch_size")
        return config, dataset, net, optim, loss_func, num_epochs, bs
        
    def hasLinear(self, net):
        """
        Checks a net for linear layers
        """
        for idx, m in enumerate(net.named_modules()):
            if 'fc' in m[0]:
                return True
        return False

    def get_fc_shape(self, dummy_net, dataset):
        """
        Get the fc dimensions of images of a dataset and a model
        dummy_net: Helps us get the tensor shape after all the conv layers
        """
        input_dict = dataset[0]
        input_dict["img"] = input_dict["img"][None]
        out_dict = dummy_net.only_conv(input_dict)
        out = out_dict["img"]
        out = out.view(1, -1)
        return out.shape[1]
    
    def make_net(self, model, dataset, continue_training=False):
        """
        Return an initialized neural net & fix the first fully connected layer to match the image input size
        """
        dummy_net = model()
        if self.hasLinear(dummy_net):
            #find the right shape of fc layer
            fc_shape = self.get_fc_shape(dummy_net, dataset)
            print("FCSHAPEIS", fc_shape)
            net = model(args_dict={"fc_shape":fc_shape})
        else:
            net = model()
        net = net.to(device)
        if continue_training:
            modelpath = os.path.join(self.logpath_prefix, str('best_train_model'))
            if os.path.exists(modelpath):
                net.load_state_dict(torch.load(modelpath))
        return net
    
    def get_model_path(self):
        modelpath = os.path.join(self.logpath_prefix, str('best_train_model'))
        return modelpath

def main():
    trainer = Trainer()

if __name__ == "__main__":
    main()
