import os, torch, logging, random, pdb, json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from steps import session
import numpy as np
from models import NVIDIA_ConvNet
from tensorboardX import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 
class Trainer(object):
    """
    Handles training & associated functions
    """
    def __init__(self):
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
        config, dataset, net, optim, loss_func, num_epochs, bs = self.configure_train() #sets sess_path, datapath & gconf
        train_dataloader, valid_dataloader = self.get_dataloaders(dataset, bs)
        
        #Make Writer
        self.writer = SummaryWriter(logdir=os.path.joint(self.sess_path, "logs"))

        #TRAIN!
        self.TRAIN(net, num_epochs, optim, loss_func, train_dataloader, valid_dataloader)

    def loss_pass(self, net, loss_func, train_dataloader, epoch, optim, train=True):
        """
        Performs one epoch & continually updates the model 
        """
        if not train:
            torch.set_grad_enabled(False)
            print(f"STARTING VALIDATION EPOCH{epoch}")
            

    def TRAIN(self, net, num_epochs, optim, loss_func, train_dataloader, valid_dataloader):
        """
        Main training loop over epochs
        """
        best_train_loss = float('inf')
        best_valid_loss = float('inf')
        for epoch in range(num_epochs):
            print("Starting epoch: {}".format(epoch))
            train_epoch_loss = self.loss_pass(net, loss_func, train_dataloader, epoch, optim, train=True)
            valid_epoch_loss = self.loss_pass(net, loss_func, train_dataloader, epoch, optim, train=False)
        print("----------------EPOCH{}STATS:".format(epoch))
        print("TRAIN LOSS:{}".format(train_epoch_loss))
        print("VALIDATION LOSS:{}".format(valid_epoch_loss))
        print("----------------------------")

    def get_dataloaders(self, dataset, bs):
        """
        Get train and valid dataloader based on vsplit parameter defined in steps.session 
        """
        vsplit = self.gconf("vsplit")
        dset_size = len(dataset)
        idxs = list(range(dset_size))
        split = int(np.floor(vsplit * dset_size))
        np.random.shuffle(idxs)
        train_idxs, val_idxs = idxs[split:], idxs[:split]

        #Using SubsetRandomSampler but should ideally sample equally form each steer angle to avoid distributional bias
        train_sampler = SubsetRandomSampler(train_idxs)
        val_sampler = SubsetRandomSampler(val_idxs)

        train_dataloader = DataLoader(dataset, batch_size=bs, sampler=train_sampler)
        valid_dataloader = DataLoader(dataset, batch_size=bs, sampler=val_sampler)
        return train_dataloader, valid_dataloader

    def configure_train(self):
        """
        Get initialized parameters for training
        """
        params = session.get("params")
        config = session.get("train")
        self.sess_path = os.path.join(params.get("abs_path"), params.get("sess_root"), str(config.get("sess_id")))
        self.datapath = os.path.join(params.get("abs_path"), params.get("sess_root"), str(config.get("sess_id")), config.get("foldername"))
        print("Datapath", self.datapath)

        #get training parameters from file
        self.gconf = lambda key: config.get(key)
        model = self.gconf("model")
        dataset = self.gconf("model")(self.datapath)
        net = self.make_net(model, dataset)
        lr = self.gconf("lr")
        optim = self.gconf("optimizer")(net.parameters, lr=lr)
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
        pdb.set_trace()
        input_dict = dataset[0]
        input_dict["img"] = input_dict["img"][None]
        out_dict = dummy_net.only_conv(input_dict)
        out = out_dict["img"]
        out = out.view(1, -1)
        return out.shape[1]
    
    def make_net(self, model, dataset):
        """
        Return an initialized neural net & fix the first fully connected layer to match the image input size
        """
        dummy_net = model()
        if self.hasLinear(dummy_net):
            #find the right shape of fc layer
            fc_shape = self.get_fc_shape(dummy_net, dataset)
            net = model({"fc_shape":fc_shape})
        net = model()
        return net.to(device)

def main():
    trainer = Trainer()
    
#     config = configure_train()

#     #Choose Net + Parameters for training
#     net = NVIDIA_ConvNet().to(device)
#     loss_func = nn.functional.mse_loss
#     optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
#     num_epochs = 300
#     batch_size = 128
    
#     #Make Dataloaders
#     dutils = Data_Utils()
#     train_dataloader, valid_dataloader = dutils.get_dataloaders(batch_size)

#     #TRAIN!
#     train(net, num_epochs, optimizer, loss_func, train_dataloader, valid_dataloader)

if __name__ == "__main__":
    main()
