from common.datasets import SteerDataset
from common.models import NVIDIA_ConvNet
import os, pickle, random, time

#torch imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

#Logging
from tensorboardX import SummaryWriter
train_writer = SummaryWriter(logdir="../logs")

device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 

__author__ = "Dhruv Karthik <dhruvkar@seas.upenn.edu>"
FOLDERPATH = "./sim_train"

def seed_env():
    seed = 6582 
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 

def load_train_metadata():
    mp = os.path.join(FOLDERPATH, "train_metadata")
    if os.path.exists(mp):
        metadata = pickle.load(open(mp, "rb"))
    else:
        metadata = {"base_epoch": 0}
    return metadata

def save_train_metadata(epoch):
    mp = os.path.join(FOLDERPATH, "train_metadata")
    metadata = {"base_epoch": epoch}
    pickle.dump(metadata, open(mp, "wb"))

def get_dataloader(dataset, bs):
    vsplit = 0.2 #Ideally have this as an argument

    if vsplit == 0.0:
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
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

def loss_pass(net, loss_func, loader, epoch, optim, op='train'):
    print("{op} epoch: {epoch}".format(op=op, epoch=epoch)) 
    t0 = time.time()
    total_epoch_loss = 0
    for i, input_dict in enumerate(loader):
        ts_imgbatch, ts_anglebatch = input_dict.get("img"), input_dict.get("angle")
        ts_imgbatch, ts_anglebatch = ts_imgbatch.to(device), ts_anglebatch.to(device)

        #Classic Training Loop
        optim.zero_grad()
        ts_anglepred = net(ts_imgbatch)
        ts_loss = loss_func(ts_anglepred, ts_anglebatch)
        if op=='train':
            ts_loss.backward()
            optim.step()
        print("loss:{}".format(ts_loss.item())) 
        total_epoch_loss += ts_loss.item()
        if i % 20 == 0:
            #do some interesting visualization of results here
            pass
    print("FINISHED {op} EPOCH{epoch}".format(op=op, epoch=epoch))
    print("----{now} seconds----".format(now=time.time()-t0, op=op, epoch=epoch))            
    return total_epoch_loss

seed_env()

# 1: Load Dataset, split into train & val
dset = SteerDataset(FOLDERPATH)
train_dataloader, valid_dataloader = get_dataloader(dset, 32)
d = dset[0]

# 2: Get Model, Optimizer, Loss Function & Num Epochs
net = NVIDIA_ConvNet().to(device)
optim = torch.optim.Adam(net.parameters())
loss_func = torch.nn.MSELoss()
num_epochs = 50

# 3: TRAIN: Main Training Loop over epochs
metadata = load_train_metadata()
base_epoch = metadata["base_epoch"]
print(f"STARTING FROM EPOCH: {base_epoch}")

best_train_loss = float('inf')
best_valid_loss = float('inf')
for epoch in range(num_epochs):
    print(f"Starting Epoch:{epoch}")
    train_epoch_loss = loss_pass(net, loss_func, train_dataloader, epoch, optim, op='train')
    print("----------------EPOCH{}STATS:".format(epoch))
    print("TRAIN LOSS:{}".format(train_epoch_loss))
    if best_train_loss > train_epoch_loss:
        best_train_loss = train_epoch_loss
        torch.save(net.state_dict(), "sim_net")
    train_writer.add_scalar("Loss", train_epoch_loss, base_epoch+epoch)