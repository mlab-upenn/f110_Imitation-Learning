from __future__ import print_function
import airsim
import cv2, sys, os, random, pickle, time
from f110_gym.sim_f110_core import SIM_f110Env
import numpy as np

#Common Imports
from common.datasets import SteerDataset
from common.utils import cart_to_polar, vis_roslidar, polar_to_rosformat
from common.models import NVIDIA_ConvNet
from oracles.FGM import FGM

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
__author__ = 'dhruv karthik <dhruvkar@seas.upenn.edu>'

RENDER = False
FOLDERPATH = './sim_train'
seed_env()
num_saves = 0

def seed_env():
    seed = 6582
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 

def save_data(obs, action):
    global num_saves
    if(num_saves == 0 and not os.path.exists(FOLDERPATH)):
        os.mkdir(FOLDERPATH)
    pkl_dict = {"obs":obs, "action":action}
    filename = f"{FOLDERPATH}/sim_{num_saves}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(pkl_dict, f)
    num_saves+=1


def generate_oracle_data(net):
    print(f"GENERATING ORACLE DATA")
    env = SIM_f110Env()
    angle_min, angle_incr = env.sensor_info.get("angle_min"), env.sensor_info.get("angle_incr")
    fgm = FGM(angle_min, angle_incr)

    obs = env.reset()
    done = False
    for i in range(50):
        while not done:
            cv_img = obs["img"][0]
            lidar = obs["lidar"]
            lidar = lidar[..., 0:2]
            if RENDER:
                print(cv_img.shape)
                cv2.imshow('FrontCamera', cv_img)
                env.render_lidar2D(lidar)

            # Pass through Net
            ts_img = torch.from_numpy(cv_img).permute(2, 0, 1).float()
            ts_angle = net(ts_img[None])

            # Take Action WITH Neural Network
            action = {"angle": ts_angle.item() * np.pi/180.0, "speed":0.3}
            next_obs, _, done, _ = env.step(action)

            # Convert xyz lidar data to ROS LaserScan message format for FGM
            ranges, theta = cart_to_polar(lidar)
            ranges = polar_to_rosformat(angle_min, -1.0 * angle_min, angle_incr, theta, ranges)

            # Use FGM to get action, save it (relabel via expert policy)
            action = {"angle":fgm.act(ranges), "speed":0.3} 
            save_data(obs, action)
            obs = next_obs

            if cv2.waitKey(3) & 0xFF == ord('q'):
                break
            if done:
                obs = env.reset()  

def save_train_metadata(epoch):
    mp = "train_metadata"
    metadata = {"base_epoch": epoch}
    pickle.dump(metadata, open(mp, "wb"))

def load_train_metadata():
    mp = "train_metadata"
    if os.path.exists(mp):
        metadata = pickle.load(open(mp, "rb"))
    else:
        metadata = {"base_epoch": 0}
    return metadata

def save_train_metadata(epoch):
    mp = "train_metadata"
    metadata = {"base_epoch": epoch}
    pickle.dump(metadata, open(mp, "wb"))

def get_dataloader(dataset, bs):
    vsplit = 0.1 #Ideally have this as an argument

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


def TRAIN(net, optim, loss_func, num_epochs):
    dset = SteerDataset(FOLDERPATH)
    train_dataloader, _ = get_dataloader(dset, 32)

    # Main Training Loop over epochs
    metadata = load_train_metadata()
    base_epoch = metadata["base_epoch"]
    print(f"STARTING FROM EPOCH: {base_epoch}")
    for epoch in range(num_epochs):
        print(f"Starting Epoch:{epoch}")
        train_epoch_loss = loss_pass(net, loss_func, train_dataloader, epoch, optim, op='train')
        print("----------------EPOCH{}STATS:".format(epoch))
        print("TRAIN LOSS:{}".format(train_epoch_loss))
        if best_train_loss > train_epoch_loss:
            best_train_loss = train_epoch_loss
            torch.save(net.state_dict(), "train_sim_net")
        train_writer.add_scalar("Loss", train_epoch_loss, base_epoch+epoch)
    save_train_metadata(epoch)

def main():
    #1: Load Warmup Net
    net = NVIDIA_ConvNet().cuda()
    net.load_state_dict(torch.load('train_sim_net'))
    num_saves = len(os.listdir(FOLDERPATH))

    #2: Get Model, Optimizer, Loss Function & Num Epochs
    optim = torch.optim.Adam(net.parameters())
    loss_func = torch.nn.MSELoss()
    num_epochs = 50

    idx = 0
    while True:
        if idx % 2 == 0: #Generate Data with currently trained network
            generate_oracle_data(net)
        else: #TRAIN for about 50 epochs
            TRAIN(net, optim, loss_func, num_epochs)
        idx+=1

if __name__ == '__main__':
    main()