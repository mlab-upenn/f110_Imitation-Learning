import os, torch, logging, random, pdb
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import NVIDIA_ConvNet
from tensorboardX import SummaryWriter
from data_utils import Data_Utils
device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 

def loss_pass(net, loss, loader, epoch, optimizer, train=True):
    """
    Performs loss pass over all inputs in this epoch
    """
    if not train:
        torch.set_grad_enabled(False)
        print("STARTING VALIDATION EPOCH")
    else:
        print("STARTING TRAINING EPOCH")
    total_epoch_loss = 0
    for i, data in enumerate(loader):
        img_tensor, angle_true = data
        img_tensor, angle_true = img_tensor.to(device), angle_true.to(device)
        #Classic train loop
        optimizer.zero_grad()
        angle_pred = net(img_tensor)
        loss_tensor = loss(angle_pred, angle_true)
        if train:
            loss_tensor.backward()
            optimizer.step()
        print("loss:{}".format(loss_tensor.item()))
        total_epoch_loss += loss_tensor.item()
    if not train:
        torch.set_grad_enabled(True)
        print("ENDED VALIDATION EPOCH")
    else:
        print("ENDED TRAINING EPOCH")
    return total_epoch_loss

def train(net, num_epochs, optimizer, loss_func, train_loader, valid_loader):
    writer = SummaryWriter("__tblogs")
    for epoch in range(0, num_epochs):
        print("Starting epoch: {}".format(epoch))
        train_epoch_loss = loss_pass(net, loss_func, train_loader, epoch, optimizer, train=True)
        valid_epoch_loss = loss_pass(net, loss_func, valid_loader, epoch, optimizer, train=False)
        print("----------------EPOCH{}STATS:".format(epoch))
        print("TRAIN LOSS:{}".format(train_epoch_loss))
        print("VALIDATION LOSS:{}".format(valid_epoch_loss))
        print("----------------------------")
        writer.add_scalar('Train Loss', train_epoch_loss)
        writer.add_scalar('Valid Loss', valid_epoch_loss)
    writer.close()
def main():
    #Set random seeds
    seed = 6582
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #Choose Net + Parameters for training
    net = NVIDIA_ConvNet().to(device)
    loss_func = nn.functional.mse_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    num_epochs = 1
    batch_size = 64
    
    #Make Dataloaders
    dutils = Data_Utils()
    train_dataloader, valid_dataloader = dutils.get_dataloaders(batch_size)
    #TRAIN!
    train(net, num_epochs, optimizer, loss_func, train_dataloader, valid_dataloader)

if __name__ == "__main__":
    main()