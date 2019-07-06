import os, torch, logging, random, pdb
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import NVIDIA_ConvNet
from tensorboardX import SummaryWriter
from data_utils import Data_Utils
device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 
logger = logging.getLogger()

def loss_pass(net, loss, train_loader, epoch, optimizer, train=True):
    """
    Performs a pass over all inputs in this epoch
    """
    if not train:
        torch.set_grad_enabled(False)
        
        
    if not train:
        torch.set_grad_enabled(True)

def train(net, num_epochs, optimizer, loss_func, train_loader, valid_loader):
    for epoch in range(0, num_epochs):
        logger.info("Starting epoch: {}".format(epoch))
        train_loss = loss_pass(net, loss_func, train_loader, epoch, optimizer, train=True)

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
    num_epochs = 1000
    batch_size = 64
    
    #Make Dataloaders
    dutils = Data_Utils()
    train_dataloader, valid_dataloader = dutils.get_dataloaders(batch_size)

    train(net, num_epochs, optimizer, loss_func, train_dataloader, valid_dataloader)

if __name__ == "__main__":
    main()