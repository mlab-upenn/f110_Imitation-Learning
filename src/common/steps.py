from common.models import *
from common.datasets import *
from common.augs import *
from functools import partial
import torch
import torch.optim
import torch.nn as nn

p = lambda func, args: partial(func, args)
session = {
    "train":
    {
        "model":NVIDIA_ConvNet,
        "modelname":'deg_nvidia_model',
        "lr":1e-3,
        "loss_func":nn.functional.mse_loss,
        "optimizer":torch.optim.Adam,
        "num_epochs":500,
        "batch_size":64,
        "sess_id": 3,
        "vsplit":0.0,
        "dataset":SteerDataset_ONLINE, 
        "root":'/home/dhruvkar/datasets/avfone/', "transforms":[toDeg()], 
        "units":'deg' 
    },
    "online":
    {
        "model":NVIDIA_ConvNet,
        "modelname":'deg_nvidia_model',
        "lr":1e-3,
        "loss_func":nn.functional.mse_loss,
        "optimizer":torch.optim.Adam,
        "num_epochs":10,
        "batch_size":32,
        "sess_id": 3,
        "vsplit":0.0,
        "dataset":SteerDataset_ONLINE,
	    "root":'/home/dhruvkar/datasets/avfone/',
        "transforms":[toDeg()],
        "units":'deg'
    }
}