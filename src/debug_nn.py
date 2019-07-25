import torch
import torch.nn as nn
import numpy as np
import pickle 
from steps import session
from nnet.models import *
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

modelpath = session["online"]["modelpath"]
net = NVIDIA_ConvNet(args_dict={"fc_shape":7360})
net.load_state_dict(torch.load(modelpath))
net.to(device)
print("Sucess")
print(net)
