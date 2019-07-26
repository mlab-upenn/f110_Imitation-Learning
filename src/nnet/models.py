import os, torch, pdb
import torch.nn as nn
import numpy as np 
import torchvision

class NVIDIA_ConvNet(nn.Module):
    """
    Similar architecture to ConvNet used by Nvidia in Bojarski et al. (https://arxiv.org/pdf/1604.07316.pdf) with modified Fully Connected layers size 
    """
    def __init__(self, args_dict={"fc_shape":7360}):
        super(NVIDIA_ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,24,5,stride=2), nn.ELU(),
            nn.Conv2d(24,36,5,stride=2), nn.ELU(),
            nn.Conv2d(36,48,5,stride=2),nn.ELU(),
            nn.Conv2d(48,64,3,stride=1),nn.ELU(), 
            nn.Conv2d(64,64,3,stride=1),nn.ELU(),nn.Dropout(p=0.5)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(args_dict["fc_shape"], 500), 
            nn.Linear(500, 100), 
            nn.Linear(100, 50),
            nn.Linear(50, 10), 
            nn.Linear(10, 1)
        )

    def forward(self, input_dict):
        """
        out_dict: dict w/ "angle" 1-Tensor 
        """
        x = input_dict["img"]
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out_dict = {"angle":out}
        return out_dict
    
    def only_conv(self, input_dict):
        """
        Does a conv pass and returns resulting tensor
        """
        ts_img = input_dict["img"]
        out = self.conv(ts_img)
        out_dict = {"img":out}
        return out_dict