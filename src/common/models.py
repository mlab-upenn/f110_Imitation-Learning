import os, torch, pdb
import torch.nn as nn
import numpy as np 
import torchvision

class NVIDIA_ConvNet(nn.Module):
    """
    Similar architecture to ConvNet used by Nvidia in Bojarski et al. (https://arxiv.org/pdf/1604.07316.pdf) with modified Fully Connected layers size 
    """
    def __init__(self, args_dict={"fc_shape":48576}):
        super(NVIDIA_ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,24,5,stride=2), nn.ELU(),
            nn.Conv2d(24,36,5,stride=2), nn.ELU(),
            nn.Conv2d(36,48,5,stride=2),nn.ELU(),
            nn.Conv2d(48,64,3,stride=1),nn.ELU(), 
            nn.Conv2d(64,64,3,stride=1),nn.ELU()
            # nn.Dropout(p=0.5)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(args_dict["fc_shape"], 500), 
            nn.Linear(500, 100), 
            nn.Linear(100, 50),
            nn.Linear(50, 10), 
            nn.Linear(10, 1)
        )

    def forward(self, ts_img):
        """
        out_dict: dict w/ "angle" 1-Tensor 
        """
        out = self.conv(ts_img)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    def only_conv(self, ts_img):
        """
        Does a conv pass and returns resulting tensor
        """
        out = self.conv(ts_img)
        return out

class KAHN_net(nn.Module):
    """
    Follows architecture similar to Kahn et al. in https://arxiv.org/pdf/1709.10489.pdf
    """
    def __init__(self, args_dict={"fc_shape":10560}):
        super(KAHN_net, self).__init__()
        pass