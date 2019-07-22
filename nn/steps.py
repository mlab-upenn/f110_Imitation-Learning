from functools import partial
from models import *
from func_utils import *
from datasets import *
import torch.nn as nn
import torch.optim as optim

p = lambda func, args: partial(func, args)
session = {
    "params":
    {
        "abs_path":"/home/dhruvkar/datasets/avfone",
        "raw_data":"raw_data",
        "sess_root":"runs",
        "comment":"Normally distributed w/ variance 1.2",
        "preview":True
    },
    "visualizer":
    {
        "vis_type":"video",
    },
    "steps":
    [
        {
            "type":"init",
            "units":"rad",
            "dlist":["lr1_left_folder", "lr1_right_folder", "f1_front_folder"],
            "funclist":
            [
                [p(filterBadData, [])], 
                [p(filterBadData, [])], 
                [p(filterBadData, [])]
            ]
        },
        {
            "type":"preprocess",
            "units":"deg",
            "funclist":
            [
                [
                    p(rot90, ["clockwise"]),
                    p(cropVertical, [200, 400]),
                    p(cannyEdge, [100, 200]),
                    p(radOffset, [0.15]),
                    p((rad2deg), []),
                    p((gaussianSamplingAngle), [1.2]),
                    p((rescaleImg), [0.5]),
                ],

                [
                    p(rot90, ["clockwise"]),
                    p(cropVertical, [200, 400]),
                    p(radOffset, [-0.15]),
                    p((rad2deg), []),
                    p((gaussianSamplingAngle), [1.2]),
                    p((rescaleImg), [0.5])
                ],

                [
                    p(rot90, ["clockwise"]),
                    p(cropVertical, [200, 400]),
                    p((rad2deg), []),
                    p((gaussianSamplingAngle), [1.2]),
                    p((rescaleImg), [0.5]) 
                ]
            ]
        },
        {
            "type":"augment",
            "units":"deg",
            "funclist":
            [
                [
                    p(flipNonZero, [])
                ],
                [
                    p(flipNonZero, [])
                ],
                [
                    p(flipNonZero, [])
                ]
            ]
        },
        {
            "type":"combine",
            "units":"deg",
            "foldername":"main"
        }
    ],
    "train":
    {
        "model":NVIDIA_ConvNet,
        "lr":1e-3,
        "loss_func":nn.functional.l1_loss,
        "optimizer":torch.optim.Adam,
        "num_epochs":5,
        "batch_size":4,
        "sess_id": 0,
        "foldername":"main",
        "vsplit":0.1,
        "dataset":SteerDataset
    },
}