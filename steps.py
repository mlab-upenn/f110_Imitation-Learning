from functools import partial
from models import *
from func_utils import *

p = lambda func, args: partial(func, args)
session = {
    "params":
    {
        "abs_path":"/home/dhruvkar/datasets/avfone",
        "raw_data":"raw_data",
        "sess_root":"runs",
        "comment":"Rescaling Images",
        "preview":True
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
                    p(radOffset, [0.15]),
                    p((rad2deg), []),
                    p((rescaleImg), [0.5])
                ],

                [
                    p(rot90, ["clockwise"]),
                    p(cropVertical, [200, 400]),
                    p(radOffset, [0.15]),
                    p((rad2deg), []),
                    p((rescaleImg), [0.5])
                ],

                [
                    p(rot90, ["clockwise"]),
                    p(cropVertical, [200, 400]),
                    p((rad2deg), []),
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
    ]
}