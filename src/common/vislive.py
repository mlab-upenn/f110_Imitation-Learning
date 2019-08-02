from nnet.Metric_Visualizer import Metric_Visualizer
from common.Trainer import Trainer
import os

vis = Metric_Visualizer()
trainer = Trainer()
exp_path = trainer.get_exp_path()
for pkl_name in os.listdir(os.path.join(exp_path, 'data')):
    print(pkl_name)
    vis.vid_from_pklpath(os.path.join(exp_path, 'data', pkl_name), 0, 0, show_steer=True, units='rad', live=True)
