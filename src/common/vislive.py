from nnet.Metric_Visualizer import Metric_Visualizer
from common.Trainer import Trainer
import os, pdb, glob

vis = Metric_Visualizer()
trainer = Trainer()
exp_path = trainer.get_exp_path()
listy = os.listdir(os.path.join(exp_path, 'data'))
for i in range(len(listy)):
    pkl_name = 'batch' + str(i)
    vis.vid_from_pklpath(os.path.join(exp_path, 'data', pkl_name), 0, 0, show_steer=True, units='rad', live=True, wk=0)