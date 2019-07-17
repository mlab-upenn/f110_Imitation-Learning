import os, cv2, math, sys, json, torch, pdb
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from Data_Utils import Data_Utils
import pandas as pd 
from tabulate import tabulate
from tensorboardX import SummaryWriter

class Metric_Visualizer(object):
    """
    Visualize metrics in Tensorboard
    """
    def __init__(self, sess_path, writer):
        """
        sess_path: current working dir of this session
        writer: Tensorboard SummmaryWriter
        """
        self.sess_path = sess_path
        self.writer = writer
        self.data_utils = Data_Utils()

    def vis_steer_point(self, frame, angle, cx, cy, r, size=10, color=(0, 0, 0)):
        """
        Tiny point on big steering graphic that shows the steering angle
        """
        x = (cx + r*math.cos(-1.0 * angle + math.pi/2.))
        y = (cy - r*math.sin(-1.0 * angle + math.pi/2.))
        cv2.circle(frame, (int(x), int(y)), size, color, -1)

    def vis_textdata(self, frame, scalar, label, pos):
        """
        Visualize text data. pos = vertical interval on the frame
        """
        #scalar = float(scalar) if float(scalar) != -0.0 else 0.0

        #TEXT PARAMETERS
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.5
        font_thickness = 1
        v_space = 20
        cx, cy = 20, pos * v_space
        #draw text on frame
        text = label + '%.3f'%(float(scalar))
        cv2.putText(frame, text, (cx, cy), font, font_size, color, font_thickness)
    
    def vis_frame(self, frame, angle, speed, timestamp, pred=None, show_steer=False):
        """
        Vis an image w/ text info log + steering_angle_graphic & display it 
        """
        #log text data in the top left
        self.vis_textdata(frame, angle, "angle:", 1)
        self.vis_textdata(frame, speed, "speed:", 2)
        self.vis_textdata(frame, timestamp, "time:", 3)
        if pred is not None:
            self.vis_textdata(frame, timestamp, "pred", 4)

        if show_steer:
            #Big Steering Graphic
            h, w, c = frame.shape
            cx, cy, r = int(w/2), h, int((h * 80)/480)
            cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
            big_steerpoint = int(math.ceil(r*10.0/80.0))
            angle_extra = angle * 2
            if pred is not None:
                pred_extra = pred * 2
            #SMALL steering point graphic (angle must be in radians)
            self.vis_steer_point(frame, angle_extra, cx, cy, r, size=big_steerpoint, color=(218, 165, 32))
            if pred is not None:
                self.vis_steer_point(frame, pred_extra, cx, cy, r, size=int(math.ceil(big_steerpoint/2.0)), color=(0, 0, 0))

    def vid_from_path(self, dpath, stepname, idx, show_steer=False, units='rad'):
        """
        Send annotated video to Tensorboard
        dpath: abs_path to data folder containing images & csv
        labelname: str name for the label
        global_step: global_step to record for slider functionality
        """
        framebuffer = []
        #get dataframe
        csvpath = os.path.join(dpath, "data.csv")
        df = pd.read_csv(csvpath)
        num_rows = int(0.1 * len(df)) #display about 10% of the frames 
        for i in range(num_rows):
            if i % 4 == 0:
                img_name, angle, speed, timestamp = df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3]

                #fix angle
                if units == 'deg':
                    angle = angle * math.pi/180

                framepath = os.path.join(dpath, img_name)
                frame = cv2.imread(framepath)
                self.vis_frame(frame, angle, speed, timestamp, show_steer=show_steer)
                framebuffer.append(frame.copy())

        self.writer.add_video(stepname, framebuffer, fps=10, global_step= idx, as_np_framebuffer=True)

    def plot_anglehist(self, dpath, tag, idx):
        csvpath = os.path.join(dpath, "data.csv")
        df = pd.read_csv(csvpath) 
        angle_column = df.iloc[:, 1].values
        num_bins = 100
        #save plot w/ matplotlib
        fig = plt.figure()
        plt.hist(angle_column, num_bins, color='green')
        self.writer.add_figure(tag, fig, global_step=idx)


    def text_table(self, dpath, labelname, foldername='', angle_unit='', global_step=0):
        df = self.data_utils.get_df(dpath)
        h, w = self.data_utils._get_image_size(dpath)
        text = f"Folder | Shape | Units | Num Images\n-----|-----|-----|-----\n{foldername}|({h}, {w})|{angle_unit}|{len(df)}"
        self.writer.add_text(labelname, text, global_step=global_step)
        

    def standard_log(self, datadir, folder, curr_step, global_step=0, units=''):
        """
        Log "Standard" things in Tensorboard
        datadir: abs_path of directory containing data folders
        folder: data folder name
        curr_step: progress in steps.json
        global_step: The "step" value to log into tensorboard (this allows for the cool slider functionality)
        units: 'rad' or 'deg'
        """
        labelname = f"STEP-{curr_step}"
        dpath = os.path.join(datadir, folder)
        self.plot_anglehist(dpath, labelname, global_step)
        self.vid_from_path(dpath, labelname, global_step, show_steer=True, units=units)
        self.text_table(dpath, labelname, foldername=folder, angle_unit=units, global_step=global_step)
