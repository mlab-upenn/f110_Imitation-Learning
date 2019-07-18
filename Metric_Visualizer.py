import os, cv2, math, sys, json, torch, pdb, random
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from Data_Utils import Data_Utils
import pandas as pd 
from tabulate import tabulate
from steps import session
from tensorboardX import SummaryWriter

class Metric_Visualizer(object):
    """
    Visualize metrics in Tensorboard
    """
    def __init__(self, sess_path=None, writer=None):
        """
        sess_path: current working dir of this session
        writer: Tensorboard SummmaryWriter
        """
        self.gvis = lambda param: session["visualizer"].get(param)
        self.fixangle = lambda angle, units: angle if units == 'rad' else angle * math.pi/180.0
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
        scalar = float(scalar) if float(scalar) != -0.0 else 0.0

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

    def vis_framelist(self, labelname, framelist, angle_list, global_step=0, show_steer=False, vel_list = None, predangle_list=None, timestamp_list=None):
        """
        Visualize a list of frames (cv2) w/ angles (radians, floats)
        """
        vislist = []
        tryget = lambda arr, index, default: default if arr is None else arr[index]
        for i, frame in enumerate(framelist):
            curr_timestamp = tryget(timestamp_list, i, 0)
            curr_predangle = tryget(predangle_list, i, None)
            curr_vel = tryget(vel_list, i, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.vis_frame(frame, angle_list[i], curr_vel, curr_timestamp, curr_predangle, show_steer=True)
            frame = cv2.bitwise_not(frame)
            vislist.append(frame)
        self.writer.add_images(labelname, vislist, global_step=global_step, dataformats='HWC')
    
    def framelist_from_path(self, dpath, stepname, idx, show_steer=False, units='rad'):
        """
        Send a list of frames to Tensorboard from a path
        """
        interesting_idxs = self.data_utils.get_interesting_idxs(dpath, 5)
        df = self.data_utils.get_df(dpath)
        datalist = list(map(lambda x: self.data_utils.df_data_fromidx(dpath, df, x), interesting_idxs)) #list of image, row pairs
        framelist = list(map(lambda x: x[0], datalist))
        rowlist = list(map(lambda x:x[1], datalist))
        splitrow = lambda idx: list(map(lambda x:x[idx], rowlist))
        angle_list = splitrow(1)
        angle_list = list(map(lambda x: self.fixangle(x, units), angle_list))
        vel_list = splitrow(2)
        timestamp_list = splitrow(3)
        self.vis_framelist(stepname, framelist, angle_list, global_step=idx, show_steer=show_steer, vel_list=vel_list, timestamp_list=timestamp_list)
        
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
                angle = self.fixangle(angle, units)
                framepath = os.path.join(dpath, img_name)
                frame = cv2.imread(framepath)
                self.vis_frame(frame, angle, speed, timestamp, show_steer=show_steer)
                framebuffer.append(frame.copy())

        self.writer.add_video(stepname, framebuffer, fps=10, global_step=idx, as_np_framebuffer=True)

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
        
    def visualize_batch(self,ts_imgbatch, ts_anglebatch, ts_predanglebatch, global_step=0):
        self.writer.add_images("Sample Batch", ts_imgbatch[:5], global_step=global_step)
        
    def dict_to_table(self, my_dict):
        """
        Converts a python dict into a table
        TODO: Consider the numerous edge cases (what if dict is already array)
        """
        tabulate_dict = {}
        for key in my_dict:
            tabulate_dict[key] = [str(my_dict[key])]
        text = tabulate(tabulate_dict, headers="keys", tablefmt="github")
        return text

    def log_training(self, config, train_id, best_train_loss, best_valid_loss):
        final_dict = config
        final_dict["train_id"] = train_id
        final_dict["t_loss"] = best_train_loss
        final_dict["v_loss"] = best_valid_loss
        text = self.dict_to_table(final_dict)
        self.writer.add_text('Train Summary', text, global_step=0)

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
        if self.gvis("vis_type") == 'video':
            self.vid_from_path(dpath, labelname, global_step, show_steer=True, units=units)
        elif self.gvis("vis_type") == 'framelist':
            self.framelist_from_path(dpath, labelname, global_step, show_steer=True, units=units)
        self.text_table(dpath, labelname, foldername=folder, angle_unit=units, global_step=global_step)
