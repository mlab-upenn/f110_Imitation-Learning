import os, cv2, math, sys, json, torch, pdb
import numpy as np
import matplotlib.pyplot as plt
import moviepy.editor as mpy
from data_utils import Data_Utils
import pandas as pd 
from tensorboardX import SummaryWriter

class Metric_Visualizer(object):
    """
    Visualize metrics in Tensorboard
    """
    def __init__(self, sess_id, writer):
        self.params_dict = json.load(open("steps.json"))["params"]
        
        #Create SummaryWriter for tensorboard that updates frequently
        self.logdir = os.path.join(self.params_dict["abs_path"], self.params_dict["sess"], str(sess_id), "logs")
        print("LOGDIR", self.logdir)
        self.writer = writer

    def vis_steer_point(self, frame, angle, cx, cy, r, size=10, color=(0, 0, 0)):
        """
        Tiny point on big steering graphic that shows the steering angle
        """
        x = (cx + r*math.cos(-1.0 * angle + math.pi/2))
        y = (cy - r*math.sin(-1.0 * angle + math.pi/2))
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
    
    def _convert_to_rads(self, angle, dpath, df):
        units = self._deg_or_rad(dpath, df)
        if units == 'deg':
            return angle * math.pi/180.0
        return angle
        
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
            h, w = frame.shape
            cx, cy, r = h, int(w/2), 80

            #SMALL steering point graphic (angle must be in radians)
            self.vis_steer_point(frame, angle, cx, cy, r, size=10, color=(32, 165, 218))
            if pred is not None:
                self.vis_steer_point(frame, pred, cx, cy, r, size=5, color=(0, 0, 0))

    def vid_from_path(self, dpath, stepname, idx):
        """
        Send annotated video to Tensorboard
        """
        framebuffer = []
        #get dataframe
        csvpath = os.path.join(dpath, "data.csv")
        df = pd.read_csv(csvpath)
        num_rows = int(0.1 * len(df)) #display about 10% of the frames 
        for i in range(num_rows):
            if i % 3 == 0:
                img_name, angle, speed, timestamp = df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3]
                #only convert angle before visualizing
                angle = self._convert_to_rads(angle, dpath, df)
                framepath = os.path.join(dpath, img_name)
                frame = cv2.imread(framepath)
                self.vis_frame(frame, angle, speed, timestamp)
                framebuffer.append(frame.copy())

        self.writer.add_video(stepname, framebuffer, global_step= idx, as_np_framebuffer=True)

    def _get_image_size(self, dpath, df):
        img_name_0 = df.iloc[0, 0]
        framepath = os.path.join(dpath, img_name_0)
        frame_0 = cv2.imread(framepath)
        h, w, c = frame_0.shape
        return h, w

    def _deg_or_rad(self, dpath, df):
        #crappy but effective way to figure out if i'm dealing with degrees or radians
        angle_column = df.iloc[:, 1].values
        max_angle = np.max(angle_column)
        if max_angle > 0.4:
            return 'deg'
        return 'rad'

    def plot_anglehist(self, dpath, tag, idx):
        csvpath = os.path.join(dpath, "data.csv")
        df = pd.read_csv(csvpath) 
        angle_column = df.iloc[:, 1].values
        num_bins = 20
        #save plot w/ matplotlib
        fig = plt.figure()
        plt.hist(angle_column, num_bins, color='green')
        self.writer.add_figure(tag, fig, global_step=idx)

    def log_tbtext(self, dpath, tag, idx, folder):
        csvpath = os.path.join(dpath, "data.csv")
        df = pd.read_csv(csvpath)
        h, w = self._get_image_size(dpath, df)
        angle_unit = self._deg_or_rad(dpath, df)
        text = f"Folder:{folder} ||| Shape:({h}, {w}) ||| AngleUnits:{angle_unit} ||| NumImages:{len(df)}"
        self.writer.add_text(tag, text, global_step=idx)
        
    def log_init(self, dlist, sess_loc):
        sess_path = os.path.join(self.params_dict["abs_path"], sess_loc)
        tag = f"Step-{0}"
        for i, folder in enumerate(dlist):
            dpath = os.path.join(sess_path, folder) #where jpgs & csv is
            self.vid_from_path(dpath, tag, i) 
            self.plot_anglehist(dpath, tag, i)
            self.log_tbtext(dpath, tag, i, folder) 