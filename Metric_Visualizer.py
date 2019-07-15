import os, cv2, math, sys, json, torch, pdb
import moviepy.editor as mpy
from data_utils import Data_Utils
import pandas as pd 
from tensorboardX import SummaryWriter

class Metric_Visualizer(object):
    """
    Visualize metrics in Tensorboard
    """
    def __init__(self, step_id):
        self.params_dict = json.load(open("steps.json"))["params"]
        
        #Create SummaryWriter for tensorboard that updates frequently
        self.logdir = os.path.join(self.params_dict["abs_path"], self.params_dict["sess"], step_id, "runs")
        print("LOGDIR", self.logdir)
        self.writer = SummaryWriter(logdir=self.logdir)
    
    def vis_steer_point(self, frame, angle, cx, cy, r, size=10, color=(0, 0, 0)):
        """
        Tiny point on big steering graphic that shows the steering angle
        """
        x = (cx + r*math.cos(-1.0 * angle + math.pi/2))
        y = (cy - r*math.sin(-1.0 * angle + math.pi/2))
        cv2.circle(frame, (int(x), int(y)), size, color, -1)

    def vis_textdata(self, frame, angle, label, pos):
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
        self.vis_textdata(frame, angle, "angle:", 0)
        self.vis_textdata(frame, speed, "speed:", 1)
        self.vis_textdata(frame, timestamp, "time:", 2)
        if pred is not None:
            self.vis_textdata(frame, timestamp, "time:", 3)

        if show_steer:
            #Big Steering Graphic
            h, w = frame.shape
            cx, cy, r = h, int(w/2), 80

            #SMALL steering point graphic (angle must be in radians)
            self.vis_steer_point(frame, angle, cx, cy, r, size=10, color=(32, 165, 218))
            if pred is not None:
                self.vis_steer_point(frame, pred, cx, cy, r, size=5, color=(0, 0, 0))

    def vid_from_path(self, dpath, vid_name):
        """
        Generate an annotated video from dpath
        """
        vid_path = os.path.join(self.logdir, vid_name) #missing extension
        print("VID PATH", vid_path)

        framebuffer = []
        #get dataframe
        csvpath = os.path.join(dpath, "data.csv")
        df = pd.read_csv(csvpath)
        num_rows = len(df)
        for i in range(num_rows):
            if i % 20 == 0:
                img_name, angle, speed, timestamp = df.iloc[i, 0], df.iloc[i, 1], df.iloc[i, 2], df.iloc[i, 3]
                framepath = os.path.join(dpath, img_name)
                frame = cv2.imread(framepath)
                self.vis_frame(frame, angle, speed, timestamp)
                framebuffer.append(frame.copy())

        clip = mpy.ImageSequenceClip(framebuffer, fps=10)
        clip.write_gif('{}.gif'.format(vid_path), fps=10)

    def log_init(self, dlist, sess_loc):
        sess_path = os.path.join(self.params_dict["abs_path"], sess_loc)
        for folder in dlist:
            dpath = os.path.join(sess_path, folder) #where jpgs & csv is
            self.vid_from_path(dpath, F"{folder}_STEP_{0}")
            