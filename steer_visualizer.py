import os, cv2, math, sys, json, torch
from data_utils import Data_Utils
from models import NVIDIA_ConvNet
import pandas as pd 

class SteerVisualizer(object):
    """
    Set of methods that allow you to visualize the F110's Driving Behaviour
    """
    def __init__(self):
        self.abs_path = json.load(open("params.txt"))["abs_path"]
        self.log_path = json.load(open("params.txt"))["log_path"]
        self.text_count = 0
        self.frame_name = 'f110 Steer Visualizer'
        cv2.namedWindow(self.frame_name, cv2.WINDOW_NORMAL)
        self.dutils = Data_Utils()
        self.flip_sign = 1.0

    def log_textdata(self, frame, scalar, label=""):
        """
        Function that logs text to the top left, automatically positioning it as you log more text
        """
        self.text_count += 1

        #fix scalar to not have -0.0 as a value
        scalar = float(scalar) if float(scalar) != -0.0 else 0.0

        #TEXT PARAMETERS
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.5
        font_thickness = 1
        v_space = 20
        cx, cy = 20, self.text_count * v_space
        #draw text on frame
        text = label + '%.3f'%(float(scalar))
        cv2.putText(frame, text, (cx, cy), font, font_size, color, font_thickness)

    def vis_steer_point(self, frame, angle, cx, cy, r, size=10, color=(0, 0, 0)):
        """
        Tiny point on big steering graphic that shows the steering angle
        """
        x = (cx + r*math.cos(-1.0 * angle + math.pi/2))
        y = (cy - r*math.sin(-1.0 * angle + math.pi/2))
        cv2.circle(frame, (int(x), int(y)), size, color, -1)

    def vis_frame(self, frame, angle, speed, pred):
        """
        Vis an image w/ text info log + steering_angle_graphic & display it 
        """
        #reset text_count
        self.text_count = 0

        #convert angle to degrees
        angle_rad = angle * math.pi/180.0

        #log text data in the top left
        self.log_textdata(frame, angle_rad, label="angle:")
        self.log_textdata(frame, speed, label="speed:")
        if pred is not None:
            self.log_textdata(frame, pred, label="pred_angle:")

        #BIG steering graphic
        cx, cy, r = 240, 640, 80
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)

        #SMALL steering point graphic
        self.vis_steer_point(frame, angle_rad, cx, cy, r, size=10, color=(32, 165, 218))
        if pred is not None:
            self.vis_steer_point(frame, pred, cx, cy, r, size=5, color=(0, 0, 0))

        #display image
        cv2.imshow(self.frame_name, frame)
        cv2.waitKey(100)

    def vis_from_path(self, foldername, pred_angle=False):
        """
        Visualize a sequence of steering angles from a csv file
        foldername: Folder to visualize from 
        pred_angle: True if csv_file has final column with predicted steering angles
        """
        csv_file_path = self.abs_path + foldername + '/data.csv'
        df = pd.read_csv(csv_file_path)
        num_rows = len(df)
        for i in range(num_rows):
            #Read info from dataframe 
            img_name, angle, speed, pred = df.iloc[i, 0], self.flip_sign * df.iloc[i, 1], -1.0 * df.iloc[i, 2], None
            if pred_angle:
                pred = df.iloc[i, 3]
            frame = cv2.imread(self.abs_path + foldername + '/' + img_name) 
            frame, angle = self.dutils.preprocess_img(frame, angle, use_for='vis')

            #visualize this frame
            self.vis_frame(frame, angle, speed, pred)

    def vis_from_model(self, model_type='train', idx=-1):
        """
        Visualize a sequence of steering angles from a csv file & pytorch model
        model_type: either 'train' or 'valid' to indicate which model to vis
        idx: pick which model from logs to vis
        """
        #stupid stuff to get the correct model from log file structure
        log_folders = os.listdir(self.log_path)
        log_folders.sort()
        foldername = log_folders[idx]
        model_path_name = self.log_path + '/' + foldername + '/best_' + model_type + '_model'

        #actually build the model
        net = NVIDIA_ConvNet()
        net.load_state_dict(torch.load(model_path_name))
        net.eval()

        #usual csv crap to get img_name, angle, etc.
        csv_file_path = self.abs_path + 'main/data.csv'
        df = pd.read_csv(csv_file_path)
        num_rows = len(df)
        for i in range(num_rows):
            img_name, angle, speed = df.iloc[i, 0], self.flip_sign * df.iloc[i, 1], -1.0 * df.iloc[i, 2]
            frame = cv2.imread(self.abs_path + '/' + img_name) 
            ts_frame, _ = self.dutils.preprocess_img(frame, label=None, use_for='vis')
            ts_frame = ts_frame[None]

            #use net to get predicted angle
            angle_pred = self.flip_sign * net(ts_frame).item()

            #visualize
            self.vis_frame(frame, angle, speed, angle_pred)

def main():
    steer_vis = SteerVisualizer()
    steer_vis.vis_from_path('front_folder')

if __name__ == '__main__':
    main()