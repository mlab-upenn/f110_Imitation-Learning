import os, cv2, math, sys
import pandas as pd 
abs_path = "/home/dhruvkar/datasets/f110_dataset"

class SteerVisualizer(object):
    """
    Set of methods that allow you to visualize the F110's Driving Behaviour
    """
    def __init__(self):
        self.text_count = 0
        self.frame_name = 'f110 Steer Visualizer'
        cv2.namedWindow(self.frame_name, cv2.WINDOW_NORMAL)

    def log_textdata(self, frame, scalar, label=""):
        """
        Function that logs text to the top left, automatically positioning it as you log more text
        """
        self.text_count += 1

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
        cv2.circle(frame, (int(x), int(y)), size, (32, 165, 218), -1)

    def vis_frame(self, frame, angle, speed, pred):
        """
        Vis an image w/ text info log + steering_angle_graphic & display it 
        """
        #reset text_count
        self.text_count = 0

        #log text data in the top left
        self.log_textdata(frame, angle, label="angle:")
        self.log_textdata(frame, speed, label="speed:")
        if pred is not None:
            self.log_textdata(frame, pred, label="pred_angle")

        #BIG steering graphic
        cx, cy, r = 320, 520, 80
        cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)

        #SMALL steering point graphic
        self.vis_steer_point(frame, angle, cx, cy, r, size=10, color=(218, 165, 32))
        if pred is not None:
            self.vis_steer_point(frame, pred, cx, cy, r, size=5, color=(0, 0, 128))

        #display image
        cv2.imshow(self.frame_name, frame)
        cv2.waitKey(10)

    def vis_from_path(self, abs_path, csv_file_path, pred_angle=False):
        """
        Visualize a sequence of steering angles
        abs_path: abs path to folder containing dashcam images
        csv_file_path: abs path to csv file w/ format in "data_exploration.ipynb" 
        pred_angle: True if csv_file has final column with predicted steering angles
        """
        df = pd.read_csv(csv_file_path)
        num_rows = len(df)
        for i in range(num_rows):
            #Read info from dataframe 
            img_name, angle, speed, pred = df.iloc[i, 0], -1.0 * df.iloc[i, 1], -1.0 * df.iloc[i, 2], None
            if pred_angle:
                pred = df.iloc[i, 3]
            frame = cv2.imread(abs_path + '/' + img_name) 

            #visualize this frame
            self.vis_frame(frame, angle, speed, pred)

def main():
    csv_file_path = abs_path + "/data.csv"
    steer_vis = SteerVisualizer()
    steer_vis.vis_from_path(abs_path, csv_file_path)

if __name__ == '__main__':
    main()