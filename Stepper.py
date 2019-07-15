import os, json, glob, pdb, cv2
import pandas as pd
from tensorboardX import SummaryWriter
from Metric_Visualizer import Metric_Visualizer
from Data_Utils import Data_Utils

class Stepper(object):
    """
    Parses steps.json and executes ins. on data folders accordingly
    """
    def __init__(self):
        self.params_dict = json.load(open("steps.json"))["params"]
        self.steplist = json.load(open("steps.json"))["steps"]
        self.curr_step = 0 #the next step we have to execute
        self.sess_loc = self.params_dict["raw_data"] #running folder of data for this session (start at raw data, update after preprocessing)
        self.dlist = None #running list of folders to operate on 
        self.vis = None #Metric Visualizer
        self.dutils = Data_Utils()

        #Get Session ID
        sess_path = os.path.join(self.params_dict["abs_path"], self.params_dict["sess"])
        if not os.path.exists(sess_path):
            os.makedirs(sess_path)
        self.sess_id = len(os.listdir(sess_path))
        print("SESSION PATH:", sess_path)
        print("SESSION ID:", self.sess_id)
        
        #Create SummaryWriter for tensorboard that updates frequently
        logdir = os.path.join(self.params_dict["abs_path"], self.params_dict["sess"], str(self.sess_id), "logs")
        print("LOGDIR", logdir)
        self.writer = SummaryWriter(logdir=logdir)
    
    def B_VER(self, ver_path, dlist):
        """
        (B)asic (VER)ification of folders in dlist
        """

        for folder in dlist:
            dpath = os.path.join(ver_path, folder)
            #Ensure folder exists
            assert(os.path.exists(dpath)),f"B_VER Error: Folder {folder} cannot be found in {ver_path}"
            
            csvpath  = os.path.join(dpath, 'data.csv')
            #Check for data.csv
            assert(os.path.isfile(csvpath)),f"B_VER Error: data.csv could not be found in {dpath}"

            #Verify 4 columns in csv
            df = pd.read_csv(csvpath)
            assert(len(df.columns) == 4),f"B_VER Error: Need four columns in data.csv in {csvpath}"

            #The number of jpg files in this directory should equal num rows
            num_jpgs = len(glob.glob1(dpath, "*.jpg"))
            assert(len(df) == num_jpgs),F"B_VER Error: num_jpgs in {dpath} must = num_rows in data.csv"

            #Index the first 20 and last 20, and ensure that those images exist
            for i in range(20):
                img_name = df.iloc[i, 0]
                framepath = os.path.join(dpath, img_name)
                frame = cv2.imread(framepath)
                assert(os.path.isfile(framepath)), F"B_VER Error: frame {framepath} is not a path, but is in the csv"

            for i in range(20):
                img_name = df.iloc[-i, 0]
                framepath = os.path.join(dpath, img_name)
                frame = cv2.imread(framepath)
                assert(os.path.isfile(framepath)), F"B_VER Error: frame {framepath} is not a path, but is in the csv"

    def step(self):
        """
        Executes the instruction for the curr_step
        """
        curr_dict = self.steplist[self.curr_step]
        insn_type = curr_dict["type"]

        #case over insn_types
        if insn_type == "init":
            assert(self.curr_step == 0 and self.dlist is None), "Step Error: init instruction can only be called once at the start"

            dlist = curr_dict["dlist"]

            #data resides in raw data at the start of the session
            ver_path = os.path.join(self.params_dict["abs_path"], self.sess_loc)
            print("VER_PATH:", ver_path)
            self.B_VER(ver_path, dlist)
            self.dlist = dlist
            print("PASSED B_VER")
            print("dlist", str(self.dlist))
        
            #Initialize Metrics Visualizer
            self.vis = Metric_Visualizer(self.sess_id, self.writer)
            self.vis.log_init(dlist, self.sess_loc)

            #increment step
            self.curr_step += 1

        elif insn_type == "preprocess":
            assert(self.curr_step > 0 and self.dlist is not None), "Step Error: Must call init before preprocess"

            ver_path = os.path.join(self.params_dict["abs_path"], self.sess_loc)
            print("VER_PATH:", ver_path) 
            self.B_VER(ver_path, self.dlist)
            print("PASSED B_VER")
            
            #Change session location to inside the current runs folder
            new_sess_loc = os.path.join(self.params_dict["abs_path"], self.params_dict["sess"], str(self.sess_id))
            print("NEW_SESS_LOC:", new_sess_loc)

            funclist = curr_dict["funclist"]
            new_dlist = self.preprocess(self.sess_loc, new_sess_loc, self.dlist, funclist)

        elif insn_type == "augment":
            pass
        
        elif insn_type =="combine":
            pass

        elif insn_type == "train":
            pass
    
    def preprocess(self, sess_loc, new_sess_loc, dlist, funclist):
        for i, folder in enumerate(dlist):
            print("PREPROCESSING:", folder)
            new_folder = "preprocess_" + str(len(os.listdir(new_sess_loc))) + folder
            sourcepath = os.path.join(self.params_dict["abs_path"], sess_loc, folder)
            destpath = os.path.join(self.params_dict["abs_path"], new_sess_loc, new_folder)
            print("SOURCEPATH:", sourcepath)
            print("DESTPATH:", destpath)
            
            #get list of transforms & call preprocess_dataset
            tf_list = funclist[i]
            self.dutils.preprocess_folder(sourcepath, destpath, tf_list)
        return dlist

s = Stepper()
s.step()
s.writer.close()