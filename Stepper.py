import os, json, glob
import pandas as pd
import Metric_Visualizer

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

        #Get Session ID
        sess_path = os.path.join(self.params_dict["abs_path"], self.params_dict["sess"])
        if not os.path.exists(sess_path):
            os.mkdirs(sess_path)
        self.sess_id = len(os.listdir(sess_path))
        print("SESSION PATH:", sess_path)
        print("SESSION ID:", sess_id)
        
        #kick off the steps
        self.step()
    
    def B_VER(self, ver_path, dlist):
        """
        (B)asic (VER)ification of folders in dlist
        """

        for folder in dlist:
            dpath = os.path.join(ver_path, folder)
            #Ensure folder exists
            assert(os.path.exists(dpath)) F"B_VER Error: Folder {folder} cannot be found in {ver_path}"
            
            csvpath  = os.path.join(dpath, 'data.csv')
            #Check for data.csv
            assert(os.path.isfile(csvpath)) F"B_VER Error: data.csv could not be found in {dpath}"

            #Verify 4 columns in csv
            df = pd.read_csv(csvpath)
            assert(len(df.columns) == 4) F"B_VER Error: Need four columns in data.csv in {csvpath}"

            #The number of jpg files in this directory should equal num rows
            num_jpgs = len(glob.glob1(dpath, "*.jpg"))
            assert(len(df) == num_jpgs) F"B_VER Error: num_jpgs in {dpath} must = num_rows in data.csv"

    def step(self):
        """
        Executes the instruction for the curr_step
        """
        curr_dict = self.steplist[self.curr_step]
        insn_type = curr_dict["type"]

        #case over insn_types
        if insn_type == "init":
            assert (self.curr_step == 0 and self.dlist is None), "Step Error: init instruction can only be called once at the start"

            dlist = curr_dict["dlist"]

            #data resides in raw data at the start of the session
            ver_path = os.path.join(self.params_dict["abs_path"], self.sess_loc)
            self.B_VER(ver_path, dlist)
            self.dlist = dlist
            
            #
            
       elif insn_type == "preprocess":
            pass

        elif insn_type == "augment":
            pass
        
        elif insn_type =="combine":
            pass

        elif insn_type == "train":
            pass