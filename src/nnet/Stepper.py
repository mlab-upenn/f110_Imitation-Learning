from __future__ import print_function
import os, json, glob, pdb, cv2
try:
    import pandas as pd
    from tensorboardX import SummaryWriter
except ImportError:
    print("WARNING:Cannot use STEPS fully without these packages")
from nnet.Metric_Visualizer import Metric_Visualizer
from nnet.Data_Utils import Data_Utils
import steps

class Stepper(object):
    """
    Parses steps.session and executes ins. on data folders accordingly to get it into a final state
    """
    def __init__(self):
        jsonfile = steps.session
        #The state of the session 
        self.params_dict = jsonfile["params"]
        self.steplist = jsonfile["steps"]
        self.curr_step_idx = 0
        self.dlist = None #running list of folders to operate on
        
        #Session ID & Logger
        self.sess_id, self.sess_path = self._create_session_data(self.params_dict["abs_path"], self.params_dict["sess_root"])
        self.writer = self._create_writer(self.sess_path, 'logs', comment=self.params_dict["comment"])

        #Periphery Stuff for data moving and visualization
        self.data_utils = Data_Utils()
        self.visualizer = Metric_Visualizer(self.sess_path, self.writer)

    def _create_session_data(self, abs_path, sess_root):
        """
        Returns an int representing the unique session ID & makes a folder for the session, along with a string representing the current working directory of the session
        sess_root: root folder containing all the sessions
        """
        sess_path = os.path.join(abs_path, sess_root)
        if not os.path.exists(sess_path):
            os.makedirs(sess_path)
        sess_id = len(os.listdir(sess_path))
        sess_path = os.path.join(sess_path, str(sess_id))
        print("SESSION PATH:", sess_path)
        print("SESSION ID:", sess_id)    
        return sess_id, sess_path

    def _create_writer(self, sess_path, log_foldername, comment=''):
        """
        Retruns a summary writer in the right place
        sess_path: current working dir of session
        """
        logdir = os.path.join(sess_path, log_foldername)
        print("LOGDIR:", logdir)
        writer = SummaryWriter(logdir=logdir, comment=comment)
        return writer
    
    def B_VER(self, ver_path, dlist):
        """
        (B)asic (VER)ification of folder structure in dlist
        ver_path is generally sess_path, but it could also be some other folder
        similarly, dlist is generally self.dlist, but it could also be someother list of folders
        """ 
        for folder in dlist:
            dpath = os.path.join(ver_path, folder)
            #Ensure folder exists
            assert(os.path.exists(dpath)),"B_VER Error: Folder {folder} cannot be found in {ver_path}".format(folder=folder, ver_path=ver_path)
            
            csvpath  = os.path.join(dpath, 'data.csv')
            #Check for data.csv
            assert(os.path.isfile(csvpath)),"B_VER Error: data.csv could not be found in {dpath}".format(dpath=dpath)

            #Verify 4 columns in csv
            df = pd.read_csv(csvpath)
            assert(len(df.columns) == 4),"B_VER Error: Need four columns in data.csv in {csvpath}".format(csvpath=csvpath)

            #The number of jpg files in this directory should equal num rows
            num_jpgs = len(glob.glob1(dpath, "*.jpg"))
            assert(len(df) == num_jpgs),"B_VER Error: num_jpgs in {dpath} must = num_rows in data.csv".format(dpath=dpath)

            #Index the first 20 and last 20, and ensure that those images exist
            for i in range(20):
                img_name = df.iloc[i, 0]
                framepath = os.path.join(dpath, img_name)
                frame = cv2.imread(framepath)
                assert(os.path.isfile(framepath)), "B_VER Error: frame {framepath} is not a path, but is in the csv".format(framepath=framepath)

            for i in range(20):
                img_name = df.iloc[-i, 0]
                framepath = os.path.join(dpath, img_name)
                frame = cv2.imread(framepath)
                assert(os.path.isfile(framepath)), "B_VER Error: frame {framepath} is not a path, but is in the csv".format(framepath=framepath)
    
    def default_vis(self, curr_step):
        #visualize data in tensorboard
        for i, folder in enumerate(self.dlist):
            self.visualizer.standard_log(self.sess_path, folder, self.curr_step_idx, global_step=i, units=curr_step.get("units", 'rad'))
            
    def exec_init(self, curr_step):
        """
        Initializes self.sess_path with filtered data
        curr_step: dict, as in steps.session
        """
        assert(self.curr_step_idx == 0 and self.dlist is None), "Step Error: init instruction can only be called once at the start" 
        
        #verify raw data & dlist
        self.dlist = curr_step["dlist"] 
        raw_datadir= os.path.join(self.params_dict["abs_path"], self.params_dict["raw_data"])
        self.B_VER(raw_datadir, self.dlist)
        print("PASSED B_VER FOR STEP {curr_step_idx}".format(curr_step_idx=self.curr_step_idx))
        dest_datadir = self.sess_path
        filter_funclist = curr_step["funclist"]
        #move & filter each folder
        new_dlist = []
        for i, folder in enumerate(self.dlist):
            new_folder = self.data_utils.MOVE(raw_datadir, folder, dest_datadir, flist=filter_funclist[i], preview=self.params_dict["preview"])
            new_dlist.append(new_folder)
        
        #visualize data in tensorboard
        self.dlist = new_dlist
        self.default_vis(curr_step)
    
    def exec_preprocess(self, curr_step):
        """
        Preprocesses data as per curr_step in steps.session
        """
        assert(self.curr_step_idx > 0 and self.dlist is not None), "Step Error: Must call init before preprocess"            

        #verify raw data & dlist
        self.B_VER(self.sess_path, self.dlist)

        #move & preprocess each folder
        funclist = curr_step["funclist"]
        raw_datadir = self.sess_path
        dest_datadir = self.sess_path
        new_dlist = []
        for i, folder in enumerate(self.dlist):
            flist = funclist[i]
            new_folder = self.data_utils.MOVE(raw_datadir, folder, dest_datadir, flist=flist, preview=False)
            new_dlist.append(new_folder)
        self.dlist = new_dlist
        self.default_vis(curr_step)

    def exec_augment(self, curr_step):
        """
        Augment data as per curr_step in steps.json
        """
        assert(self.curr_step_idx > 0 and self.dlist is not None), "Step Error: Must call init before augment"            

        #verify raw data & dlist
        self.B_VER(self.sess_path, self.dlist)

        #move & preprocess each folder
        funclist = curr_step["funclist"]
        raw_datadir = self.sess_path
        dest_datadir = self.sess_path
        for i, folder in enumerate(self.dlist):
            flist = funclist[i]
            self.data_utils.MOVE(raw_datadir, folder, dest_datadir, flist=flist, preview=False, op='aug')
        self.default_vis(curr_step)    

    def exec_combine(self, curr_step):
        """
        Combines files currently in dlist into a specified folder
        updates dlist to only have this folder
        """
        assert(self.curr_step_idx > 0 and self.dlist is not None), "Step Error: Must call init before combine"            
        
        #verify raw data & dlist
        self.B_VER(self.sess_path, self.dlist)

        #move & preprocess each folder
        raw_datadir = self.sess_path
        dest_datadir = self.sess_path
        foldername = curr_step.get("foldername", "main")
        for i, folder in enumerate(self.dlist):
            self.data_utils.MOVE(raw_datadir, folder, dest_datadir, flist=[], preview=False, op='combine_choosename|' + foldername)
        self.dlist = [foldername]
        self.default_vis(curr_step)

    def exec_attention(self,curr_step):
        """
            Function to use a detection network to predict bounding boxes around objects of interest.
            The bounding boxes will then be highlighted in the image so that the network pays more attention
            to these objects during training
        """    

        assert(self.curr_step_idx > 0 and self.dlist is not None), "Step Error: Must call init before combine"            
        
        detectType = curr_step["detectionNetwork"]
        paramsFile = curr_step["paramsFile"]
        funclist = curr_step["funclist"]

        #verify raw data & dlist
        # self.B_VER(self.sess_path, self.dlist)
        raw_datadir = self.sess_path
        dest_datadir = self.sess_path  

        model_dict = {"detectType": detectType, "paramsFile" : paramsFile}    

        for i, folder in enumerate(self.dlist):
            flist = funclist[i]
            self.data_utils.DETECT(raw_datadir, folder, dest_datadir, model_dict, flist=[], preview=False)
        self.default_vis(curr_step)
            
    def step(self):
        """
        Executes the instruction for the curr_step
        """
        curr_step = self.steplist[self.curr_step_idx] #dictionary
        insn_type = curr_step["type"]

        #case over insn_types
        if insn_type == "init":
            self.exec_init(curr_step)
            self.curr_step_idx += 1

        elif insn_type == "preprocess":
            self.exec_preprocess(curr_step)
            self.curr_step_idx += 1

        elif insn_type == "augment":
            self.exec_augment(curr_step)
            self.curr_step_idx += 1
        
        elif insn_type =="combine":
            self.exec_combine(curr_step)
            self.curr_step_idx += 1

        elif insn_type =="attention":
            self.exec_attention(curr_step)
            self.curr_step_idx += 1
