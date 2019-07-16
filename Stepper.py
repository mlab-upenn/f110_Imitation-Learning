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
        jsonfile = json.load(open("steps.json"))

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
    
    def default_vis(self, curr_step):
        #visualize data in tensorboard
        for i, folder in enumerate(self.dlist):
            self.visualizer.standard_log(self.sess_path, folder, self.curr_step_idx, global_step=i, units=curr_step.get("units", 'rad'))
            
    def exec_init(self, curr_step):
        """
        Initializes self.sess_path with filtered data
        curr_step: dict, as in steps.json
        """
        assert(self.curr_step_idx == 0 and self.dlist is None), "Step Error: init instruction can only be called once at the start" 
        
        #verify raw data & dlist
        self.dlist = curr_step["dlist"] 
        raw_datadir= os.path.join(self.params_dict["abs_path"], self.params_dict["raw_data"])
        self.B_VER(raw_datadir, self.dlist)
        print(f"PASSED B_VER FOR STEP {self.curr_step_idx}")
        dest_datadir = self.sess_path

        #move & filter each folder
        filter_funclist = [{"F":"filterBadData", "args":[]}]
        new_dlist = []
        for folder in self.dlist:
            new_folder = self.data_utils.MOVE(raw_datadir, folder, dest_datadir, flist=filter_funclist, preview=self.params_dict["preview"])
            new_dlist.append(new_folder)
        
        #visualize data in tensorboard
        self.dlist = new_dlist
        self.default_vis(curr_step)
    
    def exec_preprocess(self, curr_step):
        """
        Preprocesses data as per curr_step in steps.json
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
            new_folder = self.data_utils.MOVE(raw_datadir, folder, dest_datadir, flist=flist, preview=self.params_dict["preview"])
            new_dlist.append(new_folder)
        self.dlist = new_dlist
        self.default_vis(curr_step)

    def exec_augment(self, curr_step):
        """
        Augment data as per curr_step in steps.json
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
            new_folder = self.data_utils.MOVE(raw_datadir, folder, dest_datadir, flist=flist, preview=self.params_dict["preview"], op='aug')
            new_dlist.append(new_folder)
        self.dlist = new_dlist
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
            self.curr_step += 1
        
        elif insn_type =="combine":
            pass

        elif insn_type == "train":
            pass

    def augment(self, sess_loc, dlist, auglist):
        new_dlist = []
        for i, folder in enumerate(dlist):
            print("AUGMENTATION:", folder)
            #get list of transforms & call augment_dataset
            sourcepath = os.path.join(self.params_dict["abs_path"], sess_loc, folder)
            tf_list = auglist[i]
            self.dutils.augment_folder(sourcepath, tf_list)

    def preprocess(self, sess_loc, new_sess_loc, dlist, funclist):
        new_dlist = []
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

            #update dlist
            new_dlist.append(new_folder)
            
        return new_dlist
    
s = Stepper()
s.step()
s.step()
s.step()
s.writer.close()