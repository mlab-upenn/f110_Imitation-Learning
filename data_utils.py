import os, torch, cv2, csv, math, pdb, shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bashplotlib.histogram import plot_hist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import json
device = torch.device('cuda' if torch.cuda.is_available else 'cpu') 

class SteerDataset(Dataset):
    """
    Steer Dataset: Returns cropped image from dashcam + steering angle
    """
    def __init__(self, transforms=None):
        """
        transforms (callable, optional): Optional transforms applied on samples
        """
        super(SteerDataset, self).__init__()
        self.abs_path = json.load(open("params.txt"))["abs_path"] 
        self.steer_df = pd.read_csv(self.abs_path + "main/data.csv")
        self.transforms = transforms
        self.dutils = Data_Utils()
    
    def __len__(self):
        return len(self.steer_df)
    
    def __getitem__(self, idx):
        """
        Returns tuple (cropped_image(Tensor, C x H x W), steering angle (float 1x1 tensor))
        """
        img_name, angle = self.steer_df.iloc[idx, 0], self.steer_df.iloc[idx, 1]
        cv_img = cv2.imread(self.abs_path + 'main/' + img_name)
        #preprocess img & label
        img_tensor, label = self.dutils.preprocess_img(cv_img, angle, use_for='train')

        if self.transforms:
            img_tensor = self.transforms(img_tensor)

        return (img_tensor, angle)

class Data_Utils(object):
    """
    Fun & Useful functions for dealing with Steer Data
    """
    def __init__(self):
        self.params = json.load(open("params.txt"))
        self.abs_path = self.params["abs_path"]

    def show_steer_angle_hist(self, foldername, w_matplotlib=False):
        """
        Shows a histogram illustrating distribution of steering angles
        w_matplotlib: boolean to trigger plotting with matplotlib. will otherwise plot to bash with bashplotlib
        """
        steer_df = pd.read_csv(self.abs_path + foldername + '/data.csv')
        angle_column = steer_df.iloc[:, 1].values
        num_bins = 20
        if(w_matplotlib):
            #save plot with matplotlib
            plt.hist(angle_column, num_bins, color='green')
            plt.title("Distribution of steering angles (rads)")
            plt.savefig('steer_hist.png')
        else:
            #Plots in bash terminal
            num_bins = 20
            plot_hist(angle_column, num_bins, binwidth=0.01, colour='green', title='Distribution of steering angles (rads)', xlab=True, showSummary=True)
    
    def get_dataloaders(self, batch_size):
        """
        Returns a training & validation dataloader
        """
        steer_dataset = SteerDataset()
        vsplit = 0.2 #80, 20 split

        #idxs for train & valid
        dset_size = len(steer_dataset)
        idxs = list(range(dset_size))
        split = int(np.floor(vsplit * dset_size))
        np.random.shuffle(idxs)
        train_idxs, val_idxs = idxs[split:], idxs[:split]

        #Using SubsetRandomSampler but should ideally sample equally from each steer angle to avoid distributional bias
        train_sampler = SubsetRandomSampler(train_idxs)
        val_sampler = SubsetRandomSampler(val_idxs)

        train_dataloader = DataLoader(steer_dataset, batch_size=batch_size, sampler=train_sampler)
        valid_dataloader = DataLoader(steer_dataset, batch_size=batch_size, sampler=val_sampler)

        return train_dataloader, valid_dataloader
    
    def is_valid_img(self, cv_img, label):
        """
        A series of checks to ensure that an image label pair is valid
        cv_img: img from cv2
        label: float representing angle
        """
        #stupid check to see if the picture is exceptionally dark
        if (np.mean(cv_img) == 0):
            return False
        return True

    def preprocess_img(self, cv_img, label=None, use_for='vis', whichcam=None):
        """
        Alter img_label pair for training AND inference AND visualization
        cv_img: img from cv2
        label: float representing angle
        use_for = use for 'infer', 'train', 'vis'
        Returns an image tensor or cv image
        """
        if use_for == 'vis':
            #vis fixes the dataset to make them look like what they should look like for training
            #use 'whicham' to alter the labels 
            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)

            steer_offset = 0.15
            if whichcam == 'left':
                label += steer_offset

            elif whichcam == 'right':
                label -= steer_offset

            elif whichcam == 'front':
                label = label

            if label:
                label = label * 180.0/math.pi

            return cv_img, label

        elif use_for == 'infer':
            #We don't need or have the labels for inference, but make sure to get the inputs images to look like they do in training

            cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_CLOCKWISE)

            #fix image & convert to tensor
            cv_crop = cv_img[200:, :, :]
            img_tensor = torch.from_numpy(cv_crop).float()#size (H x W x C)
            img_tensor = img_tensor.permute(2, 0, 1)#size (C x H x W)
            return img_tensor, label
        
        elif use_for == 'train':
            #we're training on label=degrees & rotated images
            label_tensor = label
            if label:
                #fix label (turn to 1-Tensor)
                label_tensor = np.array([label])
                label_tensor = torch.from_numpy(label_tensor)

            #fix image & convert to tensor
            cv_crop = cv_img[200:, :, :]
            img_tensor = torch.from_numpy(cv_crop).float()#size (H x W x C)
            img_tensor = img_tensor.permute(2, 0, 1)#size (C x H x W)
            return img_tensor, label_tensor
        else:
            return None, None
    

    def preprocess_dataset(self, orig_foldername, new_foldername, whichcam):
        """
        Make a new dataset from data in orig_foldername and convert to new_foldername (WARNING: will overwrite folder)
        """
        nf_path = self.abs_path + new_foldername + '/'
        of_path = self.abs_path + orig_foldername + '/'
        old_df = pd.read_csv(of_path + 'data.csv')
        
        #make new folder & overrwite
        if os.path.exists(nf_path) and nf_path != of_path:
            os.system('rm -r ' + nf_path)
        os.mkdir(nf_path)

        #set up new csv file/dataframe
        col_names = old_df.columns.values
        new_df = pd.DataFrame(columns=col_names)

        for i in range(len(old_df)):
            old_row = old_df.iloc[i]
            img_name, angle = old_row[0], old_row[1]
            old_img = cv2.imread(of_path + img_name)
            if(self.is_valid_img(old_img, angle)):
                new_img, new_angle = self.preprocess_img(old_img, angle, use_for='vis', whichcam='front')
                cv2.imwrite(nf_path + img_name, new_img)
                new_row = old_row.copy()
                new_row[1] = new_angle
                new_df = new_df.append(new_row.copy())
        new_df.to_csv(nf_path + 'data.csv', index=False)

    def augment_img(self, cv_img, label):
        #for now just flip the image and the label
        cv_img = cv2.flip(cv_img, 1)
        label = label * -1.0
        return cv_img, label

    def augment_dataset(self, foldername='main'):
        """
        Read from a folder and add to its dataset
        """
        fpath = self.abs_path + foldername + '/'
        old_df = pd.read_csv(fpath + 'data.csv')
        new_df = pd.DataFrame(columns=old_df.columns.values)
        for i in range(len(old_df)):
            old_row = old_df.iloc[i]
            img_name, angle = old_row[0], old_row[1]
            old_img = cv2.imread(fpath + img_name)
            new_img, new_angle = self.augment_img(old_img, angle)
            new_img_name = 'aug_' + img_name
            cv2.imwrite(fpath + new_img_name, new_img)
            new_row = old_row.copy()
            new_row[0] = new_img_name
            new_row[1] = new_angle
            new_df = new_df.append(new_row.copy())
        old_df.append(new_df)
        os.system('rm ' + fpath + 'data.csv')
        old_df.to_csv(fpath + 'data.csv', index=False)
    
    def combine_image_folders(self, folder_list):
        final_dest = self.params['final_dest']
        for folder in folder_list:
            data = self.abs_path + folder + '/data.csv'
            df = pd.read_csv(data)
            print(data, len(df))
            image_names = df.iloc[:,0]
            [shutil.copy(os.path.join(self.abs_path + folder,file), final_dest) for file in image_names]

    def combine_csvs(self, folder_list):
        df = pd.concat([pd.read_csv(self.abs_path + f+'/data.csv') for f in folder_list])

        if os.path.exists(self.params['final_dest']):
            os.system('rm -r ' + self.params['final_dest'])
        os.mkdir(self.params['final_dest'])

        path = os.path.join(self.params['final_dest'] , 'data.csv')
        df.to_csv(path, index=False)

        self.combine_image_folders(folder_list)

def main():
    du = Data_Utils()
    du.preprocess_dataset('front_folder', 'main_front', whichcam='front')
    du.preprocess_dataset('left_folder', 'main_left', whichcam='left')
    du.preprocess_dataset('right_folder', 'main_right', whichcam='right')
    du.combine_csvs(['main_front', 'main_left', 'main_right'])

if __name__ == '__main__':
    main()