import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class KittiLoader(Dataset):
    def __init__(self, root_dir, mode, input_channels, input_height, input_width, do_augmentation = False):
        left_dir = os.path.join(root_dir, 'image_02/data/')				
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname\
                           in os.listdir(left_dir)])					
        if mode == 'train':
            right_dir = os.path.join(root_dir, 'image_03/data/')			
            self.right_paths = sorted([os.path.join(right_dir, fname) for fname\
                                in os.listdir(right_dir)])				
            assert len(self.right_paths) == len(self.left_paths)
        # self.transform = transform							
        self.mode = mode

    def __len__(self):
        return len(self.left_paths)
	
    def __getitem__(self, idx):								
        left_video = np.load(self.left_paths[idx])
	# left_video = torch.from_numpy(left_video).double() ###					
        if self.mode == 'train':
            right_video = np.load(self.right_paths[idx])
	    # right_video = torch.from_numpy(right_video).double() ###				
            sample = {'left_video': left_video, 'right_video': right_video}		
            return sample
        else:
            return left_video
