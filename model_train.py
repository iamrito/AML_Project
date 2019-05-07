import os
import torch
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

from main_monodepth_pytorch import Model

torch.cuda.is_available()

torch.cuda.device_count()

dict_parameters = edict({'data_dir':'/home/SharedData/Phalguni/Amrit/AML_Code/KITTI_train/',
                         'val_data_dir':'/home/SharedData/Phalguni/Amrit/AML_Code/KITTI_val/',
                         'model_path':'/home/SharedData/Phalguni/Amrit/AML_Code/model/monodepth_convlstm.pth',
                         'output_directory':'/home/SharedData/Phalguni/Amrit/AML_Code/output/',
                         'input_height':128,
                         'input_width':256,
                         'model':'ConvLSTM',
                         'pretrained':True,
                         'mode':'train',
                         'epochs':60,
                         'learning_rate':1e-3,
                         'batch_size': 8,
                         'adjust_lr':True,
                         'device':'cuda:0',
                         'do_augmentation':False,
                         'print_images':False,
                         'print_weights':False,
                         'input_channels': 3,
                         'num_workers': 8,
                         'use_multiple_gpu': True})

model = Model(dict_parameters)
model.load('/home/SharedData/Phalguni/Amrit/AML_Code/model/monodepth_convlstm22_last.pth')
model.train()
