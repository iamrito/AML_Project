import os
import torch
import numpy as np
import skimage.transform
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
from main_monodepth_pytorch import Model

def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


torch.cuda.is_available()

torch.cuda.device_count()

dict_parameters_test = edict({'data_dir':'/home/SharedData/Phalguni/Amrit/AML_Code/kitti_test',
                              'model_path':'/home/SharedData/Phalguni/Amrit/AML_Code/model_copy/monodepth_convlstm22_cpt.pth',
                              'output_directory':'/home/SharedData/Phalguni/Amrit/AML_Code/output',
                              'input_height':128,
                              'input_width':256,
                              'model':'ConvLSTM',
                              'pretrained':False,
                              'mode':'test',
                              'device':'cuda:0',
                              'input_channels':3,
                              'num_workers':8,
                              'use_multiple_gpu':True})
model_test = Model(dict_parameters_test)
model_test.test()

root_path = '/home/SharedData/Phalguni/Amrit/AML_Code/output/'
paths = os.listdir(root_path)
count = 0
frames = 0
for path in paths:
	count += 1
	disp_batch = np.load(os.path.join(root_path,path))
	for i in range(disp_batch.shape[0]):	
		disp_sample = disp_batch[i,:,:,:,:]
		for j in range(disp_sample.shape[0]):
			disp_frame = disp_sample[j,:,:,:]
			disp_frame_pp = post_process_disparity(disp_frame)
			disp_to_img = skimage.transform.resize(disp_frame_pp, [375, 1242], mode='constant')
			frames += 1
			plt.imsave(root_path +str(count) +'_' +str(i) + '_'+str(j) +'_test_output.png', disp_to_img, cmap='plasma')
print(frames)
