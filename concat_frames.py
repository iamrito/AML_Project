import glob, os
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
import shutil

size = (128,256) ###
root_path = "/home/SharedData/Phalguni/Amrit/AML_Code/KITTI_Dataset"
paths = os.listdir(root_path)
for path in paths:
	all_left_videos_path = os.path.join(root_path, path, 'image_02/data/')
	left_video_path = os.listdir(all_left_videos_path)
	#print(os.listdir(all_left_videos_path))
	for v_path in left_video_path:
		inside_loop_path = os.path.join(all_left_videos_path,v_path)
		if(os.path.isdir(inside_loop_path)):		
	 		image_concat = np.zeros((3,128,256))   ###
			count = 0
			sorted_images = sorted(os.listdir(inside_loop_path))	
			for f in sorted_images:
				count += 1
				image = transforms.ToTensor()(transforms.Resize(size)(Image.open(os.path.join(inside_loop_path,f))))
				image = np.asarray(image)
				image_concat = np.concatenate([image_concat,image],axis=0)
			video = np.stack([image_concat[3*i:3*i+3,:,:] for i in np.arange(1,count+1)],axis=0)
			# video = np.stack([image_concat[i:i+1,:,:] for i in np.arange(1,count+1)],axis=0) # for grayscale processing
			file_name  = sorted_images[0].split('.')[0]		
			file_name = os.path.join(all_left_videos_path,file_name)
			np.save((file_name +'.npy'), video)
			shutil.rmtree(file_name)


for path in paths:
	all_right_videos_path = os.path.join(root_path, path, 'image_03/data/')
	right_video_path = os.listdir(all_right_videos_path)
	#print(os.listdir(all_right_videos_path))
	for v_path in right_video_path:
		inside_loop_path = os.path.join(all_right_videos_path,v_path)
		if(os.path.isdir(inside_loop_path)):		
			image_concat = np.zeros((3,128,256))  ###
			count = 0
			sorted_images = sorted(os.listdir(inside_loop_path))	
			for f in sorted_images:
				count += 1
				image = transforms.ToTensor()(transforms.Resize(size)(Image.open(os.path.join(inside_loop_path,f))))
				image = np.asarray(image)
				image_concat = np.concatenate([image_concat,image],axis=0)
			video = np.stack([image_concat[3*i:3*i+3,:,:] for i in np.arange(1,count+1)],axis=0)  ###
			#video = np.stack([image_concat[i:i+1,:,:] for i in np.arange(1,count+1)],axis=0) # for grayscale processing
			file_name  = sorted_images[0].split('.')[0]		
			file_name = os.path.join(all_right_videos_path,file_name)
			np.save((file_name+'.npy'), video)
			shutil.rmtree(file_name)
