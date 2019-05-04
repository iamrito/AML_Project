import os, os.path, shutil
import numpy as np

root_path = "/home/SharedData/Phalguni/Amrit/AML_Code/KITTI_Dataset/"
paths = os.listdir(root_path)
for path in paths:
	print(path)
	left_folder_path = os.path.join(root_path, path, 'image_02/data/')
	right_folder_path = os.path.join(root_path, path, 'image_03/data/')
	
	left_arrays = sorted(os.listdir(left_folder_path))
	right_arrays = sorted(os.listdir(right_folder_path))
	count = 0	
	for i in range(len(left_arrays)):
		count += 1   		
		if(count%5==0):
			new_path_left = os.path.join("/home/SharedData/Phalguni/Amrit/AML_Code/KITTI_val/", path,'image_02/data')
			new_path_right = os.path.join("/home/SharedData/Phalguni/Amrit/AML_Code/KITTI_val/", path,'image_03/data')
			
			if(os.path.exists(new_path_left) == False):
				os.makedirs(new_path_left)
				os.makedirs(new_path_right)
			old_array_path_left = os.path.join(left_folder_path, left_arrays[i])
			new_array_path_left = os.path.join(new_path_left, left_arrays[i])
			shutil.copyfile(old_array_path_left, new_array_path_left)

			old_array_path_right = os.path.join(right_folder_path, right_arrays[i])
			new_array_path_right = os.path.join(new_path_right, right_arrays[i])
			shutil.copyfile(old_array_path_right, new_array_path_right)
		else:
			new_path_left = os.path.join("/home/SharedData/Phalguni/Amrit/AML_Code/KITTI_train/", path,'image_02/data')
			new_path_right = os.path.join("/home/SharedData/Phalguni/Amrit/AML_Code/KITTI_train/", path,'image_03/data')
			
			if(os.path.exists(new_path_left) == False):
				os.makedirs(new_path_left)
				os.makedirs(new_path_right)
			old_array_path_left = os.path.join(left_folder_path, left_arrays[i])
			new_array_path_left = os.path.join(new_path_left, left_arrays[i])
			shutil.copyfile(old_array_path_left, new_array_path_left)

			old_array_path_right = os.path.join(right_folder_path, right_arrays[i])
			new_array_path_right = os.path.join(new_path_right, right_arrays[i])
			shutil.copyfile(old_array_path_right, new_array_path_right)
	


