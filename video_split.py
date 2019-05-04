import os, os.path, shutil

root_path = "/home/SharedData/Phalguni/Amrit/AML_Code/KITTI_Dataset/"
paths = os.listdir(root_path)
for path in paths:
	folder_path = os.path.join(root_path, path, 'image_02/data/')
	print(folder_path)
	images = os.listdir(folder_path)
	images = sorted(images)
	N = 3 ###
	split_videos = (N*(len(images)/N))
	print(len(images),split_videos)
	for i in range(len(images)-split_videos):
		os.remove(os.path.join(folder_path,images[split_videos+i]))
	for i in range(split_videos):
    		if(i%N == 0): 
    			folder_name = images[i].split('.')[0]
    		new_path = os.path.join(folder_path, folder_name)
    		if not os.path.exists(new_path):
        		os.makedirs(new_path)
    		old_image_path = os.path.join(folder_path, images[i])
    		new_image_path = os.path.join(new_path, images[i])
    		shutil.move(old_image_path, new_image_path)

	folder_path = os.path.join(root_path, path, 'image_03/data/')
	print(folder_path)
	images = os.listdir(folder_path)
	images = sorted(images)
	for i in range(len(images)-split_videos):
		os.remove(os.path.join(folder_path,images[split_videos+i]))
	for i in range(split_videos):
    		if(i%N == 0):
    			folder_name = images[i].split('.')[0]
    		new_path = os.path.join(folder_path, folder_name)
    		if not os.path.exists(new_path):
        		os.makedirs(new_path)
    		old_image_path = os.path.join(folder_path, images[i])
    		new_image_path = os.path.join(new_path, images[i])
    		shutil.move(old_image_path, new_image_path)
	
