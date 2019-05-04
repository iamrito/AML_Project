import torch
import torchvision.transforms as transforms
import numpy as np


# augment_parameters = [flip, augment, gamma, brightness, colors]
def image_transforms(mode='train', augment_parameters=[0, 0, 0.9, 0.9, 0],
                     do_augmentation=True, transformations=None,  size=(256, 512)):
    if mode == 'train':
        data_transform = transforms.Compose([
            ResizeImage(train=True, size=size),
            RandomFlip(do_augmentation,augment_parameters),
            ToTensor(train=True),
            AugmentImagePair(augment_parameters, do_augmentation)
        ])
        return data_transform
    elif mode == 'test':
        data_transform = transforms.Compose([
            ResizeImage(train=False, size=size),
            ToTensor(train=False),
            DoTest(),
        ])
        return data_transform
    elif mode == 'custom':
        data_transform = transforms.Compose(transformations)
        return data_transform
    else:
        print('Wrong mode')


class ResizeImage(object):
    def __init__(self, train=True, size=(256, 512)):
        self.train = train
        self.transform = transforms.Resize(size)

    def __call__(self, sample):
	sample = self.transform(sample)
	return sample
	    


class DoTest(object):
    def __call__(self, sample):
	new_sample = torch.stack((sample, torch.flip(sample, [2])))
	return new_sample


class ToTensor(object):
    def __init__(self, train):
        self.train = train
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
	sample = self.transform(sample)
	return sample


class RandomFlip(object):
    def __init__(self, do_augmentation, augment_parameters):
        self.transform = transforms.RandomHorizontalFlip(p=1)
        self.do_augmentation = do_augmentation
	self.flip_prob = augment_parameters[0]

    def __call__(self, sample):
        k = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if k > self.flip_prob:
                sample = self.transform(sample)
        return sample


class AugmentImagePair(object):
    def __init__(self, augment_parameters, do_augmentation):
        self.do_augmentation = do_augmentation

	self.augment_prob = augment_parameters[1]
	self.random_gamma = augment_parameters[2]
	self.random_brightness = augment_parameters[3]
	self.random_colors = augment_parameters[4]

    def __call__(self, sample):
        p = np.random.uniform(0, 1, 1)
        if self.do_augmentation:
            if p > self.augment_prob:
                # randomly shift gamma
                #random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
                sample = sample ** self.random_gamma
                # randomly shift brightness
                #random_brightness = np.random.uniform(self.brightness_low, self.brightness_high)
                sample = sample * self.random_brightness

                # randomly shift color
                #random_colors = np.random.uniform(self.color_low, self.color_high, 3)
                #for i in range(3):
                  #  sample[i, :, :] *= self.random_colors[i]

                # saturate
                sample = torch.clamp(sample, 0, 1)
        return sample
