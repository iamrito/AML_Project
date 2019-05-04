import torch
import collections
import os
from torch.utils.data import DataLoader, ConcatDataset

from models_resnet import ConvLSTM, Resnet_18
from data_loader import KittiLoader

def to_device(input, device):
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device = device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device = device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")


def get_model(model, input_channels = 3, pretrained = False):				
    if model == 'ConvLSTM':
        out_model = ConvLSTM(input_channels)
    else:
        out_model = Resnet_18(input_channels)
    return out_model


def prepare_dataloader(data_directory, mode, do_augmentation, batch_size, input_channels, size, num_workers):
    data_dirs = os.listdir(data_directory)
    datasets = [KittiLoader(os.path.join(data_directory,data_dir), mode, input_channels, size[0], size[1], do_augmentation) for data_dir in data_dirs]
    # print(datasets)                       
    dataset = ConcatDataset(datasets)
    n_img = len(dataset)
    print('Use a dataset with', n_img, 'samples')
    if mode == 'train':
        loader = DataLoader(dataset, batch_size = batch_size,				
                            shuffle = False, num_workers = num_workers,
                            pin_memory = True)
    else:
        loader = DataLoader(dataset, batch_size = batch_size,
                            shuffle = False, num_workers = num_workers,
                            pin_memory = True)
    return n_img, loader
