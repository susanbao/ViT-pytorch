import logging

import torch
import numpy as np
import os

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, TensorDataset, Dataset
import json
import torch.nn.functional as F

logger = logging.getLogger(__name__)

normalize = [0.05269893, 0.053517897]

def np_read_with_tensor_output(file):
    with open(file, "rb") as outfile:
        data = np.load(outfile)
    return torch.from_numpy(data.astype(np.float32))

def read_one_json_results(path):
    with open(path, "r") as outfile:
        data = json.load(outfile)
    return data

class FeatureDataset(Dataset):
    """ Use feature from other model as dataset """
    def __init__(self, input_dir, annotation_dir, length = 0, shift = 0):
        self.annotations = np_read_with_tensor_output(annotation_dir)
        # self.annotations = (self.annotations - normalize[0])/normalize[1]
        self.annotations[self.annotations < 0.01] = 0.01
        self.annotations = torch.log(self.annotations)
        self.feature_dir = input_dir + "/output/"
        self.image_dir = input_dir + "/image/"
        self.loss_dir = input_dir + "/loss/"
        self.lens = self.annotations.shape[0] if length == 0 else length
        self.shift = shift
        self.avgpool = torch.nn.AdaptiveAvgPool2d((30,30))
    
    def __getitem__(self, index):
        index = index + self.shift
        file_name = str(index//8) + ".npy"
        one_result = np_read_with_tensor_output(self.feature_dir + file_name)
        one_image = np_read_with_tensor_output(self.image_dir + file_name)
        image_index = index % 8
        feature = one_result[image_index]
        image = one_image[image_index]
        feature = F.softmax(feature, dim=0)
        feature[feature>0.99] = 1
        feature[feature<0.01] = 0
        feature = torch.cat((image, feature), dim=0)
        annotation = self.annotations[index]
        losses = np_read_with_tensor_output(self.loss_dir + file_name)
        loss = losses[image_index]
        loss = torch.unsqueeze(loss, dim=0)
        loss = self.avgpool(loss)
        loss = torch.flatten(loss)
        loss[loss < 0.001] = 0.001
        loss = torch.log(loss)
        annotation = torch.cat((annotation.unsqueeze(0), loss), dim=0)
        return tuple((feature, annotation))
    
    def __len__(self):
        return self.lens

def get_loader_feature(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        
    model_data_path = args.data_dir
    data_name = args.data_name
    split = "train"
    inputs_path = model_data_path + split
    store_preprocess_annotations_path = model_data_path + split + "/image_true_losses.npy"
    train_datasets = FeatureDataset(inputs_path, store_preprocess_annotations_path)
    
    split = "val"
    inputs_path = model_data_path + split
    store_preprocess_annotations_path = model_data_path + split + "/image_true_losses.npy"
    test_datasets = FeatureDataset(inputs_path, store_preprocess_annotations_path)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(train_datasets) if args.local_rank == -1 else DistributedSampler(train_datasets)
    test_sampler = SequentialSampler(test_datasets)
    train_loader = DataLoader(train_datasets,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(test_datasets,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if test_datasets is not None else None

    return train_loader, test_loader