import logging

import torch
import numpy as np
import os

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, TensorDataset, Dataset
import json
import torch.nn.functional as F
import torchvision.transforms as T
import random

logger = logging.getLogger(__name__)

def np_read_with_tensor_output(file):
    with open(file, "rb") as outfile:
        data = np.load(outfile)
    return torch.from_numpy(data.astype(np.float32))

CONV_THR_DICT = {"PSPNet_VOC": [1.6, 11, 9.6, 11.5],
                 "UNet_VOC": [5.1, 17.3, 15.6, 17.7],
                 "DeepLab_VOC": [1.1, 10.1, 7.6, 11.2],
                 "FCN_VOC": [1, 12.4, 9.8, 14.2],
                 "SEGNet_VOC": [1, 13.4, 10.7, 19],
                 "PSPNet_CITY": [0.4, 10.2, 7.6, 13.9],
                 "UNet_CITY": [2.5, 12.3, 10.2, 13.3],
                 "DeepLab_CITY": [0.5, 10.7, 9.7, 12.1],
                 "FCN_CITY": [0.8, 13.1, 10.1, 14.5],
                 "SEGNet_CITY": [1.0, 17.3, 12.3, 26],
                 "PSPNet_COCO": [1.8, 19.7, 14, 20.8],
                 "UNet_COCO": [7.3, 21.8, 19.4, 24],
                 "SEGNet_COCO": [3.9, 47.6, 34.1, 97.3],
                 "FCN_COCO": [2.6, 25.4, 13.7, 33.4],
                 "PSPNet_ADE20K": [3.5, 20.3, 17.9, 21.2],
                 "UNet_ADE20K": [8.4, 65.5, 57, 69.1],
                 "SEGNet_ADE20K": [3.7, 41.4, 37.6, 60.6],
                 "FCN_ADE20K": [3.4, 19.4, 14.8, 31.5]}

def tensor_float_to_ordinal(inputs, conv_thresholds):
    ordinal_classes = torch.zeros_like(inputs, dtype=torch.long)
    for i, threshold in enumerate(conv_thresholds[:-1]):
        ordinal_classes[inputs >= threshold] = i
    return ordinal_classes

def tensor_ordinal_to_float(input_logits, conv_values):
    classification = input_logits.argmax(dim=1)
    results = conv_values[classification]
    return results

def tensor_float_to_ordinal_patch(inputs, conv_thresholds_patch):
    ordinal_classes = torch.zeros_like(inputs, dtype=torch.long)
    for i, threshold in enumerate(conv_thresholds_patch[:-1]):
        ordinal_classes[inputs >= threshold] = i
    return ordinal_classes

def tensor_ordinal_to_float_patch(input_logits, conv_values_patch):
    classification = input_logits.argmax(dim=1)
    results = conv_values_patch[classification]
    return results

class FeatureDataset(Dataset):
    """ Use feature from other model as dataset """
    def __init__(self, input_dir, annotation_dir, args, length = 0, shift = 0, aug = True):
        self.num_classes = args.ordinal_class_num + 1
        self.model_data_type = args.model_data_type
        self.annotations = np_read_with_tensor_output(annotation_dir)
        self.feature_dir = input_dir + "/feature/"
        self.image_dir = input_dir + "/image/"
        self.loss_dir = input_dir + "/loss/"
        self.entropy_dir = input_dir + "/entropy/"
        self.lens = self.annotations.shape[0] * 900 if length == 0 else length
        self.shift = shift
        self.aug = aug
        self.avgpool = torch.nn.AdaptiveAvgPool2d((30,30))
        self.conv_thresholds = torch.linspace(0, CONV_THR_DICT[self.model_data_type][0], steps=self.num_classes)
        self.conv_thresholds_patch = torch.linspace(0, CONV_THR_DICT[self.model_data_type][1], steps=self.num_classes)
        self.conv_values = (self.conv_thresholds[1:] + self.conv_thresholds[:-1]) / 2
        self.conv_values_patch = (self.conv_thresholds_patch[1:] + self.conv_thresholds_patch[:-1]) / 2
        
    
    def __getitem__(self, index):
        image_level_index = index // 900
        region_level_index = index % 900
        file_name = str(image_level_index//8) + ".npy"
        one_image = np_read_with_tensor_output(self.image_dir + file_name)
        image_index = image_level_index % 8
        feature = np_read_with_tensor_output(self.feature_dir + str(image_level_index) + ".npy")
        image = one_image[image_index]
        
        entropy = np_read_with_tensor_output(self.entropy_dir + file_name)
        entropy = entropy[image_index]
        feature = torch.cat((image, feature, entropy), dim=0)
        x_loc = region_level_index // 30
        y_loc = region_level_index % 30
        feature
        feature = feature[:, x_loc*16:(x_loc+1)*16, y_loc*16:(y_loc+1)*16]
        
        losses = np_read_with_tensor_output(self.loss_dir + file_name)
        loss = losses[image_index]
        loss = torch.unsqueeze(loss, dim=0)
        loss = self.avgpool(loss)
        loss = loss[0, x_loc, y_loc]

        if self.aug:
            if random.random()>0.5:
                feature = torch.flip(feature, [1])
            if random.random()>0.5:
                feature = torch.flip(feature, [2])
        loss = torch.flatten(loss)
        loss = tensor_float_to_ordinal_patch(loss, self.conv_thresholds_patch)
        return tuple((feature, loss))
    
    def __len__(self):
        return self.lens

def get_loader_resnet_region_feature(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        
    model_data_path = args.data_dir
    split = "train"
    inputs_path = model_data_path + split
    store_preprocess_annotations_path = model_data_path + split + "/image_true_losses.npy"
    train_datasets = FeatureDataset(inputs_path, store_preprocess_annotations_path, args)
    
    split = "val"
    inputs_path = model_data_path + split
    store_preprocess_annotations_path = model_data_path + split + "/image_true_losses.npy"
    test_datasets = FeatureDataset(inputs_path, store_preprocess_annotations_path, args, aug=False)

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

    return train_loader, test_loader, train_datasets, test_datasets