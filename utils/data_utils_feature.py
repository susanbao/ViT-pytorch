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

# [image, region_16, region_32, region_8]
CONV_THR_DICT = {"PSPNet_VOC": [1.5, 10],
                 "UNet_VOC": [0, 22],
                 "DeepLab_VOC": [1.5, 11],
                 "FCN_VOC": [2, 12],
                 "SEGNet_VOC": [4, 30],
                 "PSPNet_CITY": [0.5, 11],
                 "UNet_CITY": [3.2, 13.4],
                 "DeepLab_CITY": [0.7, 11],
                 "FCN_CITY": [1.1, 11.8],
                 "SEGNet_VOC": [1.9, 21.4]}

# PSPNet_VOC
# conv_thresholds = torch.linspace(0, 1.5, steps=num_classes)
# conv_thresholds_patch = torch.linspace(0, 10, steps=num_classes)

# UNet_coco10k
# conv_thresholds = torch.linspace(0, 8, steps=num_classes)
# conv_thresholds_patch = torch.linspace(0, 24, steps=num_classes)

# UNet_VOC
# conv_thresholds = torch.linspace(0, 8, steps=num_classes)
# conv_thresholds_patch = torch.linspace(0, 22, steps=num_classes) # 16x16
# conv_thresholds_patch = torch.linspace(0, 16, steps=num_classes) # 32x32

# DeepLab_VOC
# conv_thresholds = torch.linspace(0, 1.5, steps=num_classes)
# conv_thresholds_patch = torch.linspace(0, 11, steps=num_classes)

# FCN_VOC
# conv_thresholds = torch.linspace(0, 2, steps=num_classes)
# conv_thresholds_patch = torch.linspace(0, 12, steps=num_classes)

# SEGNet_VOC
# conv_thresholds = torch.linspace(0, 4, steps=num_classes)
# conv_thresholds_patch = torch.linspace(0, 30, steps=num_classes)

# PSPNet_CITY
# conv_thresholds = torch.linspace(0, 0.5, steps=num_classes)
# conv_thresholds_patch = torch.linspace(0, 11, steps=num_classes)

# conv_values = (conv_thresholds[1:] + conv_thresholds[:-1]) / 2

# conv_values_patch = (conv_thresholds_patch[1:] + conv_thresholds_patch[:-1]) / 2

def np_read_with_tensor_output(file):
    with open(file, "rb") as outfile:
        data = np.load(outfile)
    return torch.from_numpy(data.astype(np.float32))

def read_one_json_results(path):
    with open(path, "r") as outfile:
        data = json.load(outfile)
    return data

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
        self.conv_thresholds = torch.linspace(0, CONV_THR_DICT[self.model_data_type][0], steps=self.num_classes)
        if args.region_size == 16:
            self.conv_thresholds_patch = torch.linspace(0, CONV_THR_DICT[self.model_data_type][1], steps=self.num_classes)
            self.avgpool = torch.nn.AdaptiveAvgPool2d((30,30))
        elif args.region_size == 32:
            self.conv_thresholds_patch = torch.linspace(0, CONV_THR_DICT[self.model_data_type][2], steps=self.num_classes)
            self.avgpool = torch.nn.AdaptiveAvgPool2d((15,15))
        self.conv_values = (self.conv_thresholds[1:] + self.conv_thresholds[:-1]) / 2
        self.conv_values_patch = (self.conv_thresholds_patch[1:] + self.conv_thresholds_patch[:-1]) / 2
        
        self.annotations = np_read_with_tensor_output(annotation_dir)
        self.annotations = tensor_float_to_ordinal(self.annotations, self.conv_thresholds)
        self.feature_dir = input_dir + "/feature/"
        self.image_dir = input_dir + "/image/"
        self.loss_dir = input_dir + "/loss/"
        self.entropy_dir = input_dir + "/entropy/"
        self.lens = self.annotations.shape[0] if length == 0 else length
        self.shift = shift
        self.aug = aug
        
    
    def __getitem__(self, index):
        index = index + self.shift
        file_name = str(index//8) + ".npy"
        one_image = np_read_with_tensor_output(self.image_dir + file_name)
        image_index = index % 8
        # one_result = np_read_with_tensor_output(self.feature_dir + file_name)
        # feature = one_result[image_index]
        feature = np_read_with_tensor_output(self.feature_dir + str(index) + ".npy")
        image = one_image[image_index]
        # feature = F.softmax(feature, dim=0)
        # entropy = torch.sum(torch.mul(-feature, torch.log(feature + 1e-20)), dim=0).unsqueeze(dim=0)
        
        entropy = np_read_with_tensor_output(self.entropy_dir + file_name)
        entropy = entropy[image_index]
        feature = torch.cat((image, feature, entropy), dim=0)
        # without one part of input
        # feature = torch.cat((image, entropy), dim=0)
        
        annotation = self.annotations[index]
        losses = np_read_with_tensor_output(self.loss_dir + file_name)
        loss = losses[image_index]
        loss = torch.unsqueeze(loss, dim=0)
        loss = self.avgpool(loss)

        if self.aug:
            if random.random()>0.5:
                feature = torch.flip(feature, [1])
                loss = torch.flip(loss, [1])
            if random.random()>0.5:
                feature = torch.flip(feature, [2])
                loss = torch.flip(loss, [2])
        loss = torch.flatten(loss)
        loss = tensor_float_to_ordinal_patch(loss, self.conv_thresholds_patch)
        annotation = torch.cat((annotation.unsqueeze(0), loss), dim=0)
        return tuple((feature, annotation))
    
    def get_item_with_indices(self, index):
        index = index + self.shift
        file_name = str(index//8) + ".npy"
        one_result = np_read_with_tensor_output(self.feature_dir + file_name)
        one_image = np_read_with_tensor_output(self.image_dir + file_name)
        image_index = index % 8
        feature = one_result[image_index]
        image = one_image[image_index]
        # feature = F.softmax(feature, dim=0)
        # entropy = torch.sum(torch.mul(-feature, torch.log(feature + 1e-20)), dim=0).unsqueeze(dim=0)
        entropy = np_read_with_tensor_output(self.entropy_dir + file_name)
        entropy = entropy[image_index]
        feature = torch.cat((image, feature, entropy), dim=0)
        annotation = self.annotations[index]
        losses = np_read_with_tensor_output(self.loss_dir + file_name)
        loss = losses[image_index]
        loss = torch.unsqueeze(loss, dim=0)
        loss = self.avgpool(loss)

        if self.aug:
            if random.random()>0.5:
                feature = torch.flip(feature, [1])
                loss = torch.flip(loss, [1])
            if random.random()>0.5:
                feature = torch.flip(feature, [2])
                loss = torch.flip(loss, [2])
        loss = torch.flatten(loss)
        loss = tensor_float_to_ordinal_patch(loss, self.conv_thresholds_patch)
        annotation = torch.cat((annotation.unsqueeze(0), loss), dim=0)
        return tuple((feature, annotation, top_indices))
    
    def __len__(self):
        return self.lens

def get_loader_feature(args):
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

    return train_loader, test_loader