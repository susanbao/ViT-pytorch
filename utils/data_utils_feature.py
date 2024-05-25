import logging

import torch
import numpy as np
import os

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, TensorDataset, Dataset
import json

logger = logging.getLogger(__name__)

# region
# conv_thresholds = torch.linspace(0, 10, steps=num_classes)

# region
CONV_THR_DICT = {"DETR_COCO": [35, 37],
                "DFDETR_COCO": [8, 9],
                "DINO_COCO": [8, 8.5],
                "DETR_CITY": [9.8, 10.4],
                "DEDETR_CITY": [6.6, 7.3],
                "DELADETR_CITY": [7,8, 9.1],
                "YOLOX_COCO": [13.6, 23.4],
                "SSD_COCO": [48.2, 50]}


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
    def __init__(self, input_dir, annotation_dir, args, length = 0, shift = 0):
        self.num_classes = args.ordinal_class_num + 1

        self.model_data_type = args.model_data_type
        self.conv_thresholds = torch.linspace(0, CONV_THR_DICT[self.model_data_type][0], steps=self.num_classes)
        self.conv_thresholds_patch = torch.linspace(0, CONV_THR_DICT[self.model_data_type][1], steps=self.num_classes)
        self.conv_values = (self.conv_thresholds[1:] + self.conv_thresholds[:-1]) / 2
        self.conv_values_patch = (self.conv_thresholds_patch[1:] + self.conv_thresholds_patch[:-1]) / 2

        self.annotations = np_read_with_tensor_output(annotation_dir)
        if args.loss_range != "nobin":
            self.annotations = tensor_float_to_ordinal(self.annotations, self.conv_thresholds)
        self.feature_dir = input_dir + "/feature/"
        self.annotation_dir = input_dir + "/annotation/"
        self.lens = self.annotations.shape[0] if length == 0 else length
        self.shift = shift
        self.loss_range = args.loss_range
    
    def __getitem__(self, index):
        index = index + self.shift
        feature = np_read_with_tensor_output(self.feature_dir + str(index)+".npy")
        feature = (feature - feature.mean()) / feature.std()
        
        one_annotation = read_one_json_results(self.annotation_dir+ str(index)+".json")
        patch_loss = torch.tensor(one_annotation['loss'])
        img_loss = self.annotations[index]
        patch_index = torch.tensor(one_annotation['index'])
        if self.loss_range != "nobin":
            patch_loss = tensor_float_to_ordinal_patch(patch_loss, self.conv_thresholds_patch)

        annotation = torch.full((feature.shape[0],), 255, dtype=patch_loss.dtype)
        annotation[patch_index] = patch_loss
        annotation = torch.cat((img_loss.unsqueeze(0), annotation), dim=0)
        return tuple((feature, annotation))
    
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
    test_datasets = FeatureDataset(inputs_path, store_preprocess_annotations_path, args)

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