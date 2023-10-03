import logging

import torch
import numpy as np
import os

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, TensorDataset, Dataset
import json

logger = logging.getLogger(__name__)

normalize_list = {"loss_bbox": [0.09300409561655773, 0.0767726528828114], "loss_ce": [0.3522786986406354, 0.7470140154753052], "loss_giou": [1.1611697194208146, 0.47297514438862676], "loss": [3.1396386155214007, 1.4399662609117525]}

num_classes = 51

conv_thresholds = torch.linspace(0, 10, steps=num_classes)

conv_values = (conv_thresholds[1:] + conv_thresholds[:-1]) / 2


def np_read_with_tensor_output(file):
    with open(file, "rb") as outfile:
        data = np.load(outfile)
    return torch.from_numpy(data.astype(np.float32))

def read_one_json_results(path):
    with open(path, "r") as outfile:
        data = json.load(outfile)
    return data

def tensor_float_to_ordinal(inputs):
    ordinal_classes = torch.zeros_like(inputs, dtype=torch.long)
    for i, threshold in enumerate(conv_thresholds[:-1]):
        ordinal_classes[inputs >= threshold] = i
    return ordinal_classes

def tensor_ordinal_to_float(input_logits):
    classification = input_logits.argmax(dim=1)
    results = conv_values[classification]
    return results


class FeatureDataset(Dataset):
    """ Use feature from other model as dataset """
    def __init__(self, input_dir, annotation_dir, feature_path, loss_type, length = 0, shift = 0):
        self.annotations = np_read_with_tensor_output(annotation_dir)
        self.annotations = tensor_float_to_ordinal(self.annotations)
        # self.annotations = (self.annotations - normalize_list[loss_type][0])/normalize_list[loss_type][1]
        # self.annotations = torch.log(self.annotations)
        self.input_dir = input_dir
        self.feature_dir = feature_path
        self.lens = self.annotations.shape[0] if length == 0 else length
        self.shift = shift
    
    def __getitem__(self, index):
        one_result = read_one_json_results(self.input_dir+ str(index+self.shift)+".json")
        # token = torch.FloatTensor(one_result["self_feature"])
        # feature_idx = one_result["feature_idx"]
        # feature = np_read_with_tensor_output(self.input_dir+ "feature" + str(feature_idx)+".npy")
        # feature = torch.cat((token, feature), dim=0)
        selected_idxs = one_result["selected_idxs"]
        feature_idx = one_result["feature_idx"]
        feature = np_read_with_tensor_output(self.feature_dir + str(feature_idx)+".npy")
        feature = feature.permute(1,0,2)
        feature = feature.reshape((feature.shape[0], -1))
        feature = feature[selected_idxs]
        feature = (feature - feature.mean()) / feature.std()
        # feature[21:,:] = 0.0
        annotation = self.annotations[index+self.shift]
        return tuple((feature, annotation))
    
    def __len__(self):
        return self.lens

def get_loader_feature(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        
    model_data_path = args.data_dir
    data_name = args.data_name
    loss_type = args.loss_type
    if loss_type == "loss":
        annotation_name = "annotation.npy"
    else:
        annotation_name = f"annotation_{loss_type}.npy"
    split = "train"
    store_preprocess_inputs_path = model_data_path + split + "/feature_pre_data/"
    store_preprocess_annotations_path = model_data_path + split + "/feature_pre_data/" + annotation_name
    feature_path = model_data_path + split + "/feature_data/"
    train_datasets = FeatureDataset(store_preprocess_inputs_path, store_preprocess_annotations_path, feature_path, loss_type)
    
    split = "val"
    store_preprocess_inputs_path = model_data_path + split + "/feature_pre_data/"
    store_preprocess_annotations_path = model_data_path + split + "/feature_pre_data/" + annotation_name
    feature_path = model_data_path + split + "/feature_data/"
    test_datasets = FeatureDataset(store_preprocess_inputs_path, store_preprocess_annotations_path, feature_path, loss_type)

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