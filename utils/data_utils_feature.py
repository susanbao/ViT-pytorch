import logging

import torch
import numpy as np
import os

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, TensorDataset, Dataset

logger = logging.getLogger(__name__)

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
    def __init__(self, input_dir, annotation_dir, length = 0):
        self.annotations = np_read_with_tensor_output(annotation_dir)
        self.input_dir = input_dir
        self.lens = self.annotations.shape[0] if length == 0 else length
    
    def __getitem__(self, index):
        one_result = read_one_json_results(self.input_dir+ str(index)+".json")
        token = torch.FloatTensor(one_result["self_feature"])
        feature_idx = one_result["feature_idx"]
        feature = np_read_with_tensor_output(self.input_dir+ "feature" + str(feature_idx)+".npy")
        feature = torch.cat((token, feature), dim=0)
        annotation = self.annotation[index]
        return tuple(feature, annotation)
    
    def __len__(self):
        return self.lens

def get_loader_feature(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        
    model_data_path = args.data_dir
    data_name = args.data_name
    split = "val"
    store_preprocess_inputs_path = model_data_path + split + "/feature_pre_data/"
    store_preprocess_annotations_path = model_data_path + split + "/feature_pre_data/annotation.npy"
    train_datasets = FeatureDataset(store_preprocess_inputs_path, store_preprocess_annotations_path)
    
    splite = "train"
    store_preprocess_inputs_path = model_data_path + split + "/feature_pre_data/"
    store_preprocess_annotations_path = model_data_path + split + "/feature_pre_data/annotation.npy"
    test_datasets = FeatureDataset(store_preprocess_inputs_path, store_preprocess_annotations_path)

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