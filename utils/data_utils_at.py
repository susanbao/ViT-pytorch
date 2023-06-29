import logging

import torch
import numpy as np

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, TensorDataset


logger = logging.getLogger(__name__)


def get_loader_at(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        
    model_data_path = args.data_dir
    split = "val"
    store_preprocess_inputs_path = model_data_path + split + f"/pre_data/{split}_inputs.npy"
    store_preprocess_annotations_path = model_data_path + split + f"/pre_data/{split}_annotations.npy"
    with open(store_preprocess_inputs_path, 'rb') as outfile:
        train_inputs = torch.from_numpy(np.load(outfile).astype(np.float32))
    with open(store_preprocess_annotations_path, 'rb') as outfile:
        train_annotations = torch.from_numpy(np.load(outfile).astype(np.float32))
        
    split = "val"
    store_preprocess_inputs_path = model_data_path + split + f"/pre_data/{split}_inputs.npy"
    store_preprocess_annotations_path = model_data_path + split + f"/pre_data/{split}_annotations.npy"
    with open(store_preprocess_inputs_path, 'rb') as outfile:
        test_inputs = torch.from_numpy(np.load(outfile).astype(np.float32))
    with open(store_preprocess_annotations_path, 'rb') as outfile:
        test_annotations = torch.from_numpy(np.load(outfile).astype(np.float32))
        
    train_datasets = TensorDataset(train_inputs, train_annotations)
    test_datasets = TensorDataset(test_inputs, test_annotations)

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
