# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling_at import ActiveTestVisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils_at import get_loader_at
from utils.data_utils_feature import get_loader_feature
from utils.dist_util import get_world_size
import ipdb
import json
import wandb
import socket
from torch import nn
from torchvision.models import resnet18


logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.loss_function = torch.nn.MSELoss(reduction='mean')

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.LayerNorm(layer_sizes[i+1]))  # Adding LayerNorm between hidden layers
                layers.append(nn.ReLU())  # Adding ReLU activation between hidden layers

        self.model = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, labels=None):
        x = self.model(x)
        if labels is not None:
            loss = self.loss_function(x, labels)
            return loss
        else:
            return x, None

class MPLNet(nn.Module):
    def __init__(self, input_dims = 10000, output_dims = 1, dropout = 0.1):
        super().__init__()
        self.input_dims = 21*256
        self.layer1=nn.Linear(self.input_dims, 1000)
        self.layer2=nn.Linear(1000, 100)
        self.layer3=nn.Linear(100, 10)
        self.layer4=nn.Linear(10, 1)
        self.norm1 = nn.LayerNorm(1000)
        self.norm2 = nn.LayerNorm(100)
        self.norm3 = nn.LayerNorm(10)
        self._init_parms(self.layer1)
        self._init_parms(self.layer2)
        self._init_parms(self.layer3)
        self._init_parms(self.layer4)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)
        self.start_dims = 5*256
        self.loss_function = torch.nn.MSELoss(reduction='mean')
    
    def _init_parms(self, module):
        module.weight.data.normal_(mean=0.0, std=1.0)
        
    def forward(self, x, labels=None):
        x = x[:,:21,self.start_dims:]
        x=x.reshape((-1, self.input_dims))
        x=nn.functional.relu(self.norm1(self.layer1(x)))
        x=nn.functional.relu(self.norm2(self.layer2(x)))
        x=nn.functional.relu(self.norm3(self.layer3(x)))
        x=self.layer4(x)
        x = x.reshape(x.shape[0])
        if labels is not None:
            loss = self.loss_function(x, labels)
            return loss
        else:
            return x, None

# Define the Ridge Regression model using nn.Module
class RidgeRegression(nn.Module):
    def __init__(self, input_size, output_size, alpha=1.0):
        super(RidgeRegression, self).__init__()
        self.alpha = alpha
        self.linear = nn.Linear(input_size, output_size)
        self.loss_function = torch.nn.MSELoss(reduction='mean')

    def forward(self, x, labels=None):
        x = self.linear(x)
        if labels is not None:
            loss = self.loss_function(x, labels)
            return loss
        else:
            return x, None

# Define the ResNet-18 regression model
class ResNetRegression(nn.Module):
    def __init__(self, input_channels, output_size):
        super(ResNetRegression, self).__init__()
        self.resnet = resnet18(pretrained=True)
        # Modify the first layer to accommodate the specified input_channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, output_size)  # Change the fully connected layer to output_size units
        self.loss_function = torch.nn.MSELoss(reduction='mean')
        nn.init.kaiming_uniform_(self.resnet.conv1.weight)
        nn.init.kaiming_uniform_(self.resnet.fc.weight)

    def forward(self, x, labels=None):
        x = self.resnet(x)
        if labels is not None:
            loss = self.loss_function(x, labels)
            return loss
        else:
            return x, None

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pth" % args.name)
    torch.save({"state_dict": model_to_save.state_dict()}, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.input_feature_dim = args.input_feature_dim
    config.ash_per = args.ash_per
    # model = ResNetRegression(input_channels = 21, output_size = 1)
    model = MLP(21, [50,30,10], output_size = 1)
    # model.load_from(np.load(args.pretrained_dir), requires_grad = args.enable_backbone_grad)
    # model.load_state_dict(torch.from_numpy(np.load(args.pretrained_dir)))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def write_one_results(path, json_data):
    with open(path, "w") as outfile:
        json.dump(json_data, outfile)        

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds = []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.MSELoss(reduction='mean')
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            all_preds.extend(logits.tolist())

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)

    writer.add_scalar("test/loss", scalar_value=eval_losses.avg, global_step=global_step)
    return eval_losses.avg, all_preds


def train(args, model):
    # ipdb.set_trace()
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    # train_loader, test_loader = get_loader_at(args)
    train_loader, test_loader = get_loader_feature(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc = 0, 100000
    accuracy = 0
    wandb.watch(model, log="all")
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy, all_preds = valid(args, model, writer, test_loader, global_step)
                    if best_acc > accuracy:
                        save_model(args, model)
                        best_acc = accuracy
                        path = os.path.join(args.output_dir, "%s_losses.json" % args.name)
                        json_objects = {"losses": all_preds}
                        write_one_results(path, json_objects)
                    model.train()

                if global_step % t_total == 0:
                    break
                wandb.log({"loss":loss, "val_loss": accuracy}, step=global_step)
                wandb.log(model.state_dict())
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--data_dir", type=str, default="../orkspace/DINO/data/5_scale_31/",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--data_name", type=str, default="box_level_ViT",
                        help="the folder name to have data.")
    parser.add_argument("--input_feature_dim", type=int, default=95,
                        help="length of the input feature for each sample.")
    parser.add_argument("--ash_per", default=80, type=int,
                        help="percentage for ash technique")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=400, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--encoder_weight_train', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--enable_backbone_grad', action='store_true',
                        help="Whether to enable the retraining of backbone")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)
    
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Bosch_active_testing",
        config=args,
        entity="susanbao",
        notes=socket.gethostname(),
        name=args.name,
        job_type="training"
    )

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)
    
    wandb.finish()


if __name__ == "__main__":
    main()
