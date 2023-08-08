import torch
import torch.nn as nn
import torch.optim as optim
import logging
import argparse
import os
import random
import numpy as np
import ipdb
import json
import wandb
import socket
from utils.data_utils_feature import get_loader_feature

def write_one_results(path, json_data):
    with open(path, "w") as outfile:
        json.dump(json_data, outfile)

def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pth" % args.name)
    torch.save({"state_dict": model_to_save.state_dict()}, model_checkpoint)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.loss_function = torch.nn.SmoothL1Loss(reduction='mean')

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.LayerNorm(layer_sizes[i+1]))  # Adding LayerNorm between hidden layers
                layers.append(nn.ReLU())  # Adding ReLU activation between hidden layers

        self.model = nn.Sequential(*layers)

        self.global_avg_pooling = nn.AdaptiveAvgPool2d(30)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, labels=None):
        x,_ = torch.max(x, dim=1, keepdim=True)
        x = self.global_avg_pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.model(x)
        if labels is not None:
            loss = 100 * self.loss_function(x, labels)
            return loss
        else:
            return x, None


def train(args, model):
    # Prepare dataset
    # train_loader, test_loader = get_loader_at(args)
    train_loader, test_loader = get_loader_feature(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    
    model.zero_grad()

    if args.enable_wandb:
        wandb.watch(model, log="all")

    num_epochs = 10
    global_step = 0
    record_val_loss = 100000000
    val_loss = 0
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad()  # Zero the gradients
            loss = model(images, labels)  # Forward pass
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            if (global_step + 1) % args.eval_every == 0:
                all_preds = []
                model.eval()
                total_correct = 0
                total_samples = 0
                loss_fct = torch.nn.L1Loss(reduction='sum')
                val_loss = 0
                with torch.no_grad():
                    for val_images, val_labels in test_loader:
                        val_images, val_labels = val_images.to(args.device), val_labels.to(args.device)
                        val_outputs = model(val_images)[0]
                        val_loss += loss_fct(val_outputs, val_labels)
                        total_samples += val_labels.size(0)
                        all_preds.extend(val_outputs.tolist())

                    val_loss = val_loss / total_samples
                    print(f"val_loss: {val_loss}")
                    if val_loss < record_val_loss:
                        save_model(args, model)
                        record_val_loss = val_loss
                        path = os.path.join(args.output_dir, "%s_losses.json" % args.name)
                        json_objects = {"losses": all_preds}
                        write_one_results(path, json_objects)

            if args.enable_wandb:
                wandb.log({"loss":loss, "val_loss": val_loss}, step=global_step)
                wandb.log(model.state_dict())
            global_step += 1

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
    parser.add_argument('--enable_wandb', action='store_true',
                        help="Whether to enable wandb")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    if args.enable_wandb:
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
    model = MLP(900, [100,10], output_size = 1)
    model.to(args.device)

    # Training
    train(args, model)
    if args.enable_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()