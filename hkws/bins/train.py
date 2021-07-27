#!/usr/bin/env python

# Copyrigh 2018 houjingyong@gmail.com

# Apache 2.0.

from __future__ import print_function

import os, sys, argparse, datetime, shutil
import numpy as np
import random
import yaml

import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append("./")
from hkws.models.multi_scale_tcn import MultiScaleTCN
from hkws.models.multi_scale_tcn_causal import MultiScaleCausalTCN
from hkws.models.scheduler import WarmupLR
from hkws.bins.execution import train, validate, test
from hkws.utils.utils import count_parameters, set_mannul_seed
from hkws.dataset.audio_dataset import WavDataset, collate_fn_wav


def get_args():
    """Get arguments from stdin."""
    parser = argparse.ArgumentParser(description="Pytorch acoustic model.")
    parser.add_argument("--config", type=str, default="", help="config file")
    parser.add_argument(
        "--encoder",
        type=str,
        default="multi_scale_tcn",
        help="encoder type {default: gru}",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=40,
        metavar="N",
        help="Input feature dimension without context (default: 40).",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=2,
        metavar="N",
        help="Kernel size of Wavenet or CNN (default:3).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        metavar="N",
        help="Hidden dimension of feature extractor (default: 128).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        metavar="N",
        help="Numbers of hidden layers of feature extractor (default: 2).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=4,
        metavar="N",
        help="Number of covlutional layers for each block (default: 4).",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=1,
        metavar="N",
        help="Output dimension, number of classes (default: 1).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0001,
        metavar="DR",
        help="dropout of feature extractor (default: 0.0001).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        metavar="N",
        help="Maximum epochs to train (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Batch size for training (default: 8).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        metavar="LR",
        help="Initial learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--warm-up-steps",
        type=int,
        default=1000,
        metavar="warmup",
        help="Warm up steps (default: 1000).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="noam",
        metavar="optimizer",
        help="optimizer used for training.",
    )
    parser.add_argument(
        "--init-weight-decay",
        type=float,
        default=5e-5,
        metavar="WD",
        help="Weight decay (L2 normalization) (default: 5e-5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        metavar="S",
        help="Random seed (default: 1234).",
    )
    parser.add_argument(
        "--use-cuda", type=int, default=1, metavar="C", help="Use cuda (1) or cpu(0)."
    )
    parser.add_argument(
        "--train", type=int, default=1, help="Executing mode, train (1) or test (0)."
    )
    parser.add_argument("--train-scp", type=str, default="", help="Training data file.")
    parser.add_argument(
        "--dev-scp", type=str, default="", help="Development data file."
    )
    parser.add_argument(
        "--save-dir", type=str, default="", help="Directory to output the model."
    )
    parser.add_argument(
        "--load-model", type=str, default="", help="Previous model to load."
    )
    parser.add_argument(
        "--test", type=int, default=0, help="Executing mode, 1 for test, 0 no test"
    )
    parser.add_argument("--test-scp", type=str, default="", help="Test data file.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="How many batches to wait before logging training status.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        metavar="N",
        help="How many workers used to load data",
    )
    args = parser.parse_args()
    return args


class Model(nn.Module):
    def __init__(self, encoder, cls):
        super(Model, self).__init__()
        self.encoder = encoder
        self.cls = cls

    def forward(self, data, lenghts):
        output = self.encoder(data, lenghts)
        output = torch.mean(output, dim=1)
        output = self.cls(output)
        return output


class ModelC(nn.Module):
    def __init__(self, encoder, cls):
        super(ModelC, self).__init__()
        self.encoder = encoder
        self.cls = cls

    def forward(self, data, lenghts):
        output = self.encoder(data, lenghts)
        output = output[:, -1, :]
        output = self.cls(output)
        return output


def get_model(args):
    if args.encoder == "multi_scale_tcn":
        encoder = MultiScaleTCN(
            layer_size=args.block_size,
            stack_size=args.num_layers,
            in_channels=args.input_dim,
            res_channels=args.hidden_dim,
            out_channels=args.output_dim,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
        cls = nn.Sequential(
            nn.Linear(args.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(64, args.output_dim),
        )
        return Model(encoder, cls)
    elif args.encoder == "multi_scale_tcn_causal":
        encoder = MultiScaleCausalTCN(
            layer_size=args.block_size,
            stack_size=args.num_layers,
            in_channels=args.input_dim,
            res_channels=args.hidden_dim,
            out_channels=args.output_dim,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
        cls = nn.Sequential(
            nn.Linear(args.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(64, args.output_dim),
        )
        return ModelC(encoder, cls)
    else:
        print("we don't support this kind of neural network: %s\n" % (args.encoder))
        exit(1)


def main():
    args = get_args()
    with open(args.config) as fid:
        configs = yaml.load(fid, Loader=yaml.FullLoader)
    dataset_configs = configs["dataset"]
    feature_extraction_conf = dataset_configs["feature_extraction_conf"]
    set_mannul_seed(args.seed)
    device = torch.device("cuda" if args.use_cuda else "cpu")

    model = get_model(args).to(device)
    params = count_parameters(model)
    print("Num parameters: %d, Num Flops: %d\n" % (params, 0))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.init_weight_decay
    )
    scheduler = WarmupLR(optimizer, warmup_steps=args.warm_up_steps)
    start_epoch = 0

    if args.load_model != "" and args.train:
        print("=> Loading previous checkpoint to train: {}".format(args.load_model))
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        val_loss = checkpoint["val_loss"]
        step = checkpoint["step"]
        start_epoch = checkpoint["epoch"]
        scheduler.set_step(step)
    elif args.load_model != "" and not args.train and args.test:
        print("=> Loading previous checkpoint to test: {}".format(args.load_model))
        checkpoint = torch.load(args.load_model)
        model.load_state_dict(checkpoint["model"])
    elif not args.train:
        sys.exit("Option --load-model should not be empty for testing.")
    else:
        print("=> No checkpoint found.")

    if 0 == start_epoch and args.train:
        model_path = args.save_dir + "/0.pt"
        torch.save(
            {
                "val_loss": float("inf"),
                "step": 0,
                "epoch": 0,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            model_path,
        )
    # For training
    if args.train:
        print("Yaml configs: \n {}".format(configs))
        print("Arguments:\n {}".format(args))
        print("Model:\n {}".format(model))
        print("Optimizer:\n {}".format(optimizer))

        if args.train_scp == "" or args.dev_scp == "":
            sys.exit("Options --train-scp and --dev-scp are required for training.")

        if args.save_dir == "":
            sys.exit("Option --save-dir is required to save model.")

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # Training data loader
        speed_perturb = dataset_configs["speed_perturb"]
        feature_dither = dataset_configs["feature_dither"]
        spec_aug = dataset_configs["spec_aug"]
        spec_aug_conf = dataset_configs["spec_aug_conf"]
        train_set = WavDataset(
            args.train_scp,
            with_label=True,
            shuffle=True,
            feature_extraction_conf=feature_extraction_conf,
            feature_dither=feature_dither,
            speed_perturb=speed_perturb,
            spec_aug=spec_aug,
            spec_aug_conf=spec_aug_conf,
        )
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn_wav,
            pin_memory=True,
        )

        # Dev data loader
        dev_set = WavDataset(
            args.dev_scp,
            with_label=True,
            shuffle=False,
            feature_extraction_conf=feature_extraction_conf,
        )
        dev_loader = DataLoader(
            dataset=dev_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn_wav,
            pin_memory=True,
        )

        for epoch in range(start_epoch + 1, args.max_epochs):
            tr_loss = train(
                args, model, device, train_loader, optimizer, scheduler, epoch
            )
            val_loss = validate(args, model, device, dev_loader, epoch)

            model_path = args.save_dir + "/" + str(epoch) + ".pt"
            torch.save(
                {
                    "val_loss": val_loss,
                    "step": scheduler.get_current_step(),
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                model_path,
            )

    # For testing
    if args.test:
        # Test data loader
        if args.test_scp == "":
            sys.exit("Options --test-scp and --output-file are required for testing")
        test_set = WavDataset(
            args.test_scp,
            with_label=True,
            shuffle=False,
            feature_extraction_conf=feature_extraction_conf,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_wav,
            pin_memory=True,
        )
        test(args, model, device, test_loader, "")


if __name__ == "__main__":
    main()
