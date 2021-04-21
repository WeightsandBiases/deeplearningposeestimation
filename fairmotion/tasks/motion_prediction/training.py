# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

from fairmotion.tasks.motion_prediction import generate, utils, test
from fairmotion.utils import utils as fairmotion_utils


logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

def get_train_stats():
    pkl_file = open("stats_train.pkl", "rb")
    angle_mean = pickle.load(pkl_file)
    angle_stddev = pickle.load(pkl_file)
    angle_percentile = pickle.load(pkl_file)
    velocity_mean = pickle.load(pkl_file)
    velocity_stddev = pickle.load(pkl_file)
    velocity_percentile = pickle.load(pkl_file)
    acceleration_mean = pickle.load(pkl_file)
    acceleration_stddev = pickle.load(pkl_file)
    acceleration_percentile = pickle.load(pkl_file)
    pkl_file.close()

    return (angle_mean, angle_stddev, angle_percentile,
            velocity_mean, velocity_stddev, velocity_percentile,
            acceleration_mean, acceleration_stddev, acceleration_percentile)

class MSEWithDeviationLoss(nn.Module):
    def __init__(self, factor=1):
        super(MSEWithDeviationLoss, self).__init__()
        stats = get_train_stats()
        self.mean_pose = torch.Tensor(stats[0])
        self.std_pose = torch.Tensor(stats[1] + 1e-8)
        self.factor = factor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean_tiled = self.mean_pose.expand(input.shape).type(input.dtype).to(input.device)
        std_tiled = self.std_pose.expand(input.shape).type(input.dtype).to(input.device)

        normalization = torch.abs((input - mean_tiled)/std_tiled*self.factor).mean()
        return F.mse_loss(input, target) + normalization



class MSEWithOutlierLoss(nn.Module):
    def __init__(self, factor=1):
        super(MSEWithOutlierLoss, self).__init__()
        stats = get_train_stats()
        self.max_ranges = torch.Tensor(stats[2][:, 99].reshape(1, stats[2].shape[0]))
        self.min_ranges = torch.Tensor(stats[2][:, 1].reshape(1, stats[2].shape[0]))
        self.factor = factor

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        max_tiled = self.max_ranges.expand(input.shape).type(input.dtype).to(input.device)
        min_tiled = self.min_ranges.expand(input.shape).type(input.dtype).to(input.device)

        upper_error = (((torch.where(max_tiled > input, max_tiled, input) - max_tiled) * self.factor) ** 2).mean()
        lower_error = (((torch.where(min_tiled < input, min_tiled, input) - min_tiled) * self.factor) ** 2).mean()
        return F.mse_loss(input, target) + upper_error + lower_error


def set_seeds():
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def select_criterion(args):
    if args.criterion == "l1":
        criterion = nn.L1Loss()
    elif args.criterion == "sl1":
        criterion = nn.SmoothL1Loss()
    elif args.criterion == "d":
        criterion = MSEWithDeviationLoss()
    elif args.criterion == "o":
        criterion = MSEWithOutlierLoss()
    else:
        criterion = nn.MSELoss()
    return criterion

def train(args):
    fairmotion_utils.create_dir_if_absent(args.save_model_path)
    logging.info(args._get_kwargs())
    utils.log_config(args.save_model_path, args)

    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device if args.device else device
    logging.info(f"Using device: {device}")

    logging.info("Preparing dataset...")
    dataset, mean, std = utils.prepare_dataset(
        *[
            os.path.join(args.preprocessed_path, f"{split}.pkl")
            for split in ["train", "test", "validation"]
        ],
        batch_size=args.batch_size,
        device=device,
        shuffle=args.shuffle,
    )

    # number of predictions per time step = num_joints * angle representation
    # shape is (batch_size, seq_len, num_predictions)
    _, tgt_len, num_predictions = next(iter(dataset["train"]))[1].shape

    model = utils.prepare_model(
        input_dim=num_predictions,
        hidden_dim=args.hidden_dim,
        device=device,
        num_layers=args.num_layers,
        architecture=args.architecture,
    )
    
    criterion = select_criterion(args)

    model.init_weights()
    training_losses, val_losses = [], []

    epoch_loss = 0
    for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
        model.eval()
        src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
        outputs = model(src_seqs, tgt_seqs, teacher_forcing_ratio=1,)
        loss = criterion(outputs, tgt_seqs.to(torch.float).to(args.device))
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / (iterations + 1)
    val_loss = generate.eval(
        model, criterion, dataset["validation"], args.batch_size, device,
    )
    logging.info(
        "Before training: "
        f"Training loss {epoch_loss} | "
        f"Validation loss {val_loss}"
    )

    logging.info("Training model...")
    torch.autograd.set_detect_anomaly(True)
    opt = utils.prepare_optimizer(model, args.optimizer, args.lr, args.factor)
    for epoch in range(args.epochs):
        epoch_loss = 0
        model.train()
        teacher_forcing_ratio = np.clip(
            (1 - 2 * epoch / args.epochs), a_min=0, a_max=1,
        )
        logging.info(
            f"Running epoch {epoch} | "
            f"teacher_forcing_ratio={teacher_forcing_ratio}"
        )
        for iterations, (src_seqs, tgt_seqs) in enumerate(dataset["train"]):
            opt.optimizer.zero_grad()
            src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
            outputs = model(
                src_seqs, tgt_seqs, teacher_forcing_ratio=teacher_forcing_ratio
            )
            outputs = outputs.double()
            loss = criterion(
                outputs,
                utils.prepare_tgt_seqs(args.architecture, src_seqs, tgt_seqs),
            )
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / (iterations + 1)
        training_losses.append(epoch_loss)
        val_loss = generate.eval(
            model, criterion, dataset["validation"], args.batch_size, device,
        )
        val_losses.append(val_loss)
        opt.epoch_step(val_loss=val_loss)
        logging.info(
            f"Training loss {epoch_loss} | "
            f"Validation loss {val_loss} | "
            f"Iterations {iterations + 1}"
        )
        if epoch % args.save_model_frequency == 0:
            _, rep = os.path.split(args.preprocessed_path.strip("/"))
            _, mae, _ = test.test_model(
                model=model,
                dataset=dataset["validation"],
                rep=rep,
                device=device,
                mean=mean,
                std=std,
                max_len=tgt_len,
            )
            logging.info(f"Validation MAE: {mae}")
            torch.save(
                model.state_dict(), f"{args.save_model_path}/{epoch}.model"
            )
            if len(val_losses) == 0 or val_loss <= min(val_losses):
                torch.save(
                    model.state_dict(), f"{args.save_model_path}/best.model"
                )
    return training_losses, val_losses


def plot_curves(args, training_losses, val_losses):
    plt.plot(range(len(training_losses)), training_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.ylabel("MSE Loss")
    plt.xlabel("Epoch")
    plt.savefig(f"{args.save_model_path}/loss.svg", format="svg")


def main(args):
    train_losses, val_losses = train(args)
    plot_curves(args, train_losses, val_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sequence to sequence motion prediction training"
    )
    parser.add_argument(
        "--preprocessed-path",
        type=str,
        help="Path to folder with pickled " "files",
        required=True,
    )
    parser.add_argument(
        "--batch-size", type=int, help="Batch size for training", default=64
    )
    parser.add_argument(
        "--shuffle", action='store_true',
        help="Use this option to enable shuffling",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Hidden size of LSTM units in encoder/decoder",
        default=1024,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of layers of LSTM/Transformer in encoder/decoder",
        default=1,
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        help="Path to store saved models",
        required=True,
    )
    parser.add_argument(
        "--save-model-frequency",
        type=int,
        help="Frequency (in terms of number of epochs) at which model is "
        "saved",
        default=5,
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=200
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Training device",
        default=None,
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--architecture",
        type=str,
        help="Seq2Seq architecture to be used",
        default="seq2seq",
        choices=[
            "seq2seq",
            "tied_seq2seq",
            "transformer",
            "transformer_encoder",
            "rnn"
        ],
    )
    parser.add_argument(
        "--lr", type=float, help="Learning rate", default=None,
    )
    parser.add_argument(
        "--factor", type=float, help="Factor for noamopt", default=2,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        help="Torch optimizer",
        default="sgd",
        choices=["adam", "sgd", "noamopt"],
    )
    parser.add_argument(
        "--criterion",
        type=str,
        help="Loss Function",
        default="mse",
        choices=["mse", "l1", "sl1", "d", "o"],
    )
    args = parser.parse_args()
    main(args)
