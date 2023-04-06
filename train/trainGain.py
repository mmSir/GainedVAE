import math
import random
import sys
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# pylint: disable=E0611,E0401
from compressai.models.gain import GainedMSHyperprior
from compressai.datasets import ImageFolder
from compressai.zoo import models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename="D:/"):
    torch.save(state, filename, _use_new_zipfile_serialization=False)


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target, lmbda):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = set(
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    )
    aux_parameters = set(
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    )

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(parameters))),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(aux_parameters))),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="mbt2018-mean",
        choices=models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "--tensorboard-runs", type=str, default="D:\MXH\STPM\CompressAI\\trainmeanscale\GainedMSHP", help="Training root"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=8,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", help="Save model to disk")
    parser.add_argument(
        "--seed", type=float, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a pretrained checkpoint")
    parser.add_argument("--model-save", type=str, help="Path to save a checkpoint")
    args = parser.parse_args(argv)
    return args


def test_epoch(iteration, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            # test all discrete levels
            for s in range(0, model.levels):
                out_net = model(d, s)
                out_criterion = criterion(out_net, d, model.lmbda[s])

                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test iteration {iteration}: Average losses:"
        f"\tLoss: {loss.avg:.4f} |"
        f"\tMSE loss: {mse_loss.avg:.4f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.2f}"
    )

    return loss.avg


def train(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.crop_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.crop_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )


    net = GainedMSHyperprior()
    # net = FeatureGainedMSHyperprior()
    net = net.to(device)


    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.2)
    criterion = RateDistortionLoss()

    # Tensorboard
    train_writer = SummaryWriter(os.path.join(args.tensorboard_runs, 'train'))

    last_epoch = 0
    iterations = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        iterations = checkpoint["iterations"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    print(f"Start training from epoch {last_epoch}, iterations {iterations} !")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        net.train()
        device = next(net.parameters()).device

        for i, d in enumerate(train_dataloader):
            d = d.to(device)

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            s = random.randint(0, net.levels - 1) # choose random level from [0, levels-1]
            out_net = net(d, s)

            out_criterion = criterion(out_net, d, net.lmbda[s])
            out_criterion["loss"].backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_max_norm)
            optimizer.step()

            aux_loss = net.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            if i % 100 == 0:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i*len(d)}/{len(train_dataloader.dataset)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.4f} |'
                    f'\ts: {s} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.4f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.4f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
                train_writer.add_scalar(f"Train_{s}/Total_Loss", out_criterion["loss"].item(), iterations)
                train_writer.add_scalar(f"Train_{s}/Bpp_Loss", out_criterion["bpp_loss"].item(), iterations)
                train_writer.add_scalar(f"Train_{s}/MSE_Loss", out_criterion["mse_loss"].item(), iterations)
                train_writer.add_scalar("aux_loss", aux_loss, iterations)

            if iterations % 10000 == 0 and iterations != 0:
                loss = test_epoch(iterations, test_dataloader, net, criterion)
                lr_scheduler.step(loss)
                print(f"Iterations:{iterations}  Eval Loss:{loss}")
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
                train_writer.add_scalar("EVAL/Loss", loss, iterations)
                train_writer.add_scalar("LearningRate/lr", optimizer.param_groups[0]['lr'], iterations)

                net.train()

            iterations = iterations + 1

        if epoch % 2 == 0:
            print(f"saving models at epoch{epoch}......")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "iterations": iterations,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                filename=args.model_save + f"checkpoint_gainmshp_epoch{epoch}.pth"
            )


if __name__ == "__main__":
    train(sys.argv[1:])
