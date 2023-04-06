import math
import warnings
import random
import sys
import argparse
import os
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np


from compressai.models.gain import SCGainedMSHyperprior
from compressai.zoo import models
from roi_image_dataset import get_dataloader, get_test_dataloader_compressai

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class MovingAverage(object):
    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.queue = deque()
        self.Max_size = size

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self.queue.append(val)
        if len(self.queue) > self.Max_size:
            self.queue.popleft()
        return 1.0 * sum(self.queue) / len(self.queue)

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

def save_checkpoint(state, filename="D:/MXH/"):
    torch.save(state, filename, _use_new_zipfile_serialization=False)

def qmap2lsmap(qmap):
    '''
    function that convert qmap to lambda scaling map
    '''
    return (8/3 * qmap.pow(2) + 1/3).pow(1.6355)
    # return 1e-3 * torch.exp(4.382 * qmap) # iccv paper


class PixelwiseRateDistortionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, output, target, lmbdamap):
        # lmbdamap: (B, 1, H, W)
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out['bpp_loss'] = sum(
            (-torch.log2(likelihoods).sum() / num_pixels)
            for likelihoods in output['likelihoods'].values()
        )

        mse = self.mse(output['x_hat'], target)
        lmbdamap = lmbdamap.expand_as(mse)
        out['mse_loss'] = 255 ** 2 * torch.mean(lmbdamap * mse)
        out['loss'] = out['mse_loss'] + out['bpp_loss']

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
        "-d", "--dataset", type=str, required=True, help="Training dataset root"
    )
    parser.add_argument(
        "--tensorboard-runs", type=str, default="D:\MXH", help="Training root"
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
        default=1,
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


def test_epoch(iteration, test_dataloaders, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = [AverageMeter() for x in range(0, model.levels)]
    bpp_loss = [AverageMeter() for x in range(0, model.levels)]
    mse_loss = [AverageMeter() for x in range(0, model.levels)]
    aux_loss = AverageMeter()

    # todo: return different level eval loss by a dict/list?
    with torch.no_grad():
        for test_dataloader in test_dataloaders:
            for i, (d, qmap) in enumerate(test_dataloader):
                d = d.to(device)
                qmap = qmap.to(device)
                lmbda_scale_map = qmap2lsmap(qmap)
                for s in range(0, model.levels):
                    out_net = model(d, s, qmap)
                    out_criterion = criterion(out_net, d, model.lmbda[s]*lmbda_scale_map)

                    aux_loss.update(model.aux_loss())
                    bpp_loss[s].update(out_criterion["bpp_loss"])
                    loss[s].update(out_criterion["loss"])
                    mse_loss[s].update(out_criterion["mse_loss"])

    for s in range(model.levels):
        print(
            f"Test iteration {iteration}: Level:{s} Average losses:"
            f"\tLoss: {loss[s].avg:.4f} |"
            f"\tMSE loss: {mse_loss[s].avg:.4f} |"
            f"\tBpp loss: {bpp_loss[s].avg:.4f} |"
            f"\tAux loss: {aux_loss.avg:.2f}"
        )

    model.train()

    return {
        "loss": loss,
        "bpp_loss": bpp_loss,
        "mse_loss": mse_loss,
        "aux_loss": aux_loss
    }


def train(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_dataloader, test_dataloaders = get_dataloader(args, stage=3, L=5)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    net = SCGainedMSHyperprior()
    net = net.to(device)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.2)
    criterion = PixelwiseRateDistortionLoss()
    lossAvger = MovingAverage(100)

    train_writer = [SummaryWriter(os.path.join(args.tensorboard_runs, f"train\\level_{i}")) for i in range(net.levels)]
    test_writer = [SummaryWriter(os.path.join(args.tensorboard_runs, f"test\\level_{i}")) for i in range(net.levels)]

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
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        optimizer.param_groups[0]['lr'] = 1e-6

    # bestEvalLoss = float('inf')
    bestEvalLoss = 7.7496
    print(f"Start(Continue) to train at epoch {last_epoch}, iterations {iterations} !")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        net.train()
        device = next(net.parameters()).device

        for i, (x, qmap) in enumerate(train_dataloader):
            x = x.to(device)
            qmap = qmap.to(device)
            lmbda_scale_map = qmap2lsmap(qmap)

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            s = random.randint(0, net.levels - 1) # 随机数取[0, levels-1]两边都取得到
            sample = random.random()
            if sample < 0.15:
                s = 0
            elif sample < 0.20:
                s = 1
            # elif sample < 0.25:
            #     s = 2
            out_net = net(x, s, qmap)
            out_criterion = criterion(out_net, x, net.lmbda[s]*lmbda_scale_map)
            if out_criterion['loss'].isnan().any() or out_criterion['loss'].isinf().any() or out_criterion[
                'loss'] > 10:
                print(f"skip invalid loss! level:{s} loss:{out_criterion['loss']}")
                continue
            loss_avg = lossAvger.next(out_criterion["loss"].item())
            loss_scale = loss_avg / out_criterion["loss"].item()
            (loss_scale * out_criterion["loss"]).backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_max_norm)
            optimizer.step()
            aux_loss = net.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            if i % 300 == 0:
                print(
                    f"Train epoch {epoch}: ["
                    f"{i:05d}/{len(train_dataloader)}"
                    f" ({100. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.4f} |'
                    f'\ts: {s} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.4f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.4f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
                train_writer[s].add_scalar(f"Train/Total_Loss", out_criterion["loss"].item(), iterations)
                train_writer[s].add_scalar(f"Train/Bpp_Loss", out_criterion["bpp_loss"].item(), iterations)
                train_writer[s].add_scalar(f"Train/MSE_Loss", out_criterion["mse_loss"].item(), iterations)
                train_writer[s].add_scalar("aux_loss", aux_loss, iterations)

            if iterations % 10000 == 0 and iterations != 0:
                print("testing...")
                test_out = test_epoch(iterations, test_dataloaders, net, criterion)
                eval_total_loss = 0
                for eval_level in range(net.levels):
                    test_writer[eval_level].add_scalar("eval_loss", test_out["loss"][eval_level].avg, iterations)
                    test_writer[eval_level].add_scalar("bpp_loss", test_out["bpp_loss"][eval_level].avg, iterations)
                    test_writer[eval_level].add_scalar("mse_loss", test_out["mse_loss"][eval_level].avg, iterations)
                    train_writer[eval_level].add_scalar("LearningRate/lr", optimizer.param_groups[0]['lr'], iterations)
                    eval_total_loss += test_out["loss"][eval_level].avg

                if eval_total_loss < bestEvalLoss:
                    print(f"Saving best eval checkpoint epoch:{epoch}!!")
                    bestEvalLoss = eval_total_loss
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "iterations": iterations,
                            "state_dict": net.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        },
                        filename=args.model_save + f"\\scgainscalehp_best_checkpoint_epoch{epoch}.pth"
                    )

                lr_scheduler.step(eval_total_loss)
                print(f"Iterations:{iterations}  Eval Loss:{eval_total_loss:.4f} Learning rate: {optimizer.param_groups[0]['lr']}")


            iterations = iterations + 1

        if epoch % 1 == 0:
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
                filename=args.model_save + f"\\scgainscalehp_checkpoint_epoch{epoch}.pth"
            )


if __name__ == "__main__":
    train(sys.argv[1:])
    # net = SCGainedMSHyperprior()
    # parameters = set(
    #     n
    #     for n, p in net.named_parameters()
    #     if not n.endswith(".quantiles") and p.requires_grad
    # )
    # aux_parameters = set(
    #     n
    #     for n, p in net.named_parameters()
    #     if n.endswith(".quantiles") and p.requires_grad
    # )
    #
    # # Make sure we don't have an intersection of parameters
    # params_dict = dict(net.named_parameters())
    # inter_params = parameters & aux_parameters
    # union_params = parameters | aux_parameters
    #
    # assert len(inter_params) == 0
    # assert len(union_params) - len(params_dict.keys()) == 0
    #
    # optimizer = optim.Adam(
    #     (params_dict[n] for n in sorted(list(parameters))),
    #     lr=1e-4,
    # )
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    # for epoch in range(100):
    #     # train(...)
    #     # validate(...)
    #     scheduler.step()
    #     print(epoch, optimizer.param_groups[0]['lr'])

    # net = SCGainedMSHyperprior()
    # netstatedict = net.state_dict()
    # print(netstatedict.keys())
    # checkpoint = torch.load("D:\MXH\STPM\CompressAI\\trainmeanscale\GainedMSHP\models_bottleneckgained\\checkpoint_gainmshp_epoch90.pth", map_location="cpu")
    # statedict = checkpoint["state_dict"]
    # print(statedict.keys())

    # net = SCGainedMSHyperprior()
    # args = parse_args(sys.argv[1:])
    # train_dataloader, test_dataloaders = get_dataloader(args, L=5)
    # for i, (images, Qmap) in enumerate(train_dataloader):
    #     lmbdamap = quality2lambda(Qmap)
    #     level = random.randint(0,net.levels)
    #     print(f"idx:{i}, size:{images.size()}, Qmap:{Qmap.size()}")
    #     out = net(images, level, Qmap)
    #     print(f"y:{out['y'].size()}  x_hat:{out['x_hat']}")
    #
    # images = torch.rand([16,3,256,256])
    # Qmap = torch.rand([16, 1, 256, 256])
    # level = random.randint(0, net.levels)
    # print(f"idx:{0}, size:{images.size()}, Qmap:{Qmap.size()}")
    # out = net(images, level, Qmap)
    # print(f"y:{out['y'].size()}  x_hat:{out['x_hat'].size()}")
