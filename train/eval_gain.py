import argparse
import json
import math
import os
import sys
import time
import random
import numpy as np

import cv2


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from pathlib import Path

import compressai
from compressai.zoo import models
from compressai.models.gain import GainedMSHyperprior, SCGainedMSHyperprior

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TEST_MODES = ['gain', 'scgain']
QMAP_TESTMODE_MAX = 7


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)

def getQmap(shape, mode):
    assert mode >= 0 and mode <= QMAP_TESTMODE_MAX
    qmap = np.zeros(shape, dtype=float)
    uniform_step = 0.2
    if mode <= 5:
        qmap[:] = uniform_step * mode
    elif mode == 6:
        qmap = np.tile(np.linspace(0, 1, shape[1]), (shape[0], 1)).astype(float)
    elif mode == 7:
        qmap = np.tile(np.linspace(0, 1, shape[0]), (1, shape[1])).astype(float)
    else:
        raise ValueError(f"Undefined Qmap Test Mode? get mode:{mode}")

    qmap = torch.FloatTensor(qmap).unsqueeze(dim=0)
    return qmap

@torch.no_grad()
def inference_scgain(model, x, s, qmap):
    x = x.unsqueeze(0) # 增加batch维度

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p # padding为64的倍数
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model.compress(x_padded, s=s, l=0, qmap=qmap)
    out_forward = model(x_padded, s=s, qmap=qmap)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], s=s, l=0)
    y_quantized = out_dec["gained_y_hat"]  # gained_y_quantized
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    estimate_bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                       for likelihoods in out_forward["likelihoods"].values()).item()


    return {
        "x_hat": out_dec["x_hat"],
        "y_quantized": y_quantized,
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "estimate_bpp": estimate_bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_gain(model, x, s, l):
    x = x.unsqueeze(0) # 增加batch维度

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p # padding为64的倍数
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

    start = time.time()
    out_enc = model.compress(x_padded, s=s, l=l)
    out_forward = model(x_padded, s=s) if l != 1.0 else model(x_padded, s=s+1)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"], s=s, l=l)
    y_quantized = out_dec["gained_y_hat"]  # gained_y_quantized
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    estimate_bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                       for likelihoods in out_forward["likelihoods"].values()).item()


    return {
        "x_hat": out_dec["x_hat"],
        "y_quantized": y_quantized,
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "estimate_bpp": estimate_bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


def evalSCGain(model, dataset_path, logfile):
    device = next(model.parameters()).device
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise RuntimeError(f'Invalid directory "{dataset_path}"')

    for s in range(0, model.levels-1):
        for mode in range(0, QMAP_TESTMODE_MAX-1):
            print(f"--------------------Testing s:{s} mode:{mode}--------------------------")
            PSNR, MSSSIM, BPP, Esti_Bpp = [], [], [], []
            for img_path in dataset_path.iterdir():
                image = Image.open(img_path)
                qmap = getQmap(image.size[::-1], mode).unsqueeze(dim=0).to(device)
                image = transforms.ToTensor()(image).to(device)

                out = inference_scgain(model, image, s, qmap)

                PSNR.append(out["psnr"])
                MSSSIM.append(out["ms-ssim"])
                BPP.append(out["bpp"])
                Esti_Bpp.append(out["estimate_bpp"])

            logfile.write(
                f's={s} mode={mode}  '
                f'PSNR_AVE: {np.array(PSNR).mean():.3f}  '
                f'MS-SSIM_AVE: {np.array(MSSSIM).mean():.3f}  '
                f'BPP_AVE: {np.array(BPP).mean():.3f}  '
                f'Est_BPP_AVE: {np.array(Esti_Bpp).mean():.3f}\n'
            )


def evalGain(model, dataset_path, logfile):
    '''
        Eval for continuous variable rate model. Channel Gain Module support vbr capability.
    '''
    device = next(model.parameters()).device
    dataset_path = Path(dataset_path)
    if not dataset_path.is_dir():
        raise RuntimeError(f'Invalid directory "{dataset_path}"')

    l_step = 0.1
    for s in range(0, model.levels - 1):
        for l in np.arange(0.0, 1.0 + l_step, l_step):
            if l == 1.0 and s != model.levels - 2:
                continue
            print(f"--------------------Testing s:{s} l:{l:.2f}--------------------------")
            PSNR, MSSSIM, BPP, Esti_Bpp = [], [], [], []
            for img_path in dataset_path.iterdir():
                image = Image.open(img_path)
                image = transforms.ToTensor()(image).to(device)

                out = inference_gain(model, image, s=s, l=l)

                PSNR.append(out["psnr"])
                MSSSIM.append(out["ms-ssim"])
                BPP.append(out["bpp"])
                Esti_Bpp.append(out["estimate_bpp"])

            PSNR = np.array(PSNR)
            MSSSIM = np.array(MSSSIM)
            BPP = np.array(BPP)
            Esti_Bpp = np.array(Esti_Bpp)
            logfile.write(f's+l = {s+l:.2f}  ')
            logfile.write(f'PSNR_AVE: {PSNR.mean():.3f}  MS-SSIM_AVE: {MSSSIM.mean():.3f}  BPP_AVE: {BPP.mean():.3f} Est_BPP_AVE: {Esti_Bpp.mean():.3f}\n')


def eval(model, args):
    logfile = open(args.logpath, 'w+')
    print("Result Output Path:" + args.logpath)
    logfile.write("Eval Time: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "\n")
    logfile.write("Eval model:" + args.checkpoint + '\n')
    if args.mode == TEST_MODES[0]:
        evalGain(model, args.dataset, logfile)
    elif args.mode == TEST_MODES[1]:
        evalSCGain(model, args.dataset, logfile)
    logfile.close()


def parse_args(argv):
    parser = argparse.ArgumentParser(description="variable-rate image compression evaluation script.")
    parser.add_argument(
        "-m",
        "--model",
        default="mbt2018-mean",
        choices=models.keys(),
        help="I Frame Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="C:\\Users\Administrator\Dataset\ImageCompressionDataset\\test"
        , help="Evaluation dataset")
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument(
        "-md",
        "--mode",
        choices=TEST_MODES,
        default=TEST_MODES[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--checkpoint", type=str, help="Path to the test checkpoint", required=True)
    parser.add_argument("--logpath", type=str, help="Result Output Path", required=True)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.mode == TEST_MODES[0]:
        ImageCompressor = GainedMSHyperprior().to(device)
    elif args.mode == TEST_MODES[1]:
        ImageCompressor = SCGainedMSHyperprior().to(device)
    else:
        raise ValueError(f"Unsupported Eval Mode. Get mode:{args.mode}")

    # Load Model
    if args.checkpoint:  # load ImageCompressor
        print("Loading ImageCompressor ", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        ImageCompressor.load_state_dict(checkpoint["state_dict"])
        ImageCompressor.update(force=True)
        ImageCompressor.eval()

    compressai.set_entropy_coder(args.entropy_coder)

    start = time.time()
    eval(ImageCompressor, args=args)
    eval_time = time.time() - start
    print(f"eval time:{eval_time:.3f}s.")



if __name__ == "__main__":
    main(sys.argv[1:])




