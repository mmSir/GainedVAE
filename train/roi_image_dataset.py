import pandas as pd
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import hflip, to_tensor
from torch.distributions.multivariate_normal import MultivariateNormal

from compressai.datasets import ImageFolder

class QualityMapDataset(Dataset):
    def __init__(self, root, cropsize=256, mode='train', level_range=(0, 100), level=0, p=0.4):
        splitdir = Path(root) / mode  # root/train or root/test

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.paths = [f for f in splitdir.iterdir() if f.is_file()]
        self.map_paths = None # for dataset contains segment map. mxh set to None.
        self.cropsize = cropsize
        self.mode = mode
        self.level_range = level_range
        self.level = level # use when mode='test'. set level for uniform qmap mode
        self.p = p # use when mode='train'. probability for choosing different qmap mode
        self.grid = self._get_grid((self.cropsize, cropsize))

        assert self.map_paths is None or len(self.paths) == len(self.map_paths)
        assert level_range[0] == 0 and level_range[1] == 100
        if self.mode == 'train':
            print(f'[{mode}set] {len(self.paths)} images')
        elif self.mode == 'test':
            print(f'[{mode}set] {len(self.paths)} images for quality {level/100}')
            self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def _get_crop_params(self, img):
        w, h = img.size
        if w == self.cropsize and h == self.cropsize:
            return 0, 0, h, w

        if self.mode == 'train':
            top = random.randint(0, h - self.cropsize)
            left = random.randint(0, w - self.cropsize)
        else:
            # center
            top = int(round((h - self.cropsize) / 2.))
            left = int(round((w - self.cropsize) / 2.))
        return top, left

    def _get_grid(self, size):
        x1 = torch.tensor(range(size[0]))
        x2 = torch.tensor(range(size[1]))
        grid_x1, grid_x2 = torch.meshgrid(x1, x2)

        grid1 = grid_x1.view(size[0], size[1], 1)
        grid2 = grid_x2.view(size[0], size[1], 1)
        grid = torch.cat([grid1, grid2], dim=-1)
        return grid

    def __getitem__(self, idx):
        '''
            mxh accelerate version:
                1. no crop\flip operation. (move to preprocess)
                2. no seg map process
        '''
        img = Image.open(self.paths[idx]).convert('RGB')
        qmap = np.zeros(img.size[::-1], dtype=float)

        '''
        # crop if training
        if self.mode == 'train':
            top, left = self._get_crop_params(img)
            region = (left, top, left + self.cropsize, top + self.cropsize)
            img = img.crop(region)

        # horizontal flip
        if random.random() < 0.5 and self.mode == 'train':
            img = hflip(img)
        '''

        # random rate for each class
        if self.mode == 'train':
            sample = random.random()
            if sample < self.p:
                # uniform
                if random.random() < 0.01:
                    qmap[:] = 0
                else:
                    qmap[:] = (self.level_range[1] + 1) * random.random()
            elif sample < 2 * self.p:
                # gradation between two levels
                v1 = random.random() * self.level_range[1]
                v2 = random.random() * self.level_range[1]
                qmap = np.tile(np.linspace(v1, v2, self.cropsize), (self.cropsize, 1)).astype(float)
                if random.random() < 0.5:
                    qmap = qmap.T # 水平垂直翻转
            else:
                # gaussian kernel
                gaussian_num = int(1 + random.random() * 20)
                for i in range(gaussian_num):
                    mu_x = self.cropsize * random.random()
                    mu_y = self.cropsize * random.random()
                    var_x = 2000 * random.random() + 1000
                    var_y = 2000 * random.random() + 1000

                    m = MultivariateNormal(torch.tensor([mu_x, mu_y]), torch.tensor([[var_x, 0], [0, var_y]]))
                    p = m.log_prob(self.grid)
                    kernel = torch.exp(p).numpy()
                    qmap += kernel
                qmap *= 100 / qmap.max() * (0.5 * random.random() + 0.5)
        else: # mode == 'test'
            if self.level == -100:
                w, h = img.size
                # gradation
                if idx % 3 == 0:
                    v1 = idx/len(self.paths) * self.level_range[1]
                    v2 = (1-idx/len(self.paths)) * self.level_range[1]
                    qmap = np.tile(np.linspace(v1, v2, w), (h, 1)).astype(float)
                # gaussian kernel
                else:
                    gaussian_num = 1
                    for i in range(gaussian_num):
                        mu_x = h / 4 + (h/2)*idx/len(self.paths)
                        mu_y = w / 4 + (w/2)*(1-idx/len(self.paths))
                        var_x = 20000 * (1-idx/len(self.paths)) + 5000
                        var_y = 20000 * idx/len(self.paths) + 5000

                        m = MultivariateNormal(torch.tensor([mu_x, mu_y]), torch.tensor([[var_x, 0], [0, var_y]]))
                        grid = self._get_grid((h, w))
                        p = m.log_prob(grid)
                        kernel = torch.exp(p).numpy()
                        qmap += kernel
                    qmap *= 100 / qmap.max() * (0.4 * idx/len(self.paths) + 0.6)
            else:
                # uniform level
                qmap[:] = self.level

        # to tensor
        img = to_tensor(img)
        qmap = torch.FloatTensor(qmap).unsqueeze(dim=0)
        qmap *= 1 / self.level_range[1]  # 0~100 -> 0~1

        return img, qmap


def get_dataloader(args, stage, L=5):
    p = 1
    if stage == 1:
        p = 1
    elif stage == 2:
        p = 0.5
    else:
        p = 0.3

    train_dataset = QualityMapDataset(args.dataset, args.crop_size, mode='train', p=p)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=(args.cuda and torch.cuda.is_available()))
    levels = [-100] + [int(100*(i/L)) for i in range(L+1)] # level=-100 will random generate qmap, other positive levels will generate uniform map
    test_dataloaders = []
    for level in levels:
        test_dataset = QualityMapDataset(args.dataset, args.crop_size, mode='test', p=p, level=level)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=(args.cuda and torch.cuda.is_available()))
        test_dataloaders.append(test_dataloader)

    return train_dataloader, test_dataloaders


def get_test_dataloader_compressai(config):
    test_dataset = QualityMapDataset(config['testset'], mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=config['batchsize_test'], shuffle=False,
                                 num_workers=2)

    return test_dataloader
