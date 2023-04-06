import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(
    min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS
):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels)) # 为什么要先ln再求e次方，是为了更高的精度吗？


class ResBlock(nn.Module):
    def __init__(self, input_channels):
        super(ResBlock, self).__init__()

        self.channels = input_channels

        self.block = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, stride=(1, 1), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels, self.channels, stride=(1, 1), kernel_size=3, padding=1),
        )

    def forward(self, x):
        identity_map = x
        res = self.block(x)

        return torch.add(res, identity_map)


class NonLocalAttention(nn.Module):
    def __init__(self, input_channels):
        super(NonLocalAttention, self).__init__()

        self.channels = input_channels

        self.resBlock1_trunk = ResBlock(input_channels)
        self.resBlock2_trunk = ResBlock(input_channels)
        self.resBlock3_trunk = ResBlock(input_channels)

        self.resBlock1_attention = ResBlock(input_channels)
        self.resBlock2_attention = ResBlock(input_channels)
        self.resBlock3_attention = ResBlock(input_channels)
        self.activate_attention = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, stride=(1, 1), kernel_size=1, padding=0),
            nn.Sigmoid(),
        )


    def forward(self, x):

        trunk_branch = self.resBlock1_trunk(x)
        trunk_branch = self.resBlock2_trunk(trunk_branch)
        trunk_branch = self.resBlock3_trunk(trunk_branch)

        attention_branch = self.resBlock1_attention(x)
        attention_branch = self.resBlock2_attention(attention_branch)
        attention_branch = self.resBlock3_attention(attention_branch)
        attention_branch = self.activate_attention(attention_branch)

        out = x + trunk_branch * attention_branch

        return x


def UpConv2d(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


class SFT(nn.Module):
    def __init__(self, x_nc, prior_nc=1, ks=3, nhidden=128):
        super().__init__()
        pw = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(prior_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, x_nc, kernel_size=ks, padding=pw)

    def forward(self, x, qmap):
        qmap = F.adaptive_avg_pool2d(qmap, x.size()[2:])
        actv = self.mlp_shared(qmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = x * (1 + gamma) + beta

        return out