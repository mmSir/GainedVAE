import torch
import torch.nn as nn
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