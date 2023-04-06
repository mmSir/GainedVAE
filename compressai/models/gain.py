import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

# pylint: disable=E0611,E0401
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d

from .utils import conv, deconv, update_registered_buffers
from .priors import CompressionModel
from .gain_utils import get_scale_table, ResBlock, NonLocalAttention, SFT, UpConv2d
# pylint: enable=E0611,E0401



class GainedScaleHyperprior(CompressionModel):
    '''
    Bottleneck scaling version.
    '''
    def __init__(self, N = 192, M = 320, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)

        self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...

        # Condition on Latent y, so the gain vector length M
        # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
        # treat all channels the same in initialization
        self.levels = len(self.lmbda) # 8
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)


    def forward(self, x, s):
        '''
            x: input image
            s: random num to choose gain vector
        '''
        y = self.g_a(x)
        y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]
        z = self.h_a(y)
        z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat)

        return {
            "y": y,
            "y_hat": y_hat,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, s, l):
        assert s in range(0,self.levels-1), f"s should in range(0, {self.levels-1}), but get s:{s}"
        assert l >=0 and l <=1, "l should in [0,1]"
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        InterpolatedGain = torch.abs(self.Gain[s]).pow(1-l) * torch.abs(self.Gain[s+1]).pow(l)
        # InterpolatedInverseGain = self.InverseGain[s].pow(l) * self.InverseGain[s+1].pow(1-l)
        InterpolatedHyperGain = torch.abs(self.HyperGain[s]).pow(1-l) * torch.abs(self.HyperGain[s+1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        # Linear Interpolation can achieve the same result?
        # InterpolatedGain = torch.abs(self.Gain[s]) * (1 - l) + torch.abs(self.Gain[s + 1]) * l
        # InterpolatedHyperGain = torch.abs(self.HyperGain[s]) * (1 - l) + torch.abs(self.HyperGain[s + 1]) * l
        # InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]) * (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)

        y = self.g_a(x)
        ungained_y = y
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z = self.h_a(y)
        z = z * InterpolatedHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)  # y_string 是 list, 且只包含一个元素
        gained_y_hat = self.gaussian_conditional.quantize(y, "symbols")
        return {"strings": [y_strings, z_strings],
                "shape": z.size()[-2:],
                "ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": gained_y_hat}


    def decompress(self, strings, shape, s, l):
        assert isinstance(strings, list) and len(strings) == 2 # 保证有y和z
        assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        # InterpolatedGain = self.Gain[s].pow(l) * self.Gain[s + 1].pow(1 - l)
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1-l) * torch.abs(self.InverseGain[s+1]).pow(l)
        # InterpolatedHyperGain = self.HyperGain[s].pow(l) * self.HyperGain[s + 1].pow(1 - l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        # Linear Interpolation can achieve the same result
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]) * (1 - l) + torch.abs(self.InverseGain[s + 1]) * (l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])* (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes
        )
        gained_y_hat = y_hat
        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "gained_y_hat": gained_y_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def getY(self, x, s, l):
        assert s in range(0, self.levels - 1), f"s should in range(0, {self.levels - 1}), but get s:{s}"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        InterpolatedGain = torch.abs(self.Gain[s]).pow(1 - l) * torch.abs(self.Gain[s + 1]).pow(l)

        # 如果x不是64的倍数，就对x做padding
        h, w = x.size(2), x.size(3)
        p = 64  # maximum 6 strides of 2
        new_h = (h + p - 1) // p * p  # padding为64的倍数
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

        y = self.g_a(x_padded)
        ungained_y = y
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        y_quantized = self.gaussian_conditional.quantize(y, "noise")

        return {"ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": y_quantized}

        # return y, y_quantized

    def getScale(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        gaussian_params = self.h_s(z)
        scales, means = gaussian_params.chunk(2, 1)
        return scales

    def getX(self, y_hat, s, l):
        assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1 - l) * torch.abs(self.InverseGain[s + 1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1 - l) * torch.abs(
            self.InverseHyperGain[s + 1]).pow(l)

        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class GainedMSHyperprior(CompressionModel):
    '''
    Bottleneck scaling version.
    '''
    def __init__(self, N = 128, M = 192, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            # conv(N, N, stride=1, kernel_size=3),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            # conv(N, N, stride=1, kernel_size=3),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            # conv(N, N, stride=1, kernel_size=3),
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            # conv(M, M, stride=1, kernel_size=3),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.gaussian_conditional = GaussianConditional(None)

        self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...

        # Condition on Latent y, so the gain vector length M
        # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
        # treat all channels the same in initialization
        self.levels = len(self.lmbda) # 8
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)


    def forward(self, x, s):
        '''
            x: input image
            s: random num to choose gain vector
        '''
        y = self.g_a(x)
        y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]
        # y_hat = self.gaussian_conditional.quantize(y, "noise")
        z = self.h_a(y)
        z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat)

        return {
            "y": y,
            "y_hat": y_hat,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, s, l):
        assert s in range(0,self.levels-1), f"s should in range(0, {self.levels-1}), but get s:{s}"
        assert l >=0 and l <=1, "l should in [0,1]"
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        InterpolatedGain = torch.abs(self.Gain[s]).pow(1-l) * torch.abs(self.Gain[s+1]).pow(l)
        # InterpolatedInverseGain = self.InverseGain[s].pow(l) * self.InverseGain[s+1].pow(1-l)
        InterpolatedHyperGain = torch.abs(self.HyperGain[s]).pow(1-l) * torch.abs(self.HyperGain[s+1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        # Linear Interpolation can achieve the same result?
        InterpolatedGain = torch.abs(self.Gain[s]) * (1 - l) + torch.abs(self.Gain[s + 1]) * l
        InterpolatedHyperGain = torch.abs(self.HyperGain[s]) * (1 - l) + torch.abs(self.HyperGain[s + 1]) * l
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]) * (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)

        y = self.g_a(x)
        ungained_y = y
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # y_hat = self.gaussian_conditional.quantize(y, "symbols").type(torch.cuda.FloatTensor)
        z = self.h_a(y)
        z = z * InterpolatedHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)  # y_string 是 list, 且只包含一个元素
        gained_y_hat = self.gaussian_conditional.quantize(y, "symbols", means_hat)
        return {"strings": [y_strings, z_strings],
                "shape": z.size()[-2:],
                "ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": gained_y_hat}

    def decompress(self, strings, shape, s, l):
        assert isinstance(strings, list) and len(strings) == 2 # 保证有y和z
        assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        # InterpolatedGain = self.Gain[s].pow(l) * self.Gain[s + 1].pow(1 - l)
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1-l) * torch.abs(self.InverseGain[s+1]).pow(l)
        # InterpolatedHyperGain = self.HyperGain[s].pow(l) * self.HyperGain[s + 1].pow(1 - l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        # Linear Interpolation can achieve the same result
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]) * (1 - l) + torch.abs(self.InverseGain[s + 1]) * (l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])* (1 - l) + torch.abs(self.InverseHyperGain[s + 1]) * (l)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        gained_y_hat = y_hat
        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "gained_y_hat": gained_y_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def getY(self, x, s, l):
        assert s in range(0, self.levels - 1), f"s should in range(0, {self.levels - 1}), but get s:{s}"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        InterpolatedGain = torch.abs(self.Gain[s]).pow(1 - l) * torch.abs(self.Gain[s + 1]).pow(l)

        # 如果x不是64的倍数，就对x做padding
        h, w = x.size(2), x.size(3)
        p = 64  # maximum 6 strides of 2
        new_h = (h + p - 1) // p * p  # padding为64的倍数
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

        y = self.g_a(x_padded)
        ungained_y = y
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        y_quantized = self.gaussian_conditional.quantize(y, "noise")

        return {"ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": y_quantized}

        # return y, y_quantized

    def getScale(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        gaussian_params = self.h_s(z)
        scales, means = gaussian_params.chunk(2, 1)
        return scales

    def getX(self, y_hat, s, l):
        assert s in range(0, self.levels - 1), f"s should in range(0,{self.levels - 1})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1 - l) * torch.abs(self.InverseGain[s + 1]).pow(l)
        InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1 - l) * torch.abs(
            self.InverseHyperGain[s + 1]).pow(l)

        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


class SCGainedMSHyperprior(CompressionModel):
    '''
    Bottleneck scaling version.
    modulate in both spatial and channel dim
    '''
    def __init__(self, N = 128, M = 192, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.qmap_feature_ga0 = nn.Sequential(
            conv(4, N * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(N * 2, N, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 3, 1)
        )
        self.qmap_feature_ga1 = nn.Sequential(
            conv(N, N, 3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 1, 1)
        )
        self.ga_SFT1 = SFT(N, N, ks=3)
        self.qmap_feature_ga2 = nn.Sequential(
            conv(N, N, 3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 1, 1)
        )
        self.ga_SFT2 = SFT(N, N, ks=3)
        self.qmap_feature_ga3 = nn.Sequential(
            conv(N, N, 3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 1, 1)
        )
        self.ga_SFT3 = SFT(N, N, ks=3)

        # self.g_a = None
        self.g_a1 = nn.Sequential(
            conv(3, N),
            GDN(N),
        )
        self.g_a2 = nn.Sequential(
            conv(N, N),
            GDN(N),
        )
        self.g_a3 = nn.Sequential(
            conv(N, N),
            GDN(N),
        )
        self.g_a4 = nn.Sequential(
            conv(N, M),
        )

        # f_c
        self.qmap_feature_generation = nn.Sequential(
            UpConv2d(N, N // 2, 3),
            nn.LeakyReLU(0.1, True),
            UpConv2d(N // 2, N // 4),
            nn.LeakyReLU(0.1, True),
            conv(N // 4, N // 4, 3, 1)
        )

        self.qmap_feature_gs0 = nn.Sequential(
            conv(M + N // 4, N * 4, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(N * 4, N * 2, 3, 1),
            nn.LeakyReLU(0.1, True),
            conv(N * 2, N, 3, 1)
        )
        self.gs_SFT0 = SFT(M, N, ks=3)
        self.qmap_feature_gs1 = nn.Sequential(
            UpConv2d(N, N, 3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 1, 1)
        )
        self.gs_SFT1 = SFT(N, N, ks=3)
        self.qmap_feature_gs2 = nn.Sequential(
            UpConv2d(N, N, 3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 1, 1)
        )
        self.gs_SFT2 = SFT(N, N, ks=3)
        self.qmap_feature_gs3 = nn.Sequential(
            UpConv2d(N, N, 3),
            nn.LeakyReLU(0.1, True),
            conv(N, N, 1, 1)
        )
        self.gs_SFT3 = SFT(N, N, ks=3)


        # self.g_s = None
        self.g_s1 = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
        )
        self.g_s2 = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
        )
        self.g_s3 = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
        )
        self.g_s4 = nn.Sequential(
            deconv(N, 3),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.gaussian_conditional = GaussianConditional(None)

        self.lmbda = [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001, 0.0003]  # mxh add from HUAWEI CVPR2021 Gained...

        # Condition on Latent y, so the gain vector length M
        # e.g.: self.levels = 6 means we have 6 pairs gain vectors corresponding to 6 level RD performance
        # treat all channels the same in initialization
        self.levels = len(self.lmbda) # 8
        self.Gain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.InverseGain = torch.nn.Parameter(torch.ones(size=[self.levels, M]), requires_grad=True)
        self.HyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)
        self.InverseHyperGain = torch.nn.Parameter(torch.ones(size=[self.levels, N]), requires_grad=True)

    def g_a(self, x, qmap):
        qmap = self.qmap_feature_ga0(torch.cat([qmap, x], dim=1))
        qmap = self.qmap_feature_ga1(qmap)
        x = self.g_a1(x)
        x = self.ga_SFT1(x, qmap)

        qmap = self.qmap_feature_ga2(qmap)
        x = self.g_a2(x)
        x = self.ga_SFT2(x, qmap)

        qmap = self.qmap_feature_ga3(qmap)
        x = self.g_a3(x)
        x = self.ga_SFT3(x, qmap)

        x = self.g_a4(x)

        return x

    def g_s(self, x, z):
        w = self.qmap_feature_generation(z)
        w = self.qmap_feature_gs0(torch.cat([w, x], dim=1))
        x = self.gs_SFT0(x, w)

        w = self.qmap_feature_gs1(w)
        x = self.g_s1(x)
        x = self.gs_SFT1(x, w)

        w = self.qmap_feature_gs2(w)
        x = self.g_s2(x)
        x = self.gs_SFT2(x, w)

        w = self.qmap_feature_gs3(w)
        x = self.g_s3(x)
        x = self.gs_SFT3(x, w)

        x = self.g_s4(x)

        return x

    def forward(self, x, s, qmap):
        '''
            x: input image
            s: random num to choose gain vector
            qmap: spatial quality map
        '''
        y = self.g_a(x, qmap)
        y = y * torch.abs(self.Gain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # Gain[s]: [M]  -->  [1,M,1,1]
        z = self.h_a(y)
        z = z * torch.abs(self.HyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        z_hat = z_hat * torch.abs(self.InverseHyperGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = y_hat * torch.abs(self.InverseGain[s]).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat, z_hat)

        return {
            "y": y,
            "y_hat": y_hat,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, s, l, qmap):
        assert s in range(0,self.levels), f"s should in range(0, {self.levels}), but get s:{s}"
        assert l >=0 and l <=1, "l should in [0,1]"
        assert s + l <= self.levels-1
        # l = 0, Interpolated = s+1; l = 1, Interpolated = s
        if s == self.levels-1:
            InterpolatedGain = torch.abs(self.Gain[s])
            InterpolatedHyperGain = torch.abs(self.HyperGain[s])
            InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])
        else:
            InterpolatedGain = torch.abs(self.Gain[s]).pow(1-l) * torch.abs(self.Gain[s+1]).pow(l)
            InterpolatedHyperGain = torch.abs(self.HyperGain[s]).pow(1-l) * torch.abs(self.HyperGain[s+1]).pow(l)
            InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        y = self.g_a(x, qmap)
        ungained_y = y
        y = y * InterpolatedGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # y_hat = self.gaussian_conditional.quantize(y, "symbols").type(torch.cuda.FloatTensor)
        z = self.h_a(y)
        z = z * InterpolatedHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        gained_y_hat = self.gaussian_conditional.quantize(y, "symbols", means_hat)
        return {"strings": [y_strings, z_strings],
                "shape": z.size()[-2:],
                "ungained_y": ungained_y,
                "gained_y": y,
                "gained_y_hat": gained_y_hat}

    def decompress(self, strings, shape, s, l):
        assert isinstance(strings, list) and len(strings) == 2 # 保证有y和z
        assert s in range(0, self.levels), f"s should in range(0,{self.levels})"
        assert l >= 0 and l <= 1, "l should in [0,1]"
        assert s + l <= self.levels-1
        if s == self.levels-1:
            InterpolatedInverseGain = torch.abs(self.InverseGain[s])
            InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s])
        else:
            InterpolatedInverseGain = torch.abs(self.InverseGain[s]).pow(1-l) * torch.abs(self.InverseGain[s+1]).pow(l)
            InterpolatedInverseHyperGain = torch.abs(self.InverseHyperGain[s]).pow(1-l) * torch.abs(self.InverseHyperGain[s+1]).pow(l)

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat = z_hat * InterpolatedInverseHyperGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        gained_y_hat = y_hat
        y_hat = y_hat * InterpolatedInverseGain.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x_hat = self.g_s(y_hat, z_hat).clamp_(0, 1)
        return {"x_hat": x_hat, "gained_y_hat": gained_y_hat}

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated