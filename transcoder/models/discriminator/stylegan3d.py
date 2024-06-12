
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union
from einops import rearrange

from transcoder.utils.misc import grad_inspect_hook
from transcoder.models.discriminator.stylegan import MinibatchStdLayer, FullyConnectedLayer

class ScaledLeakyReLU(nn.Module):
    """
    The scaled leaky ReLU activation function.
    This is a modified version of the leaky ReLU activation function, which scales the ouput
    by the gain parameter.
    """
    def __init__(self, negative_slope=0.2, gain=1):
        super().__init__()
        self.negative_slope = negative_slope
        self.gain = gain

    def forward(self, x):
        return F.leaky_relu_(x, negative_slope=self.negative_slope) * self.gain
    

class ScaledConv3d(nn.Conv3d):
    """
    A modified conv3d module that:
    - Initializes the weights with a normal distribution and use the scaled version of the weights in computation
    - The weight scaler is 1 / sqrt(in_channels * kernel_volume), in the 3d case kernel_volume = kernel_size^3
    - Initializes the bias with zeros if bias is True

    These operation means:
    - This op is similar to a conv3d that are initialized with a normal distribution with 1/sqrt(in_channels * kernel_volume) std.dev.
    - However, the learning rate on the conv3d parameters are scaled down by this std value. 
    - This leads to slow update of the discriminator and potentially avoids overfitting.

    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, gain=1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        ksize = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.weight_gain = gain / math.sqrt(in_channels * math.prod(ksize))
        nn.init.normal_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.weight_gain, self.bias)


class ResBlock(nn.Module):

    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: Union[int, tuple] =(3, 3, 3), 
                 padding: str = 'same',
                 stride: int = 1,
                 downsample_layer: Callable = nn.AvgPool3d):
        super().__init__()
        ksize = kernel_sizes if isinstance(kernel_sizes, tuple) else (kernel_sizes, kernel_sizes, kernel_sizes)

        self.function_path = nn.Sequential(
            ScaledConv3d(in_channels, out_channels, ksize, stride, padding),
            ScaledLeakyReLU(gain=2**0.5),
            downsample_layer(),
            ScaledConv3d(out_channels, out_channels, ksize, stride, padding),
            ScaledLeakyReLU()
        )

        self.skip_path = nn.Sequential(
            downsample_layer(),
            ScaledConv3d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x):
        residual = self.skip_path(x)
        x = self.function_path(x)
        # residual connection
        out = (residual + x) / torch.sqrt(torch.tensor(2.0))
        return out
    

class StyleGAN3DDiscriminator(nn.Module):

    def __init__(self, 
                 num_frames, input_res,
                 channel_list: list =[64, 128, 256, 256, 256, 256], 
                 linear_units: int =256,
                 kernel_sizes: int = 3,
                 video_channels: int = 3,
                 lrelu_leakage: float = 0.2,
                 spatial_downsample_factor: int = 2,
                 temporal_downsample_list: Union[int, list] = [1, 1, 2, 2, 2],
                 mbstd_group_size=4, mbstd_num_channels=1):
        super().__init__()

        self.channel_list = channel_list
        self.kernel_sizes = kernel_sizes
        self.num_frames = num_frames
        self.input_res = input_res

        assert len(temporal_downsample_list) == len(channel_list) - 1
        
        # input 
        assert len(channel_list) > 0
        self.input_conv = nn.Sequential(
                                    ScaledConv3d(video_channels, 
                                            channel_list[0], 
                                            1, 
                                            padding='same'),
                                    ScaledLeakyReLU(negative_slope=lrelu_leakage)
                                    )
        self.conv_blocks = nn.ModuleList()
        
        # intermediate layers   
        in_channel = channel_list[0]
        current_duration = self.num_frames
        current_res = input_res
        for out_channel, t_ds in zip(channel_list[1:], temporal_downsample_list):
            assert current_duration // t_ds > 0, "Inssufficient duration for temporal downsampling."
            ds_kernel = (t_ds, spatial_downsample_factor, spatial_downsample_factor)
            self.conv_blocks.append(ResBlock(in_channel, 
                                             out_channel, 
                                             kernel_sizes, 
                                             padding='same', 
                                             downsample_layer=lambda : nn.AvgPool3d(ds_kernel)))
            in_channel = out_channel
            current_duration = current_duration // t_ds
            current_res = current_res // spatial_downsample_factor
        
        # discriminator output
        self.mbstd = MinibatchStdLayer(mbstd_group_size, mbstd_num_channels)
        self.output_conv = nn.Sequential(
            ScaledConv3d(channel_list[-1] + mbstd_num_channels, channel_list[-1], 
                        kernel_sizes, 
                        padding='same'),
            ScaledLeakyReLU(negative_slope=lrelu_leakage),
        )
        
        self.output_mlp = nn.Sequential(
            nn.Flatten(),
            FullyConnectedLayer(current_duration * (current_res ** 2) * channel_list[-1], linear_units, activation='lrelu'),
            FullyConnectedLayer(linear_units, 1)
        )

    def forward(self, x):
        inspect=False
        if x.requires_grad:
            x.register_hook(functools.partial(grad_inspect_hook, name='input', enabled=inspect, terminate=True))
        x = self.input_conv(x)
        x.register_hook(functools.partial(grad_inspect_hook, name='input-conv', enabled=inspect, terminate=False))
        cnt = 0
        for block in self.conv_blocks:
            x = block(x)
            x.register_hook(functools.partial(grad_inspect_hook, name=f'conv-{cnt}', enabled=inspect, terminate=False))
            cnt += 1
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> b c (t h) w')
        x = self.mbstd(x)
        x = rearrange(x, 'b c (t h) w -> b c t h w', t=t)
        x = self.output_conv(x)
        x.register_hook(functools.partial(grad_inspect_hook, name='output-conv', enabled=inspect, terminate=False))
        x = self.output_mlp(x)
        return x