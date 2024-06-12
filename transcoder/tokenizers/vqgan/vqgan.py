import torch.nn as nn

from .utils import load_config, load_vqgan


class VQGAN(nn.Module):
    def __init__(self, config_path, ckpt_path, is_gumbel=False):
        super().__init__()
        config = load_config(config_path, display=False)
        self.model = load_vqgan(config, ckpt_path, is_gumbel=is_gumbel)

    def forward(self, x):
        z, _, [_, _, indices] = self.model.encode(x)
        x_rec = self.model.decode(z)
        return x_rec
