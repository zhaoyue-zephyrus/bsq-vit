import torch.nn as nn

from diffusers.models import AutoencoderKL


class SDXLTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

    def forward(self, x):
        output = self.model(x)
        return output.sample

class SD2Tokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

    def forward(self, x):
        output = self.model(x)
        return output.sample