import torch
import torch.nn as nn
import torch.nn.functional as F

from dall_e import map_pixels, unmap_pixels, load_model


class Dalle(nn.Module):
    def __init__(self, model_rootpath=None):
        super().__init__()
        if model_rootpath is None:
            encoder_path = 'https://cdn.openai.com/dall-e/encoder.pkl'
            decoder_path = 'https://cdn.openai.com/dall-e/decoder.pkl'
        else:
            encoder_path = f"{model_rootpath}/encoder.pkl"
            decoder_path = f"{model_rootpath}/decoder.pkl"
        self.enc = load_model(encoder_path)
        self.dec = load_model(decoder_path)

    def forward(self, x):
        x = map_pixels(x)
        z_logits = self.enc(x)
        z = torch.argmax(z_logits, axis=1)
        z = F.one_hot(z, num_classes=self.enc.vocab_size).permute(0, 3, 1, 2).float()
        x_stats = self.dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return x_rec