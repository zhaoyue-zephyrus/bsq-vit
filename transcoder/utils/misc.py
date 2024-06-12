import math
import moviepy.editor as mpy
import sys
import torch
from collections import OrderedDict
from PIL import Image


def check_loss_nan(loss):
    if not math.isfinite(loss.item()):
        print("Loss is {}, stopping training".format(loss.item()))
        sys.exit(1)


def get_metrics_dict(iter, meters):
    metrics = OrderedDict()
    for name, meter in meters.items():
        metrics[name] = meter.val
    metrics['iter'] = iter
    return metrics


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm = (p.grad.data ** 2).sum()
    total_norm = total_norm ** (1. / 2)
    return total_norm


def tensor2image(tensor, zero_mean=True):
    rescale_tensor = ((tensor + 1) * 127.5) if zero_mean else (tensor.detach() * 255)
    image_tensor = rescale_tensor.clamp_(0, 255).to("cpu", torch.uint8).numpy()
    return Image.fromarray(image_tensor)

def tensor2video(tensor, zero_mean=True):
    rescale_tensor = ((tensor + 1) * 127.5) if zero_mean else (tensor.detach() * 255)
    video_tensor = rescale_tensor.clamp_(0, 255).to("cpu", torch.uint8).numpy()
    return video_tensor

def grad_inspect_hook(grad, name=None, enabled=True, terminate=False):
    if enabled:
        print(f"grad abs mean of {name} is {grad.abs().mean()}")
        if terminate:
            exit()

def save_video(filename, tensor, fps=24):
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)
    clip.write_videofile(filename)
