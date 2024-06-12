from pytorch_msssim import ssim, ms_ssim
import torch

def get_ssim(x_input, x_recon, zero_mean=False, is_video=False):
    if zero_mean:
        x_input_0_255 = ((x_input + 1) * 127.5)
        x_recon_0_255 = ((x_recon + 1) * 127.5)
    else:
        x_input_0_255 = x_input * 255
        x_recon_0_255 = x_recon * 255
    if is_video:
        ssim_val = [
            ssim(x_input_0_255[:, :, t, :, :], x_recon_0_255[:, :, t, :, :], data_range=255, size_average=False)
            for t in range(x_input_0_255.shape[2])
        ]
        ssim_val = torch.stack(ssim_val).mean(0)
    else:
        ssim_val = ssim(x_input_0_255, x_recon_0_255, data_range=255, size_average=False)
    return ssim_val


def get_ssim_and_msssim(x_input, x_recon, zero_mean=False, is_video=False):
    if x_input.shape[2 + is_video] < 256 or x_input.shape[3 + is_video] < 256:
        ssim_val = get_ssim(x_input, x_recon, zero_mean, is_video)
        return ssim_val, torch.ones_like(ssim_val) * torch.nan
    if zero_mean:
        x_input_0_255 = ((x_input + 1) * 127.5)
        x_recon_0_255 = ((x_recon + 1) * 127.5)
    else:
        x_input_0_255 = x_input * 255
        x_recon_0_255 = x_recon * 255
    if is_video:
        ssim_val = [
            ssim(x_input_0_255[:, :, t, :, :], x_recon_0_255[:, :, t, :, :], data_range=255, size_average=False)
            for t in range(x_input_0_255.shape[2])
        ]
        ms_ssim_val = [
            ms_ssim(x_input_0_255[:, :, t, :, :], x_recon_0_255[:, :, t, :, :], data_range=255, size_average=False)
            for t in range(x_input_0_255.shape[2])
        ]
        ssim_val = torch.stack(ssim_val).mean(0)
        ms_ssim_val = torch.stack(ms_ssim_val).mean(0)
    else:
        ssim_val = ssim(x_input_0_255, x_recon_0_255, data_range=255, size_average=False)
        ms_ssim_val = ms_ssim(x_input_0_255, x_recon_0_255, data_range=255, size_average=False)
    return ssim_val, ms_ssim_val
