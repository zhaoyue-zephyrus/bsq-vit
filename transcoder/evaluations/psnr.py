import torch


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2, dim=[1, 2, 3]):

        mse = torch.mean((img1 - img2) ** 2, dim=dim)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


def get_psnr(x_input, x_recon, zero_mean=False, is_video=False):
    if zero_mean:
        x_input_0_255 = ((x_input + 1) * 127.5)
        x_recon_0_255 = ((x_recon + 1) * 127.5)
    else:
        x_input_0_255 = x_input * 255
        x_recon_0_255 = x_recon * 255
    if is_video:
        psnr = PSNR()(x_input_0_255, x_recon_0_255, dim=[1,2,3,4])
    else:
        psnr = PSNR()(x_input_0_255, x_recon_0_255)
    return psnr
