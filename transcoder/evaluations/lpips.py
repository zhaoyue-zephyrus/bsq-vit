from einops import rearrange
import lpips


def get_lpips(x_input, x_recon, zero_mean=False, network_type='alex', is_video=False):
    assert network_type in ['alex', 'vgg']
    if not zero_mean:
        x_input = x_input * 2 - 1
        x_recon = x_recon * 2 - 1
    loss_fn = lpips.LPIPS(net=network_type, verbose=False).to(x_input.device)
    if is_video:
        b = x_input.shape[0]
        x_input = rearrange(x_input, 'b c t h w -> (b t) c h w')
        x_recon = rearrange(x_recon, 'b c t h w -> (b t) c h w')
        d = loss_fn.forward(x_input, x_recon).squeeze()
        d = rearrange(d, '(b t) -> b t', b=b).mean(1)
    else:
        d = loss_fn.forward(x_input, x_recon)
    return d
