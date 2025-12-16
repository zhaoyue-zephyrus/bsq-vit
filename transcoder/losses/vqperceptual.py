from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from transcoder.losses.lpips import LPIPS
from transcoder.models.discriminator.patchgan import PatchGANDiscriminator, weights_init
from transcoder.models.discriminator.stylegan import StyleGANDiscriminator
from transcoder.models.discriminator.stylegan3d import StyleGAN3DDiscriminator
from transcoder.models.stylegan_utils.ops.conv2d_gradfix import no_weight_gradients


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def hinge_g_loss(logits_fake):
    return -torch.mean(logits_fake)


def vanilla_g_loss(logits_fake):
    return torch.mean(F.softplus(-logits_fake))

def d_r1_loss(logits_real, img_real):
    with no_weight_gradients():
        grad_real, = torch.autograd.grad(
            outputs=logits_real.sum(), inputs=img_real,
            allow_unused=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty



class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_type='patchgan',
                 disc_input_size=256,
                 disc_channel_base=32768,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 disc_reg_freq=0, disc_reg_r1=10,
                 reconstruct_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", skip_disc=False, use_bf16=False, 
                 use_adaptive_disc_weight=True,
                 zero_mean=False, num_frames=1,
                 codebook_rampup_multiplier=1.0, codebook_rampup_steps=0,
                 **kwargs):
        """
        Inputs:
            - disc_start: int, the global step at which the discriminator starts to be trained
            - codebook_weight: float, the weight of the codebook loss
            - perceptual_weight: float, the weight of the perceptual loss
            - disc_weight: float, the weight of the discriminator loss
            - disc_factor: {0, 1} whether to mask the discriminator loss
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.codebook_rampup_multiplier = codebook_rampup_multiplier
        self.codebook_rampup_steps = codebook_rampup_steps
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS(zero_mean=zero_mean).eval()
        self.reconstruct_weight = reconstruct_weight
        self.perceptual_weight = perceptual_weight
        self.adaptive_dweight = use_adaptive_disc_weight
        self.disc_type = disc_type
        self.num_frames = num_frames

        if disc_type.lower() == 'patchgan':
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels,
                n_layers=disc_num_layers,
                use_actnorm=use_actnorm,
                ndf=disc_ndf,
            ).apply(weights_init)
        elif disc_type.lower() == 'stylegan':
            self.discriminator = StyleGANDiscriminator(
                0, disc_input_size, disc_in_channels,
                channel_base=disc_channel_base,
                num_fp16_res=8 if use_bf16 else 0,    # 8 is sufficiently large to cover all res
                epilogue_kwargs={'mbstd_group_size': 4},
            )
        elif disc_type.lower() == 'stylegan3d':
            self.discriminator = StyleGAN3DDiscriminator(
                num_frames, disc_input_size,
                video_channels=disc_in_channels,
            )
        else:
            raise ValueError(f"Unsupported disc_type {disc_type}")
        self.discriminator_iter_start = disc_start
        self.skip_disc_before_start = skip_disc
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
            self.gen_loss = hinge_g_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
            self.gen_loss = vanilla_g_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        self.disc_reg_freq = disc_reg_freq
        self.disc_reg_r1 = disc_reg_r1

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        if inputs.ndim == 5:
            assert self.num_frames == inputs.shape[2], f"Number of frames does not match input "
            inputs = rearrange(inputs, 'n c t h w -> (n t) c h w')
            reconstructions = rearrange(reconstructions, 'n c t h w -> (n t) c h w')
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        else:
            p_loss = torch.zeros_like(rec_loss)

        nll_loss = self.reconstruct_weight * rec_loss + self.perceptual_weight * p_loss
        nll_loss = torch.mean(nll_loss)

        if global_step < self.discriminator_iter_start and self.skip_disc_before_start:
            # before the discriminator joins the party, we only care about the reconstruction loss
            # no need to run the discriminator if it does not update anything
            loss = nll_loss + self.codebook_weight * codebook_loss
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean()
                   }
            return nll_loss, log
        else:
            disc_factor = 1.0 if global_step >= self.discriminator_iter_start else 0.

        # now the GAN part
        
        if optimizer_idx == 0:
            # generator update
            if self.disc_type == 'stylegan3d':
                reconstructions = rearrange(reconstructions, '(n t) c h w -> n c t h w', t=self.num_frames)
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = self.gen_loss(logits_fake)

            try:
                if self.adaptive_dweight:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                else:
                    d_weight = torch.tensor(self.discriminator_weight)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            if self.codebook_rampup_steps > 0:
                rampup_rate =  min(self.codebook_rampup_steps, global_step) / self.codebook_rampup_steps
                cb_weight = self.codebook_weight * (1.0 * rampup_rate  + self.codebook_rampup_multiplier * (1 - rampup_rate))
            else:
                cb_weight = self.codebook_weight
            loss = nll_loss + d_weight * disc_factor * g_loss + cb_weight * codebook_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if self.disc_type == 'stylegan3d':
                inputs = rearrange(inputs, '(n t) c h w -> n c t h w', t=self.num_frames)
                reconstructions = rearrange(reconstructions, '(n t) c h w -> n c t h w', t=self.num_frames)
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }

            if self.disc_reg_freq > 0 and (global_step + 1) % self.disc_reg_freq == 0:
                inputs.requires_grad = True
                logits_real = self.discriminator(inputs.contiguous())
                r1_loss = d_r1_loss(logits_real, inputs)
                r1_loss_scale = self.disc_reg_r1 / 2 * r1_loss * self.disc_reg_freq
                d_loss = d_loss + r1_loss_scale
                log.update({
                    "{}/disc_r1_loss".format(split): r1_loss.detach().mean(),
                    "{}/disc_r1_loss_scale".format(split): r1_loss_scale.detach().mean(),
                })

            return d_loss, log
