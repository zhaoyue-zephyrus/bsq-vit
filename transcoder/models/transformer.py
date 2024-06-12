from collections import OrderedDict
from typing import Callable, Optional, Union
from einops import rearrange
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.layers import to_2tuple
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath

from transcoder.models.attention_mask import get_attention_mask


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            use_preln: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=attn_drop)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            # disable this following JAX implementation.
            # Reference: https://github.com/google-research/magvit/blob/main/videogvt/models/simplified_bert.py#L112
            # ("drop1", nn.Dropout(drop)),
            ("c_proj", nn.Linear(mlp_width, d_model)),
            ("drop2", nn.Dropout(drop)),
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.use_preln = use_preln

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask, is_causal=is_causal)[0]

    def checkpoint_forward(self, x: torch.Tensor, 
                           attn_mask: Optional[torch.Tensor] = None,
                           is_causal: bool = False):
        state = x
        if self.use_preln:
            x = checkpoint(self.ln_1, x, use_reentrant=False)
            x = self.attention(x, attn_mask, is_causal)
            x = checkpoint(self.ls_1, x, use_reentrant=False)
            state = state + self.drop_path(x)
            x = checkpoint(self.ln_2, state, use_reentrant=False)
            x = self.mlp(x)
            x = checkpoint(self.ls_2, x, use_reentrant=False)
            state = state + self.drop_path(x)
        else:
            x = self.attention(x, attn_mask, is_causal)
            x = state + self.drop_path(x)
            state = checkpoint(self.ln_1, x, use_reentrant=False)
            x = self.mlp(state)
            state = state + self.drop_path(x)
            state = checkpoint(self.ln_2, state, use_reentrant=False)
        return state

    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None, is_causal: bool =False,
                selective_checkpointing: bool = False):
        if selective_checkpointing:
            return self.checkpoint_forward(x, attn_mask, is_causal=is_causal)
        if self.use_preln:
            x = x + self.drop_path(self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask, is_causal=is_causal)))
            x = x + self.drop_path(self.ls_2(self.mlp(self.ln_2(x))))
        else:
            x = x + self.drop_path(self.attention(x, attn_mask=attn_mask, is_causal=is_causal))
            x = self.ln_1(x)
            x = x + self.drop_path(self.mlp(x))
            x = self.ln_2(x)
        return x


class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 mlp_ratio: float = 4.0,
                 ls_init_value: float = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 use_preln: bool = True,
                 ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.selective_checkpointing = False
        self.grad_checkpointing_params = {'use_reentrant': False}
        if attn_drop == 0 and drop_path == 0 and drop_path == 0:
            self.grad_checkpointing_params.update({'preserve_rng_state': False})
        else:
            self.grad_checkpointing_params.update({'preserve_rng_state': True})

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                act_layer=act_layer, norm_layer=norm_layer,
                use_preln=use_preln)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None,
                is_causal: bool =False):
        for r in self.resblocks:
            if self.training and self.grad_checkpointing and not torch.jit.is_scripting():
                if not self.selective_checkpointing:
                    x = checkpoint(r, x, attn_mask, is_causal=is_causal, **self.grad_checkpointing_params)
                else:
                    x = r(x, attn_mask=attn_mask, is_causal=is_causal, selective_checkpointing=True)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 mlp_ratio: float,
                 num_frames: int = 1,
                 cross_frames: bool = True,
                 ls_init_value: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 ln_pre: bool = True,
                 ln_post: bool = True,
                 act_layer: str = 'gelu',
                 norm_layer: str = 'layer_norm',
                 mask_type: Union[str, None] = 'none',
                 mask_block_size: int = -1
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.patches_per_frame = self.grid_size[0] * self.grid_size[1]
        self.mask_type = mask_type
        self.mask_block_size = mask_block_size

        if act_layer.lower() == 'gelu':
            self.act_layer = nn.GELU
        else:
            raise ValueError(f"Unsupported activation function: {act_layer}")
        if norm_layer.lower() == 'layer_norm':
            self.norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unsupported normalization: {norm_layer}")

        self.conv1 = nn.Linear(
            in_features=3 * self.patch_size[0] * self.patch_size[1],
            out_features=width,
            bias=not ln_pre
        )

        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1], width))
        assert num_frames >= 1
        self.num_frames = num_frames
        self.cross_frames = cross_frames
        if num_frames > 1 and cross_frames:
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(num_frames, width))
        else:
            self.temporal_positional_embedding = None

        self.ln_pre = self.norm_layer(width) if ln_pre else nn.Identity()

        self.transformer = Transformer(
            width, layers, heads, mlp_ratio, ls_init_value=ls_init_value,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            act_layer=self.act_layer, norm_layer=self.norm_layer,
        )

        self.ln_post = self.norm_layer(width)

        self.init_parameters()

    def init_parameters(self):
        if self.positional_embedding is not None:
            nn.init.normal_(self.positional_embedding, std=0.02)
        trunc_normal_(self.conv1.weight, std=0.02)
        for block in self.transformer.resblocks:
            for n, p in block.named_parameters():
                if 'weight' in n:
                    if 'ln' not in n:
                        trunc_normal_(p, std=0.02)
                elif 'bias' in n:
                    nn.init.zeros_(p)
                else:
                    raise NotImplementedError(f'Unknown parameters named {n}')

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True, selective=False):
        self.transformer.grad_checkpointing = enable
        self.transformer.selective_checkpointing = selective
        

    def forward(self, x):
        if self.num_frames == 1:
            x = rearrange(
                x, "b c (hh sh) (ww sw) -> b (hh ww) (c sh sw)",
                sh=self.patch_size[0], sw=self.patch_size[1]
            )
            x = self.conv1(x)
            x = x + self.positional_embedding.to(x.dtype)
        elif self.cross_frames:
            num_frames = x.shape[2]
            assert num_frames <= self.num_frames, 'Number of frames should be less or equal to the model setting'
            x = rearrange(
                x, "b c t (hh sh) (ww sw) -> b (t hh ww) (c sh sw)",
                sh=self.patch_size[0], sw=self.patch_size[1]
            )
            x = self.conv1(x)
            tile_pos_embed = self.positional_embedding.repeat(num_frames, 1)
            tile_tem_embed = self.temporal_positional_embedding[:num_frames].repeat_interleave(self.patches_per_frame, 0)
            total_pos_embed = tile_pos_embed + tile_tem_embed
            x = x + total_pos_embed.to(x.dtype).squeeze(0)
        else:
            x = rearrange(
                x, "b c t (hh sh) (ww sw) -> (b t) (hh ww) (c sh sw)",
                sh=self.patch_size[0], sw=self.patch_size[1]
            )
            x = self.conv1(x)
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        block_size = self.grid_size[0] * self.grid_size[1] if self.mask_block_size <= 0 else self.mask_block_size
        attn_mask = get_attention_mask(x.size(0), x.device, mask_type=self.mask_type, block_size=block_size)
        x = self.transformer(x, attn_mask, is_causal=self.mask_type == 'causal')
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self,
                 image_size: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 mlp_ratio: float,
                 num_frames: int = 1,
                 cross_frames: bool = True,
                 ls_init_value: float = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 ln_pre: bool = True,
                 ln_post: bool = True,
                 act_layer: str = 'gelu',
                 norm_layer: str = 'layer_norm',
                 use_ffn_output: bool = True,
                 dim_ffn_output: int = 3072,
                 logit_laplace: bool = False,
                 mask_type: Union[str, None] = 'none',
                 mask_block_size: int = -1
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.patches_per_frame = self.grid_size[0] * self.grid_size[1]
        self.mask_type = mask_type
        self.mask_block_size = mask_block_size

        if act_layer.lower() == 'gelu':
            self.act_layer = nn.GELU
        else:
            raise ValueError(f"Unsupported activation function: {act_layer}")
        if norm_layer.lower() == 'layer_norm':
            self.norm_layer = nn.LayerNorm
        else:
            raise ValueError(f"Unsupported normalization: {norm_layer}")

        self.use_ffn_output = use_ffn_output
        if use_ffn_output:
            self.ffn = nn.Sequential(
                nn.Linear(width, dim_ffn_output),
                nn.Tanh(),
            )
            self.conv_out = nn.Linear(
            in_features=dim_ffn_output,
            out_features=3 * self.patch_size[0] * self.patch_size[1] * (1 + logit_laplace)
        )
        else:
            self.ffn = nn.Identity()
            self.conv_out = nn.Linear(
                in_features=width,
                out_features=3 * self.patch_size[0] * self.patch_size[1] * (1 + logit_laplace)
            )

        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1], width))
        assert num_frames >= 1
        self.num_frames = num_frames
        self.cross_frames = cross_frames
        if num_frames > 1 and cross_frames:
            self.temporal_positional_embedding = nn.Parameter(torch.zeros(num_frames, width))
        else:
            self.temporal_positional_embedding = None

        self.ln_pre = self.norm_layer(width) if ln_pre else nn.Identity()

        self.transformer = Transformer(
            width, layers, heads, mlp_ratio, ls_init_value=ls_init_value,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate,
            act_layer=self.act_layer, norm_layer=self.norm_layer,
        )

        self.ln_post = self.norm_layer(width) if ln_post else nn.Identity()

        self.init_parameters()

    def init_parameters(self):
        if self.positional_embedding is not None:
            nn.init.normal_(self.positional_embedding, std=0.02)

        for block in self.transformer.resblocks:
            for n, p in block.named_parameters():
                if 'weight' in n:
                    if 'ln' not in n:
                        trunc_normal_(p, std=0.02)
                elif 'bias' in n:
                    nn.init.zeros_(p)
                else:
                    raise NotImplementedError(f'Unknown parameters named {n}')
        if self.use_ffn_output:
            trunc_normal_(self.ffn[0].weight, std=0.02)
        trunc_normal_(self.conv_out.weight, std=0.02)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True, selective=False):
        self.transformer.grad_checkpointing = enable
        self.transformer.selective_checkpointing = selective

    def forward(self, x):
        if self.num_frames == 1 or not self.cross_frames:
            x = x + self.positional_embedding.to(x.dtype)
        else:
            num_frames = x.shape[1] // self.patches_per_frame
            assert num_frames <= self.num_frames, 'Number of frames should be less or equal to the model setting'
            tile_pos_embed = self.positional_embedding.repeat(num_frames, 1)
            tile_tem_embed = self.temporal_positional_embedding[:num_frames].repeat_interleave(self.patches_per_frame, 0)
            total_pos_embed = tile_pos_embed + tile_tem_embed
            x = x + total_pos_embed.to(x.dtype).squeeze(0)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        block_size = self.grid_size[0] * self.grid_size[1] if self.mask_block_size <= 0 else self.mask_block_size
        attn_mask = get_attention_mask(x.size(0), x.device, mask_type=self.mask_type, block_size=block_size)
        x = self.transformer(x, attn_mask, is_causal=self.mask_type == 'causal')
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        x = self.ffn(x)
        x = self.conv_out(x)
        if self.num_frames == 1:
            x = rearrange(
                x, "b (hh ww) (c sh sw) -> b c (hh sh) (ww sw)",
                hh = self.grid_size[0], ww=self.grid_size[1],
                sh=self.patch_size[0], sw=self.patch_size[1]
            )
        elif self.cross_frames:
            x = rearrange(
                x, "b (t hh ww) (c sh sw) -> b c t (hh sh) (ww sw)",
                t = num_frames, hh = self.grid_size[0], ww=self.grid_size[1],
                sh=self.patch_size[0], sw=self.patch_size[1]
            )
        else:
            x = rearrange(
                x, "(b t) (hh ww) (c sh sw) -> b c t (hh sh) (ww sw)",
                t = num_frames, hh = self.grid_size[0], ww=self.grid_size[1],
                sh=self.patch_size[0], sw=self.patch_size[1]
            )

        return x
