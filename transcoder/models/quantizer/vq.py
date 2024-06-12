from einops import rearrange
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, l2_norm, beta, input_format='bchw'):
        super().__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.l2_norm = l2_norm
        self.beta = beta
        assert input_format in ['bchw', 'blc']
        self.input_format = input_format

        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1 / n_embed, 1 / n_embed)
        self.bits_per_index = int(np.ceil(np.log2(n_embed)))

    def forward(self, z):
        batch = z.shape[0]
        if self.input_format == 'bchw':
            z = rearrange(z, 'b c h w -> b h w c')

        if self.l2_norm:
            z = F.normalize(z, dim=-1)
            z_flatten = z.reshape(-1, self.embed_dim)
            embedding_weight = F.normalize(self.embedding.weight, dim=-1)
            d = -z_flatten @ embedding_weight.t()
        else:
            z_flatten = z.reshape(-1, self.embed_dim)
            d = torch.sum(z_flatten ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * z_flatten @ self.embedding.weight.t()

        min_encoding_indices = torch.argmin(d.detach(), dim=1)
        if not self.training:
            used_codes = torch.unique(min_encoding_indices, return_counts=False)
        else:
            used_codes = None
        cb_usage = F.one_hot(min_encoding_indices, self.n_embed).sum(0)
        cb_entropy = self.get_entropy(cb_usage)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        if self.l2_norm:
            z_q = F.normalize(z_q, dim=-1)

        # fix the issue with loss scaling
        # loss weight should not associate with the dimensionality of words
        # loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        loss = self.beta * torch.mean(((z_q.detach() - z) ** 2).sum(dim=-1)) + torch.mean(((z_q - z.detach()) ** 2).sum(dim=-1))

        z_q = z + (z_q - z).detach()
        if self.input_format == 'bchw':
            z_q = rearrange(z_q, 'b h w c -> b c h w')
        return z_q, loss, {"H":cb_entropy, "used_codes": used_codes, 'indices': min_encoding_indices.view(batch, -1)}

    def get_entropy(self, count, eps=1e-4):
        probs = (count + eps) / (count + eps).sum()
        H = -(probs * torch.log(probs)).sum()
        return H


    def get_codebook_entry(self, indices):
        z_q = self.embedding(indices)
        if self.l2_norm:
            z_q = F.normalize(z_q, dim=-1)

        if self.input_format == 'bchw':
            h = w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], 'Invalid sequence length'
            z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h)
        return z_q


class EMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim, l2_norm, beta, decay=0.99, eps=1e-5, random_restart=True, restart_threshold=1.0, input_format='bchw'):
        super().__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.l2_norm = l2_norm
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.random_restart = random_restart
        self.restart_threshold = restart_threshold
        self.input_format = input_format

        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1 / n_embed, 1 / n_embed) # TODO (yzhao): test other initialization methods 
        self.register_buffer("ema_cluster_size", torch.zeros(self.n_embed))
        self.embedding_avg = nn.Parameter(torch.Tensor(self.n_embed, self.embed_dim))
        self.embedding_avg.data.copy_(self.embedding.weight.data)

    def _tile(self, z):
        n_z, embedding_dim = z.shape
        if n_z < self.n_embed:
            n_repeats = (self.n_embed + n_z - 1) // n_z
            std = 0.01 / np.sqrt(embedding_dim)
            z = z.repeat(n_repeats, 1)
            z = z + torch.randn_like(z) * std
        return z

    def forward(self, z):
        if self.input_format == 'bchw':
            z = rearrange(z, 'b c h w -> b h w c')
        z_flatten = z.reshape(-1, self.embed_dim)

        d = torch.sum(z_flatten ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * z_flatten @ self.embedding.weight.t()

        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.n_embed, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        z_q = self.embedding(encoding_indices).view(z.shape)
        if self.l2_norm:
            z = F.normalize(z, dim=-1)
            z_q = F.normalize(z_q, dim=-1)

        if self.training:
            # EMA update cluster size
            encodings_sum = encodings.sum(0)
            if dist.is_initialized(): dist.all_reduce(encodings_sum)
            self.ema_cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1-self.decay)

            # EMA update of the embedding vectors
            dw = encodings.t() @ z_flatten
            if dist.is_initialized(): dist.all_reduce(dw)
            self.embedding_avg.data.mul_(self.decay).add_(dw, alpha=1-self.decay)
 
            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size)
            weights = (self.ema_cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            self.embedding.weight.data = self.embedding_avg.data / weights.unsqueeze(1)

            if self.random_restart:
                zz = self._tile(z_flatten)
                _k_rand = zz[torch.randperm(zz.size(0))][:self.n_embed]
                if dist.is_initialized(): dist.broadcast(_k_rand, 0)
                usage = (self.ema_cluster_size.view(-1, 1) > self.restart_threshold).float()
                self.embedding.weight.data.mul_(usage).add_(_k_rand * (1 - usage))

        loss = self.beta * torch.mean((z_q.detach() - z) ** 2)

        z_q = z + (z_q - z).detach()
        if self.input_format == 'bchw':
            z_q = rearrange(z_q, 'b h w c -> b c h w')
        # TODO (yzhao): monitor utility of the dictionary
        return z_q, loss, {}
