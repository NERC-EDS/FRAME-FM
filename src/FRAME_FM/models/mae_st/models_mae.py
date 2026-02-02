# Copyright (c) Matt Arran.

# This source code is adapted from code (c) Meta Platforms, Inc. and affiliates,
# licensed under the license found in the LICENSE file in this folder.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae/tree/main
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from math import prod
from timm.models.vision_transformer import Block
import torch
from torch import _assert, nn

from .pos_embed import get_nd_sincos_pos_embed

_conv_dim_dict = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}


def _count_patches(input_shape, patch_shape):
    grid_shape = tuple(s_i // s_p for s_i, s_p in zip(input_shape, patch_shape))
    n_patches = prod(grid_shape)
    return grid_shape, n_patches


class PatchEmbed(nn.Module):
    """ 1-3D Image to Patch Embedding
    """
    def __init__(self,
                 input_shape: tuple[int, ...],
                 patch_shape: tuple[int, ...],
                 n_channels: int,
                 embed_dim: int,
                 bias: bool = True,
                 norm_layer: nn.Module | None = None,
                 device=None,
                 dtype=None):
        super().__init__()
        assert len(input_shape) in _conv_dim_dict.keys(), \
            f"{len(input_shape)}-D input not supported"
        self.input_shape, self.patch_shape = input_shape, patch_shape
        self.grid_shape, self.n_patches = _count_patches(input_shape, patch_shape)
        self.n_channels = n_channels
        conv_class = _conv_dim_dict[len(input_shape)]
        self.proj = conv_class(
            n_channels,
            embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
            bias=bias,
            device=device,
            dtype=dtype,
            )
        if norm_layer is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm_layer(embed_dim, device=device, dtype=dtype)

    def patchify(self, inputs: torch.Tensor):
        patch_shape = self.patch_shape
        assert len(inputs.shape) - 2 == len(patch_shape), \
            f"{len(inputs.shape) - 2}-D input not divisible into {len(patch_shape)}-D patches"
        for dim, (s_i, s_p) in enumerate(zip(inputs.shape[2:], patch_shape)):
            assert s_i % s_p == 0, \
                f"Input dimension {dim} not divisible into patches ({s_i} % {s_p} != 0)"

        grid_shape, n_patches = _count_patches(inputs.shape[2:], patch_shape)
        x = inputs.reshape(shape=(
            (inputs.shape[0], self.n_channels)
            + sum([(s_g, s_p) for s_g, s_p in zip(grid_shape, patch_shape)], ())
            ))
        # (N, C, H, P, W, P) for 2D square patches
        x = x.permute(
            0, *range(2, 2 + 2 * len(patch_shape), 2), *range(3, 3 + 2 * len(patch_shape), 2), 1
            )
        # (N, H, W, P, P, C) for 2D square patches
        x = x.reshape(shape=(inputs.shape[0], n_patches, prod(patch_shape) * self.n_channels))
        # (N, H W, P**2 C) for 2D square patches
        return x

    def unpatchify(self, x: torch.Tensor, output_shape: tuple[int, ...] | None = None):
        patch_shape = self.patch_shape
        n_channels = self.n_channels
        if output_shape is None:
            grid_shape, n_patches = self.grid_shape, self.n_patches
        else:
            assert len(output_shape) == len(patch_shape), \
                f"{len(output_shape)}-D output not formable from {len(patch_shape)}-D patches"
            grid_shape, n_patches = _count_patches(output_shape, patch_shape)
        assert x.shape[1] == n_patches, \
            f"Grid shape {grid_shape} not formable from {x.shape[1]} patches"
        assert x.shape[2] == prod(patch_shape) * self.n_channels, \
            f"{n_channels}-channel {patch_shape} patch not formable from {x.shape[2]} values"

        x = x.reshape(shape=(x.shape[0],) + grid_shape + patch_shape + (n_channels,))
        # (N, H, W, P, P, C) for 2D square patches
        x = x.permute(
            0, -1, *sum([(id, len(patch_shape) + id) for id in range(1, 1 + len(patch_shape))], ())
            )
        # (N, C, H, P, W, P) for 2D square patches
        imgs = x.reshape(shape=(
            (x.shape[0], n_channels)
            + sum([(s_g * s_p,) for s_g, s_p in zip(grid_shape, patch_shape)], ())
            ))
        # (N, C, H P, W P) for 2D square patches
        return imgs

    def forward(self, x):
        for dim, (s_actual, s_expected) in enumerate(zip(x.shape[2:], self.input_shape)):
            _assert(
                s_actual == s_expected,
                f"Input dimension {dim} ({s_actual}) doesn't match specification ({s_expected})"
                )
        x = self.proj(x)  # Project each patch into embedding, by convolution
        x = x.flatten(start_dim=2).transpose(1, 2)  # NC[img_shape] -> N[Prod(img_shape)]C
        x = self.norm(x)
        return x


class MaskedAutoencoderND(nn.Module):
    """ Masked Autoencoder with n-D VisionTransformer backbone
    """
    def __init__(self,
                 input_shape: tuple[int, ...] = 3*(224,),
                 patch_shape: tuple[int, ...] = 3*(16,),
                 input_channels: int = 3,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 decoder_embed_dim: int = 512,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
                 norm_pix_loss: bool = False):
        super().__init__()
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(input_shape, patch_shape, input_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + n_patches, embed_dim), requires_grad=False
            )  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
            ])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, decoder_embed_dim), requires_grad=False
            )  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
            ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, prod(patch_shape) * input_channels, bias=True
            )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_nd_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_shape, cls_token=True
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_nd_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.patch_embed.grid_shape, cls_token=True
            )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, C, H, W]
        pred: [N, L, p*p*C]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patch_embed.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*C]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
