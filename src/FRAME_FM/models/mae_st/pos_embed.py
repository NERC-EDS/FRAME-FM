# Copyright (c) Matt Arran.

# This source code is adapted from code (c) Meta Platforms, Inc. and affiliates,
# licensed under the license found in the LICENSE file in this folder.
# --------------------------------------------------------
# Embedding utils
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae/tree/main
# --------------------------------------------------------

from math import prod
import numpy as np
import torch


_conv_dim_dict: dict[int, type[torch.nn.modules.conv._ConvNd]] = {
    1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d
    }


def get_nd_sincos_pos_embed(
        embed_dim: int,
        grid_shape: tuple[int, ...],
        cls_token: bool = False,
        ) -> np.ndarray:
    """Create a sin-cos embedding of positions on an n-dimensional grid.

    Args:
        embed_dim (int): Number of dimensions into which to embed grid positions.
        grid_shape (Tuple[int, ...]): Shape of grid for which to create an embedding.
        cls_token (bool, optional): Whether to add a whole-input class token. Defaults to False.

    Returns:
        np.ndarray: Array with each row the sin-cos embedding of a grid position.
    """
    # Divide sin and cos embeddings according to size of grid in each dimension
    embed_dims = np.round(
        [embed_dim // 2 * grid_s / sum(grid_shape) for grid_s in grid_shape]
        ).astype(int)
    embed_dims[embed_dims.argmin()] += embed_dim // 2 - embed_dims.sum()

    grid = np.meshgrid(*[
        np.arange(grid_s, dtype=np.float32) for grid_s in grid_shape
        ])  # len(grid_shape)-tuple of np.arrays of shape grid_shape
    embedding = np.concatenate([
        sincos_embed_coords(dim, coords.flatten()) for dim, coords in zip(embed_dims, grid)
        ], axis=1)  # np.array of shape (prod(grid_shape), 2 * sum(sin_embed_dims))
    if cls_token:
        embedding = np.concatenate([np.zeros([1, 2 * sum(embed_dims)]), embedding], axis=0)
    return embedding


def sincos_embed_coords(
        embed_dim: int, coordinates: np.ndarray, max_period: int = 10000
        ) -> np.ndarray:
    """Create a sin-cos embedding of an array of 1D coordinates.

    Args:
        embed_dim (int): Number of dimensions in which to embed coordinates.
        coordinates (np.ndarray): Coordinates to embed.
        max_period (int, optional): Maximum sin-cos period. Defaults to 10000.

    Returns:
        np.ndarray: Array with each row the sin-cos embedding of a coordinate.
    """
    omega = 2 * np.pi / max_period**np.linspace(0, 1, embed_dim, dtype=float)  # (D,)
    out = np.einsum('m,d->md', coordinates, omega)  # (M, D), outer product
    sin_embedding = np.sin(out)  # (M, D)
    cos_embedding = np.cos(out)  # (M, D)
    return np.concatenate([sin_embedding, cos_embedding], axis=1)  # (M, 2D)


def _count_patches(input_shape, patch_shape):
    grid_shape = tuple(s_i // s_p for s_i, s_p in zip(input_shape, patch_shape))
    n_patches = prod(grid_shape)
    return grid_shape, n_patches


class BaseEmbedder(torch.nn.Module):
    n_patches: int
    embed_dim: int
    reconstruct_dim: int

    def initialize_weights(self):
        raise NotImplementedError

    def patchify(self, inpt: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def embed_pos(self, pos: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError

    def reconstruct_patches(self, embedding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PatchEmbed(BaseEmbedder):
    """ 1-3D Image to Patch Embedding
    """
    def __init__(self,
                 input_shape: tuple[int, ...],
                 patch_shape: tuple[int, ...],
                 n_channels: int,
                 embed_dim: int,
                 reconstruct_dim: int,
                 bias: bool = True,
                 norm_layer: torch.nn.Module | None = None,
                 device=None,
                 dtype=None,
                 **conv_kwargs):
        super().__init__()
        assert len(input_shape) in _conv_dim_dict.keys(), \
            f"{len(input_shape)}-D input not supported"
        self.input_shape, self.patch_shape = input_shape, patch_shape
        self.grid_shape, self.n_patches = _count_patches(input_shape, patch_shape)
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        conv_class = _conv_dim_dict[len(input_shape)]
        self.proj = conv_class(
            n_channels,
            embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
            bias=bias,
            device=device,
            dtype=dtype,
            **conv_kwargs
            )
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim), requires_grad=False
            )  # fixed sin-cos embedding
        if norm_layer is None:
            self.norm = torch.nn.Identity()
        else:
            self.norm = norm_layer(embed_dim, device=device, dtype=dtype)
        self.reconstruct_dim = reconstruct_dim
        self.decoder_pos_embed = torch.nn.Parameter(
            torch.zeros(1, self.n_patches, reconstruct_dim), requires_grad=False
            )  # fixed sin-cos embedding
        self.reconstruct_layer = torch.nn.Linear(
            reconstruct_dim, prod(patch_shape) * n_channels, bias=True
            )

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_nd_sincos_pos_embed(
            self.embed_dim, self.grid_shape, cls_token=False
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_nd_sincos_pos_embed(
            self.reconstruct_dim, self.grid_shape, cls_token=False
            )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize projection like nn.Linear (instead of nn.Conv2d)
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def patchify(self, inputs: torch.Tensor) -> torch.Tensor:
        patch_shape, n_dim = self.patch_shape, len(self.patch_shape)
        assert len(inputs.shape) - 2 == n_dim, \
            f"{len(inputs.shape) - 2}-D input not divisible into {n_dim}-D patches"
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
            0, *range(2, 2 + 2 * n_dim, 2), *range(3, 3 + 2 * n_dim, 2), 1
            )
        # (N, H, W, P, P, C) for 2D square patches
        x = x.reshape(shape=(inputs.shape[0], n_patches, prod(patch_shape) * self.n_channels))
        # (N, H W, P**2 C) for 2D square patches
        return x

    def unpatchify(self, x: torch.Tensor,
                   output_shape: tuple[int, ...] | None = None) -> torch.Tensor:
        patch_shape, n_dim = self.patch_shape, len(self.patch_shape)
        n_channels = self.n_channels
        if output_shape is None:
            grid_shape, n_patches = self.grid_shape, self.n_patches
        else:
            assert len(output_shape) == n_dim, \
                f"{len(output_shape)}-D output not formable from {n_dim}-D patches"
            grid_shape, n_patches = _count_patches(output_shape, patch_shape)
        assert x.shape[1] == n_patches, \
            f"Grid shape {grid_shape} not formable from {x.shape[1]} patches"
        assert x.shape[2] == prod(patch_shape) * self.n_channels, \
            f"{n_channels}-channel {patch_shape} patch not formable from {x.shape[2]} values"

        x = x.reshape(shape=(x.shape[0],) + grid_shape + patch_shape + (n_channels,))
        # (N, H, W, P, P, C) for 2D square patches
        x = x.permute(
            0, -1, *sum([(id, n_dim + id) for id in range(1, 1 + n_dim)], ())
            )
        # (N, C, H, P, W, P) for 2D square patches
        imgs = x.reshape(shape=(
            (x.shape[0], n_channels)
            + sum([(s_g * s_p,) for s_g, s_p in zip(grid_shape, patch_shape)], ())
            ))
        # (N, C, H P, W P) for 2D square patches
        return imgs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for dim, (s_actual, s_expected) in enumerate(zip(x.shape[2:], self.input_shape)):
            torch._assert(
                s_actual == s_expected,
                f"Input dimension {dim} ({s_actual}) doesn't match specification ({s_expected})"
                )
        x = self.proj(x)  # Project each patch into embedding, by convolution
        x = x.flatten(start_dim=2).transpose(1, 2)  # NC[img_shape] -> N[Prod(img_shape)]C
        x = self.norm(x)
        x = x + self.pos_embed
        return x

    def embed_pos(self, pos: torch.Tensor | None) -> torch.Tensor:
        return self.decoder_pos_embed

    def reconstruct_patches(self, x: torch.Tensor) -> torch.Tensor:
        return self.reconstruct_layer(x)
