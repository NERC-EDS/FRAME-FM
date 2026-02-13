# Copyright (c) Matt Arran.

# This source code is adapted from Masked Autoencoder (MAE) code, copyright
# Meta Platforms, Inc. and affiliates, licensed under the license found in the
# licences/LICENSE_MAE.txt file.
# PatchEmbed is inspired by the equivalent class in PyTorch Image Models (timm).
# --------------------------------------------------------
# Embedding utils
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae/tree/main
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from collections.abc import Callable
from math import prod
import numpy as np
import torch

_Conv_dim_dict: dict[int, type[torch.nn.modules.conv._ConvNd]] = {
    1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d
    }
_conv_dim_dict: dict[int, Callable[..., torch.Tensor]] = {
    1: torch.nn.functional.conv1d, 2: torch.nn.functional.conv2d, 3: torch.nn.functional.conv3d
    }


def sincos_embed_coords(
        embed_dim: int, coordinates: torch.Tensor, max_period: float = 10000
        ) -> torch.Tensor:
    """Create a sin-cos embedding of an array of 1D coordinates.

    Args:
        embed_dim (int): Number of dimensions in which to embed coordinates.
        coordinates (torch.Tensor): Coordinates to embed.
        max_period (int, optional): Maximum sin-cos period. Defaults to 10000.

    Returns:
        torch.Tensor: Array with each row the sin-cos embedding of a coordinate.
    """
    omega = 2 * np.pi / max_period**torch.linspace(0, 1, embed_dim)  # (D,)
    phases = torch.einsum('m,d->md', coordinates, omega)  # (M, D), outer product
    sin_embedding = torch.sin(phases)  # (M, D)
    cos_embedding = torch.cos(phases)  # (M, D)
    return torch.cat([sin_embedding, cos_embedding], dim=1)  # (M, 2D)


class BaseEmbedder(torch.nn.Module):
    n_patches: int
    embed_dim: int
    reconstruct_dim: int

    def initialize_weights(self):
        raise NotImplementedError

    def tokenify(self, inpt: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def reconstruct_tokens(self, embedding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def untokenify(self, inpt: torch.Tensor) -> torch.Tensor:
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
                 **conv_kwargs):
        """Instantiate embedder for patches in 1-3D, n-channel images.

        Args:
            input_shape (tuple[int, ...]): Shape of input image.
            patch_shape (tuple[int, ...]): Shape of patches into which image will be divided.
            n_channels (int): Number of channels recorded in image.
            embed_dim (int): Number of dimensions into which to embed each patch.
            reconstruct_dim (int): Number of embedding dimensions from which to reconstruct patch.
            bias (bool, optional): Whether to include bias in patch embedding. Defaults to True.
            norm_layer (torch.nn.Module | None, optional): Layer with which to normalise embedding.
                Defaults to None: no normalisation.
            **conv_kwargs: Keyword arguments to pass for convolution layer instantiation.
        """
        super().__init__()
        assert len(input_shape) in _conv_dim_dict.keys(), \
            f"{len(input_shape)}D input not supported"
        self.input_shape, self.patch_shape = input_shape, patch_shape
        self.grid_shape, self.n_patches = self._count_patches(input_shape)
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        conv_class = _Conv_dim_dict[len(input_shape)]
        self.proj = conv_class(
            n_channels,
            embed_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
            bias=bias,
            **conv_kwargs
            )
        if norm_layer is None:
            self.norm = torch.nn.Identity()
        else:
            self.norm = norm_layer(embed_dim)
        self.reconstruct_dim = reconstruct_dim
        self.reconstruct_layer = torch.nn.Linear(
            reconstruct_dim, prod(patch_shape) * n_channels, bias=True
            )

    def _count_patches(self, input_shape):
        grid_shape = tuple(s_i // s_p for s_i, s_p in zip(input_shape, self.patch_shape))
        n_patches = prod(grid_shape)
        return grid_shape, n_patches

    def _define_position_embedding(self, embed_dim: int) -> torch.nn.Parameter:
        """Create a sin-cos embedding of positions on an n-dimensional grid.

        Args:
            embed_dim (int): Number of dimensions into which to embed grid positions.
        Returns:
            torch.Parameter: Array with each row the sin-cos embedding of a grid position.
        """
        # Divide sin and cos embeddings according to size of grid in each dimension
        embed_dims = np.round(
            [embed_dim // 2 * grid_s / sum(self.grid_shape) for grid_s in self.grid_shape]
            ).astype(int)
        embed_dims[embed_dims.argmin()] += embed_dim // 2 - embed_dims.sum()
        # Define grid: len(grid_shape)-tuple of np.arrays of shape grid_shape
        grid = torch.meshgrid([
            torch.arange(grid_s, dtype=torch.float32) for grid_s in self.grid_shape
            ], indexing='ij')
        # Create sincos embedding: np.array of shape (prod(grid_shape), 2 * sum(embed_dims))
        embedding = torch.cat([
            sincos_embed_coords(dim, coords.flatten()) for dim, coords in zip(embed_dims, grid)
            ], dim=1)
        return torch.nn.Parameter(embedding.float().unsqueeze(0), requires_grad=False)

    def initialize_weights(self):
        """Set up embedder weights and parameters.
        """
        # define fixed sin-cos embeddings of patch position within image
        self.pos_embed = self._define_position_embedding(self.embed_dim)
        self.decoder_pos_embed = self._define_position_embedding(self.reconstruct_dim)
        # initialize projection like nn.Linear (instead of nn.Conv2d)
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def tokenify(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reshape batched input n-D images into sequences of patch tokens.

        Shapes are given for the example of a batch of B, C-channel 2D images,
        with image size (Hh, Ww) and patch size (h, w).

        Args:
            inputs (torch.Tensor): Batched sequence of images, shape e.g. [B, C, Hh, Ww]

        Returns:
            torch.Tensor: Batched sequences of patch tokens, shape e.g. [B, HW, hwC]
        """
        patch_shape, n_dim = self.patch_shape, len(self.patch_shape)
        assert len(inputs.shape) - 2 == n_dim, \
            f"{len(inputs.shape) - 2}-D input not divisible into {n_dim}-D patches"
        for dim, (s_i, s_p) in enumerate(zip(inputs.shape[2:], patch_shape)):
            assert s_i % s_p == 0, \
                f"Input dimension {dim} not divisible into patches ({s_i} % {s_p} != 0)"

        grid_shape, n_patches = self._count_patches(inputs.shape[2:])
        x = inputs.reshape(shape=(
            (inputs.shape[0], self.n_channels)
            + sum([(s_g, s_p) for s_g, s_p in zip(grid_shape, patch_shape)], ())
            ))
        # (B, C, H, h, W, w) for 2D patches
        x = x.permute(
            0, *range(2, 2 + 2 * n_dim, 2), *range(3, 3 + 2 * n_dim, 2), 1
            )
        # (B, H, W, h, w, C) for 2D patches
        x = x.reshape(shape=(inputs.shape[0], n_patches, prod(patch_shape) * self.n_channels))
        # (B, H W, h w C) for 2D patches
        return x

    def untokenify(self, x: torch.Tensor,
                   output_shape: tuple[int, ...] | None = None) -> torch.Tensor:
        """Reshape batched sequences of patch tokens into n-D images.

        Shapes are given for the example of a batch of B, C-channel 2D images,
        with image size (Hh, Ww) and patch size (h, w).

        Args:
            x (torch.Tensor): Batched sequences of tokens, shape e.g. [B, HW, hwC]
            output_shape (tuple[int, ...] | None, optional): Required image shape, if different
                from input_shape in class instantiation. Defaults to None: use input_shape.

        Returns:
            torch.Tensor: Batched sequence of images, shape e.g. [B, C, Hh, Ww]
        """
        patch_shape, n_dim = self.patch_shape, len(self.patch_shape)
        n_channels = self.n_channels
        if output_shape is None:
            grid_shape, n_patches = self.grid_shape, self.n_patches
        else:
            assert len(output_shape) == n_dim, \
                f"{len(output_shape)}-D output not formable from {n_dim}-D patches"
            grid_shape, n_patches = self._count_patches(output_shape)
        assert x.shape[1] == n_patches, \
            f"Grid shape {grid_shape} not formable from {x.shape[1]} patches"
        assert x.shape[2] == prod(patch_shape) * self.n_channels, \
            f"{n_channels}-channel {patch_shape} patch not formable from {x.shape[2]} values"

        x = x.reshape(shape=(x.shape[0],) + grid_shape + patch_shape + (n_channels,))
        # (N, H, W, h, w, C) for 2D patches
        x = x.permute(
            0, -1, *sum([(id, n_dim + id) for id in range(1, 1 + n_dim)], ())
            )
        # (N, C, H, h, W, w) for 2D patches
        imgs = x.reshape(shape=(
            (x.shape[0], n_channels)
            + sum([(s_g * s_p,) for s_g, s_p in zip(grid_shape, patch_shape)], ())
            ))
        # (N, C, H h, W w) for 2D patches
        return imgs

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert batched input n-D images into sequences of patch embeddings.

        Shapes are given for the example of a batch of B, C-channel 2D images,
        with image size (Hh, Ww) and patch size (h, w), with a D-D embedding.

        Args:
            x (torch.Tensor): Batched sequence of images, shape e.g. [B, C, Hh, Ww].
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                * Batched sequences of embeddings, shape e.g. [B, HW, D]
                * Batched sequences of positions, shape e.g. [B, HW, D]
        """
        for dim, (s_actual, s_expected) in enumerate(zip(x.shape[2:], self.input_shape)):
            torch._assert(
                s_actual == s_expected,
                f"Input dimension {dim} ({s_actual}) doesn't match specification ({s_expected})"
                )
        x = self.proj(x)  # Project each patch into embedding, by convolution
        x = x.flatten(start_dim=2).transpose(1, 2)  # (B, D, H, W) -> (B, HW, D) for 2D patches
        x = self.norm(x)
        x = x + self.pos_embed
        return x, self.decoder_pos_embed

    def reconstruct_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct patch tokens from an embedding.

        Args:
            x (torch.Tensor): Batched sequence of images, shape e.g. [B, C, Hh, Ww].
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
        """
        return self.reconstruct_layer(x)


class STPatchEmbed(PatchEmbed):
    """ 1-3D spatiotemporally located input to Patch Embedding
    """
    def __init__(self,
                 input_shape: tuple[int, ...],
                 patch_shape: tuple[int, ...],
                 n_channels: int,
                 position_space: tuple[tuple[float, float], ...],
                 embed_dim: int,
                 reconstruct_dim: int,
                 pos_embed_ratio: tuple[float, ...],
                 bias: bool = True,
                 norm_layer: torch.nn.Module | None = None,
                 **conv_kwargs):
        """Instantiate embedder for patches in 1-3D, n-channel images.

        Args:
            input_shape (tuple[int, ...]): Shape of input image.
            patch_shape (tuple[int, ...]): Shape of patches into which image will be divided.
            n_channels (int): Number of channels recorded in image.
            position_space (tuple[tuple[float, float], ...]): Space in which pixels are positioned.
                (range(x_0), ...) for coordinates x_i.
            embed_dim (int): Number of dimensions into which to embed each patch.
            reconstruct_dim (int): Number of embedding dimensions from which to reconstruct patch.
            pos_embed_ratio (tuple[float, ...]): Relative sizes of position embedding dimensions.
            bias (bool, optional): Whether to include bias in patch embedding. Defaults to True.
            norm_layer (torch.nn.Module | None, optional): Layer with which to normalise embedding.
                Defaults to None: no normalisation.
            **conv_kwargs: Keyword arguments to pass for convolution layer instantiation.
        """
        super().__init__(
            input_shape=input_shape,
            patch_shape=patch_shape,
            n_channels=n_channels,
            embed_dim=embed_dim,
            reconstruct_dim=reconstruct_dim,
            bias=bias,
            norm_layer=norm_layer,
            **conv_kwargs,
            )
        self.position_space = position_space
        self.pos_embed_ratio = pos_embed_ratio

    def _define_position_embedding(self, embed_dim: int):
        # Divide sin and cos embeddings according to size of grid in each dimension
        embed_dims = np.round([
            embed_dim // 2 * embed_size / sum(self.pos_embed_ratio)
            for embed_size in self.pos_embed_ratio]
            ).astype(int)
        embed_dims[embed_dims.argmin()] += embed_dim // 2 - embed_dims.sum()

        conv_fn = _conv_dim_dict[len(self.position_space)]
        conv_kernel = torch.ones(self.patch_shape) / prod(self.patch_shape)

        def embedding(pos: torch.Tensor) -> torch.Tensor:
            # shape B, N, Hh, Ww for 2D patches
            batch_size = pos.shape[0]
            pos = conv_fn(pos, conv_kernel, stride=self.patch_shape)  # B, N, H, W for 2D patches
            pos = pos.transpose(0, 1).flatten(start_dim=1)  # N, BHW for 2D patches
            embeddings = torch.cat([
                sincos_embed_coords(dim, coords, max_period=x_max - x_min)
                for dim, (x_min, x_max), coords in zip(embed_dims, self.position_space, pos)
                ], dim=1)  # BHW, D for 2D patches
            return embeddings.reshape([batch_size, -1, embed_dim])  # B, HW, D for 2D patches
        return embedding

    def initialize_weights(self):
        """Set up embedder weights and parameters.
        """
        # define fixed sin-cos embedding functions for patch positions
        self.pos_embed = self._define_position_embedding(self.embed_dim)
        self.decoder_pos_embed = self._define_position_embedding(self.reconstruct_dim)
        # initialize projection like nn.Linear (instead of nn.Conv2d)
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert batched spatiotemporal inputs into sequences of patch embeddings.

        Shapes are given for the example of a batch of B, C-channel 2D inputs in N-D space,
        with image size (Hh, Ww) and patch size (h, w), with a D-D embedding.

        Args:
            x (torch.Tensor): Batched sequence of inputs, shape e.g. [B, C, Hh, Ww].
            pos (torch.Tensor): Batched sequence of input locations, shape e.g. [B, Hh, Ww, N].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
             * Batched sequences of embeddings, shape e.g. [B, HW, D]
             * Batched sequences of position embeddings, shape e.g. [B, HW, D_d]
        """
        torch._assert(
            x.shape[1] == self.n_channels,
            f"# of input channels ({x.shape[1]}) doesn't match spec. ({self.n_channels})"
            )
        torch._assert(
            pos.shape[1] == len(self.position_space),
            f"({pos.shape[1]})-D position space doesn't match spec. ({len(self.position_space)})"
            )
        sizes_iterator = enumerate(zip(x.shape[2:], pos.shape[2:], self.input_shape))
        for dim, (s_x, s_pos, s_spec) in sizes_iterator:
            torch._assert(
                s_x == s_spec,
                f"Input values dimension {dim} ({s_x}) doesn't match spec. ({s_spec})"
                )
            torch._assert(
                s_pos == s_spec,
                f"Input positions dimension {dim} ({s_pos}) doesn't match spec. ({s_spec})"
                )
        x = self.proj(x)  # Project each patch into embedding, by convolution
        x = x.flatten(start_dim=2).transpose(1, 2)  # (B, D, H, W) -> (B, HW, D) for 2D patches
        x = self.norm(x)
        x = x + self.pos_embed(pos)
        return x, self.decoder_pos_embed(pos)
