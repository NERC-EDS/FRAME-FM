# Copyright (c) Matt Arran.

# This source code is adapted from code (c) Meta Platforms, Inc. and affiliates,
# licensed under the license found in the LICENSE file in this folder.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae/tree/main
# --------------------------------------------------------

import numpy as np


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
