# Copyright (c) Matt Arran.

# This source code is adapted from code (c) Meta Platforms, Inc. and affiliates,
# licensed under the license in the LICENSE_MAE.txt file in this folder.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae/tree/main
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from timm.models.vision_transformer import Block
import torch
from torch import nn
from typing import Sequence

from .embedders import BaseEmbedder


class MultimodalMaskedAutoencoder(nn.Module):
    """Masked Autoencoder with flexible multi-input embeddings and transformer backbone
    """
    def __init__(self,
                 input_embedders: list[BaseEmbedder],
                 encoder_depth: int = 24,
                 encoder_num_heads: int = 16,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: float = 4.,
                 norm_layer: type[nn.LayerNorm] = nn.LayerNorm,
                 norm_token_loss: bool = False):
        """Instantiate Multimodal Masked Autoencoder

        Args:
            input_embedders (list[BaseEmbedder]): Embedding classes for each model input.
            encoder_depth (int, optional): Number of attention layers for encoding. Defaults to 24.
            encoder_num_heads (int, optional): Number of attention heads per layer. Defaults to 16.
            decoder_depth (int, optional): Number of attention layers for decoding. Defaults to 8.
            decoder_num_heads (int, optional): Number of attention heads per layer. Defaults to 16.
            mlp_ratio (float, optional): Ratio of MLP and embedding dimensions in attention blocks.
                Defaults to 4..
            norm_layer (type[nn.LayerNorm], optional): Layer class for [en/de]coder normalisation.
                Defaults to nn.LayerNorm.
            norm_token_loss (bool, optional): Whether to variance-normalise per-token loss.
                Defaults to False.
        """
        super().__init__()
        # --------------------------------------------------------------------------
        self.input_embedders = input_embedders
        encoder_embed_dim = input_embedders[0].embed_dim
        decoder_embed_dim = input_embedders[0].reconstruct_dim
        for id, ie in enumerate(input_embedders[1:], 1):
            assert ie.embed_dim == encoder_embed_dim, (
                "Input embedding dimensions inconsistent"
                f" ({encoder_embed_dim} of 0 != {ie.embed_dim} of {id})")
            assert ie.reconstruct_dim == decoder_embed_dim, (
                "Reconstruction embedding dimensions inconsistent"
                f" ({decoder_embed_dim} of 0 != {ie.reconstruct_dim} of {id})")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(encoder_depth)
            ])
        self.norm = norm_layer(encoder_embed_dim)
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
            ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        # --------------------------------------------------------------------------
        self.norm_token_loss = norm_token_loss
        self.initialize_weights()

    def initialize_weights(self):
        """Initialise layer weights and parameters, including in input embedders.
        """
        # initialization
        for input_embedder in self.input_embedders:
            input_embedder.initialize_weights()
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x: torch.Tensor, mask_ratio: float
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shuffle batched token embeddings and mask random selection.

        Args:
            x (torch.Tensor): Batched token embeddings, shape [B, L, D].
            mask_ratio (float): Proportion p of token embeddings to mask per batch.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
             * Randomly selected token embeddings, shape [B, pL, D].
             * Mask with 0 where token extracted, 1 otherwise, shape [B, L].
             * IDs with which to restore original, unshuffled token embeddings.
        """
        B, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, inputs: list[torch.Tensor], mask_ratio: float
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenise and embed inputs, randomly mask tokens, and encode using a transformer.

        Args:
            inputs (list[torch.Tensor]): Batched model inputs, for conversion by
                input_embedders into token embeddings of shapes ([B, L_i, D])_i.
            mask_ratio (float): Proportion p of token embeddings to mask per batch.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
             * Encodings of randomly selected input embeddings, shape [B, 1 + (1-p)sum(L_i), D].
             * Mask with 0 where token extracted, 1 otherwise, shape [B, sum(L_i)].
             * IDs with which to restore original, unshuffled token embeddings, shape [B, sum(L_i)].
        """
        # embed patches
        x = torch.cat([embed(inpt) for embed, inpt in zip(self.input_embedders, inputs)], dim=1)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor,
                        positions: Sequence[torch.Tensor | None]) -> list[torch.Tensor]:
        """Transform encoding of masked inputs, decode using a transformer, and reconstruct tokens.

        Args:
            x (torch.Tensor): Encodings of shuffled, masked tokens, shape [B, 1 + (1-p)L, D].
            ids_restore (torch.Tensor): IDs with which to restore original, unshuffled encodings,
                shape [B, L].
            positions (Sequence[torch.Tensor | None]): List with token positions for each input,
                if variable position embedding is implemented for that input, otherwise None.

        Returns:
            list[torch.Tensor]: Decoded tokens for each input, as reconstructed by input_embedders,
                shapes ([B, L_i, D_i])_i with sum(L_i) = L.
        """
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence, excluding cls token
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        # unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        # add pos embed
        x_ = x_ + torch.cat(
            [ie.embed_pos(pos) for ie, pos in zip(self.input_embedders, positions)], dim=1
            )
        # append cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        preds, start_patch = [], 1
        for embedder in self.input_embedders:
            end_patch = start_patch + embedder.n_patches
            preds.append(embedder.reconstruct_tokens(x[:, start_patch:end_patch]))
            start_patch = end_patch

        return preds

    def forward_loss(self, inputs: list[torch.Tensor], predictions: list[torch.Tensor],
                     mask: torch.Tensor) -> torch.Tensor:
        """Calculate masked-token MSE between batched inputs and model predictions.

        Args:
            inputs (list[torch.Tensor]):  Batched model inputs, for conversion by
                input_embedders into tokens of shapes ([B, L_i, D_i])_i.
            predictions (list[torch.Tensor]): Model predictions, shapes ([B, L_i, D_i])_i.
            mask (torch.Tensor): Mask with 1 where token masked, shape [B, sum(L_i)].

        Returns:
            torch.Tensor: Average mean squared error over the batch, shape [1].
        """
        losses = []
        for ie, inpt, prediction in zip(self.input_embedders, inputs, predictions):
            target = ie.tokenify(inpt)
            norm = target.var(dim=[2, 3], keepdim=True) if self.norm_token_loss else 1
            loss = (prediction - target) ** 2 / norm
            losses.append(loss.mean(dim=-1))

        loss = torch.concat(losses, dim=1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, inputs: list[torch.Tensor], mask_ratio: float = 0.75
                ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Apply MMMAE to inputs and return the loss, predictions, and mask.

        Args:
            inputs (list[torch.Tensor]): Batched model inputs.
            mask_ratio (float, optional): Proportion of token embeddings to mask per batch.
                Defaults to 0.75.

        Returns:
            tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
             * Mean squared error of model predictions, over masked tokens, shape [1].
             * Model predictions of input tokens, shapes ([B, L_i, D_i])_i.
             * Mask with 0 where token extracted, 1 otherwise, shape [B, sum(L_i)].
        """
        positions = [None for _ in range(len(inputs))]
        # Positions not yet implemented
        latent, mask, ids_restore = self.forward_encoder(inputs, mask_ratio)
        preds = self.forward_decoder(latent, ids_restore, positions)
        loss = self.forward_loss(inputs, preds, mask)
        return loss, preds, mask
