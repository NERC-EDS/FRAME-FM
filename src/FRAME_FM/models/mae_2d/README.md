# Files from Masked Autoencoders: A PyTorch Implementation

Files in this folder were created by Facebook Research, downloaded from https://github.com/facebookresearch/mae/tree/main under the CC-BY-NC 4.0 licence at LICENCE.txt, and modified as described below.

Modifications:
 * Licence location corrected in `*.py`.
 * Utilities import location corrected in `models_mae.py`, `engine_pretrain.py`, `main_pretrain.py`.
 * Whitespace altered for Flake8 conformity.
 * Import `torch._six.inf` updated to `torch.inf`.
 * Removed `timm.models.vision_transformer.Block` argument `qk_scale` in `models_mae.py`.
 * Updated `float32` and `np.float` to `np.float32` and `float` in `pos_embed.py`.
 * Generalised `patchify` and `unpatchify` to n-channel images in `models_mae.py`.