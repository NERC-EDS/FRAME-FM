# Files from Masked Autoencoders: A PyTorch Implementation

Files in this folder are adapted from files created by Facebook Research and downloaded from https://github.com/facebookresearch/mae/tree/main under the CC-BY-NC 4.0 licence at LICENCE.txt. Modifications are summarised below.

Modifications:
 * Licence location corrected in `*.py`.
 * Utilities import location corrected in `models_mae.py`, `engine_pretrain.py`, `main_pretrain.py`.
 * Whitespace altered for Flake8 conformity.
 * Import `torch._six.inf` updated to `torch.inf`.
 * Removed `timm.models.vision_transformer.Block` argument `qk_scale` in `models_mae.py`.
 * Updated `float32` and `np.float` to `np.float32` and `float` in `pos_embed.py`.
 * Generalised `patchify` and `unpatchify` to n-channel images in `models_mae.py`.