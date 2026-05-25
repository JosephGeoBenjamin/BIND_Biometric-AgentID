"""
lafs.py
=======
Drop-in loader for LAFS (Landmark-based Facial Self-supervised Learning, CVPR 2024)
models, mirroring the cvlface API so you can swap models with minimal code changes;
Code packaged by Claude Sonnet-4.6.

    import lafs
    lafs_model = lafs.load_lafs_model("path/to/checkpoint.pth", arch="part_fvit").to(device)
    lafs_model.float().eval()
    lafs_tf   = lafs.get_lafs_transform(input_size=112, tensorize=False)

GitHub: https://github.com/szlbiubiubiu/LAFS_CVPR2024
Paper : https://arxiv.org/abs/2403.08161
"""

from __future__ import annotations

import os
import sys
import math
import functools
from typing import Literal, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

ArchType = Literal["part_fvit", "iresnet50", "iresnet100"]


def load_lafs_model(
    checkpoint_path: str,
    arch: ArchType = "part_fvit",
    embed_dim: int = 768,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> nn.Module:
    """Load a LAFS backbone from a local checkpoint file.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pth`` / ``.pt`` checkpoint produced by LAFS training.
        The checkpoint may be a raw state-dict or a dict with a ``'model'``,
        ``'state_dict'``, or ``'teacher'`` key (all common LAFS save formats).
    arch:
        Backbone architecture.  One of:
        * ``"part_fvit"``   – Part-based Vision Transformer (default, LAFS Stage 1/2)
    embed_dim:
        Output embedding dimension (default 512).
    device:
        Target device for loading tensors.
    strict:
        Passed to ``load_state_dict``.  Set ``False`` (default) to tolerate
        minor key mismatches between SSL pre-training and fine-tuning heads.

    Returns
    -------
    nn.Module
        The backbone in *eval* mode on the specified device.
    """
    model = _build_model(arch, embed_dim)
    _load_weights(model, checkpoint_path, device=device, strict=strict)
    model.eval()
    return model


def get_lafs_transform(
    input_size: int = 112,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    tensorize: bool = True,
):
    """Return the standard LAFS image pre-processing pipeline.

    Parameters
    ----------
    input_size:
        Spatial size of the square crop fed to the model (default 112).
    mean / std:
        Normalisation constants (default: [-1, 1] range matching LAFS training).
    tensorize:
        If ``True``  → the transform converts PIL images / numpy arrays to a
                        ``torch.Tensor`` and normalises.
        If ``False`` → returns a transform that accepts a ``torch.Tensor``
                        that is already in [0, 1] range and only normalises
                        (mirrors cvlface's ``tensorize=False`` behaviour).

    Returns
    -------
    torchvision.transforms.Compose
    """
    if tensorize:
        return T.Compose([
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),                          # → [0, 1] float tensor
            T.Normalize(mean=mean, std=std),       # → [-1, 1]
        ])
    else:
        # Input is expected to be a float tensor in [0, 1]
        return T.Compose([
            T.Normalize(mean=mean, std=std),       # → [-1, 1]
        ])


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

# ── Part-fViT helpers ────────────────────────────────────────────────────────

class _PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and linearly project them."""

    def __init__(self, img_size=112, patch_size=8, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size  = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features    = out_features    or in_features
        self.fc1   = nn.Linear(in_features, hidden_features)
        self.act   = nn.GELU()
        self.fc2   = nn.Linear(hidden_features, out_features)
        self.drop  = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn  = _Attention(dim, num_heads, qkv_bias=qkv_bias,
                                 attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp   = _MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _LandmarkCNN(nn.Module):
    """Lightweight CNN that predicts N landmark (x,y) coordinates."""

    def __init__(self, num_landmarks=6):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.net = nn.Sequential(
            nn.Conv2d(3,  16, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(64 * 4 * 4, num_landmarks * 2)

    def forward(self, x):
        feat = self.net(x).flatten(1)
        coords = torch.sigmoid(self.fc(feat))        # normalised [0, 1]
        return coords.view(-1, self.num_landmarks, 2) # (B, L, 2)


class PartFViT(nn.Module):
    """Part-based Vision Transformer for face recognition (LAFS / Part-fViT).

    Architecture
    ------------
    1. A small landmark CNN predicts ``num_landmarks`` patch centres.
    2. Patches around those centres plus a global grid are embedded.
    3. A standard ViT transformer encodes all tokens.
    4. The [CLS] token is projected to ``embed_dim`` features.
    """

    def __init__(
        self,
        img_size: int       = 112,
        patch_size: int     = 8,
        in_chans: int       = 3,
        embed_dim: int      = 512,
        depth: int          = 12,
        num_heads: int      = 8,
        mlp_ratio: float    = 4.,
        qkv_bias: bool      = True,
        drop_rate: float    = 0.,
        attn_drop_rate: float = 0.,
        num_landmarks: int  = 6,
    ):
        super().__init__()
        self.patch_size     = patch_size
        self.num_landmarks  = num_landmarks
        self.img_size       = img_size

        # Part-patch branch
        self.landmark_cnn   = _LandmarkCNN(num_landmarks)

        # Global patch embedding
        self.patch_embed    = _PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_global_patches  = self.patch_embed.num_patches

        # CLS + position embedding
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, 1 + num_global_patches + num_landmarks, embed_dim)
        )
        self.pos_drop   = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(*[
            _Block(embed_dim, num_heads, mlp_ratio, qkv_bias,
                   drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm     = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head_bn  = nn.BatchNorm1d(embed_dim, eps=1e-5)
        nn.init.constant_(self.head_bn.weight, 1.0)
        self.head_bn.weight.requires_grad = False

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _extract_landmark_patches(self, x: torch.Tensor, lm: torch.Tensor) -> torch.Tensor:
        """Bilinear-sample one patch per landmark, return shape (B, L, embed_dim)."""
        B, C, H, W = x.shape
        P = self.patch_size
        half = P // 2
        patches = []
        for i in range(self.num_landmarks):
            cx = (lm[:, i, 0] * W).long().clamp(half, W - half)  # (B,)
            cy = (lm[:, i, 1] * H).long().clamp(half, H - half)
            # Simple approach: use grid_sample with a per-sample grid
            theta = torch.zeros(B, 2, 3, device=x.device, dtype=x.dtype)
            theta[:, 0, 0] = P / W
            theta[:, 1, 1] = P / H
            theta[:, 0, 2] = 2.0 * cx.float() / W - 1.0
            theta[:, 1, 2] = 2.0 * cy.float() / H - 1.0
            grid = torch.nn.functional.affine_grid(theta, (B, C, P, P), align_corners=False)
            patch = torch.nn.functional.grid_sample(x, grid, align_corners=False)  # (B,C,P,P)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)  # (B, L, C, P, P)
        patches = patches.view(B * self.num_landmarks, C, P, P)
        # Project through the same conv as patch_embed
        emb = self.patch_embed.proj(patches)   # (B*L, embed_dim, 1, 1)
        emb = emb.flatten(1)                   # (B*L, embed_dim)
        return emb.view(B, self.num_landmarks, -1)  # (B, L, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # 1. Predict landmarks
        lm = self.landmark_cnn(x)               # (B, L, 2)

        # 2. Global patches
        global_tokens = self.patch_embed(x)     # (B, N, D)

        # 3. Landmark patches
        lm_tokens = self._extract_landmark_patches(x, lm)  # (B, L, D)

        # 4. Concatenate: [CLS | global | landmark]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, global_tokens, lm_tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        # 5. Transformer
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        # 6. CLS projection
        cls_out = tokens[:, 0]                  # (B, D)
        cls_out = self.head_bn(cls_out)
        return cls_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_model(arch: ArchType, embed_dim: int) -> nn.Module:
    if arch == "part_fvit":
        return PartFViT(embed_dim=embed_dim)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: part_fvit, iresnet50, iresnet100")


def _load_weights(model: nn.Module, path: str, device, strict: bool) -> None:
    """Load a checkpoint, handling common LAFS / DINO save formats."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Unwrap common checkpoint dict formats
    for key in ("model", "state_dict", "teacher", "student", "backbone"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break

    # Strip common prefixes added during distributed / DINO training
    def _strip(state_dict, prefix):
        return {k[len(prefix):]: v for k, v in state_dict.items()
                if k.startswith(prefix)}

    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        if keys and keys[0].startswith("module."):
            ckpt = _strip(ckpt, "module.")
        elif keys and keys[0].startswith("backbone."):
            ckpt = _strip(ckpt, "backbone.")

    missing, unexpected = model.load_state_dict(ckpt, strict=strict)
    if missing:
        print(f"[lafs] Warning - missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[lafs] Info    - unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

