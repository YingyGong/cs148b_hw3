"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from basics.model import Block


class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project each to d_model.

    Implemented with a strided Conv2d whose kernel size and stride both equal
    `patch_size`.

    Args:
        img_size:   Input image side length (assumed square). Must be divisible
                    by patch_size.
        patch_size: Side length of each patch in pixels.
        d_model:    Output embedding dimension per patch.

    Forward:
        x: (B, 3, img_size, img_size) float tensor.
        returns: (B, num_patches, d_model) where num_patches = (img_size // patch_size) ** 2.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # Hint: use nn.Conv2d with kernel_size=patch_size, stride=patch_size,
        # in_channels=3, out_channels=d_model. Then flatten the spatial dims
        # and transpose so each patch is a token.
        self.embeddings = nn.Conv2d(kernel_size=patch_size, stride=patch_size, in_channels=3, out_channels=d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: 
            x: (B, 3, img_size, img_size) float tensor.
        Returns:
            (B, num_patches, d_model) float tensor.
        """
        return self.embeddings(x).flatten(2, 3).transpose(-1, -2)


class ViT(nn.Module):
    """Vision Transformer.

    Pipeline:
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3. Add a learnable positional embedding of shape (1, num_patches+1, d_model).
      4. Pass the sequence through `num_blocks` Transformer Blocks
         (with is_decoder=False).
      5. Apply a final LayerNorm.
      6. Return only the [CLS] slice — shape (B, d_model).

    For §5 (VLM), you may want a `return_all_tokens=True` flag that returns the
    full (B, num_patches+1, d_model) sequence instead. Add it when you get there.

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # TODO: implement.
        # Hint: store self.cls_token as nn.Parameter(torch.zeros(1, 1, d_model))
        # and self.pos_embed as nn.Parameter(torch.zeros(1, num_patches+1, d_model)).
        # Use basics.model.Block(..., is_decoder=False) for the encoder blocks.
        self.patch_embed = PatchEmbeddings(img_size=img_size, patch_size=patch_size, d_model=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, d_model))
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.blocks = nn.ModuleList(Block(d_model=self.d_model, num_heads=self.num_heads,block_size=self.patch_embed.num_patches+1, is_decoder=False, dropout=self.dropout) for _ in range(num_blocks))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_embeddings = self.patch_embed(x)
        cls_expanded = self.cls_token.expand(x.shape[0], -1, -1)
        token = torch.cat([cls_expanded, patch_embeddings], dim=1)
        token += self.pos_embed

        # pass the sequence through num_blocks Transformer blocks
        for transformer_block in self.blocks:
            token = transformer_block(token)

        # apply a final layernorm
        ln = nn.LayerNorm(self.d_model)
        token = ln(token)

        cls_token = token[:, 0, :]
        return cls_token

