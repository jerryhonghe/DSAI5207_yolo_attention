import torch
import torch.nn as nn
import torch.nn.functional as F


class D2ProjectionHead(nn.Module):
    """
    d2 method projection head: 1x1 conv projecting YOLO neck features to DINOv2 dimension.
    Can be removed at inference time.
    """

    def __init__(self, student_channels, dinov2_dim=1024):
        """
        Args:
            student_channels: list, channel counts per level [P3_ch, P4_ch, P5_ch]
            dinov2_dim: DINOv2 embedding dimension
        """
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Conv2d(ch, dinov2_dim, 1, bias=False)
            for ch in student_channels
        ])

    def forward(self, features):
        """
        Args:
            features: list of [B, C_i, H_i, W_i]
        Returns:
            list of [B, dinov2_dim, H_i, W_i] (L2 normalized)
        """
        embeddings = []
        for proj, feat in zip(self.projectors, features):
            emb = proj(feat)
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb)
        return embeddings
