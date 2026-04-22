import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationLoss(nn.Module):
    """SmoothL1 Loss for aligning student embeddings with teacher patch tokens."""

    def forward(self, student_embeddings, teacher_embeddings):
        """
        Args:
            student_embeddings: list of [B, D, H, W] (L2 normalized)
            teacher_embeddings: list of [B, D, H, W] (L2 normalized, detached)
        Returns:
            loss: scalar
        """
        total_loss = 0
        for s_emb, t_emb in zip(student_embeddings, teacher_embeddings):
            total_loss += F.smooth_l1_loss(s_emb, t_emb.detach())
        return total_loss / len(student_embeddings)
