import torch
import torch.nn as nn


class RelationDistillationLoss(nn.Module):
    """KL Divergence loss for aligning teacher/student relation distributions."""

    def __init__(self):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_relation, teacher_relation):
        """
        Args:
            student_relation: [B, N, N] softmax normalized
            teacher_relation: [B, N, N] softmax normalized
        Returns:
            loss: scalar
        """
        return self.kl_loss(
            student_relation.log(),
            teacher_relation.detach()
        )
