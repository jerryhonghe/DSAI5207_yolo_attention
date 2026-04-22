import pytest
import torch
import torch.nn.functional as F

from losses.relation_loss import RelationDistillationLoss
from losses.feature_loss import FeatureDistillationLoss


class TestRelationDistillationLoss:
    def test_identical_distributions_near_zero_loss(self):
        """Same distribution should yield KL divergence near zero."""
        loss_fn = RelationDistillationLoss()
        # Create a valid probability distribution via softmax
        logits = torch.randn(2, 8, 8)
        dist = F.softmax(logits, dim=-1)
        loss = loss_fn(dist, dist)
        assert loss.item() < 1e-5

    def test_different_distributions_positive_loss(self):
        """Different distributions should yield positive KL divergence."""
        loss_fn = RelationDistillationLoss()
        student = F.softmax(torch.randn(2, 8, 8), dim=-1)
        teacher = F.softmax(torch.randn(2, 8, 8), dim=-1)
        loss = loss_fn(student, teacher)
        assert loss.item() > 0

    def test_loss_gradient_flows_to_student(self):
        """Gradient should flow to student but not to teacher."""
        loss_fn = RelationDistillationLoss()
        student = F.softmax(torch.randn(2, 8, 8), dim=-1).requires_grad_(True)
        teacher = F.softmax(torch.randn(2, 8, 8), dim=-1).requires_grad_(True)
        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None
        assert teacher.grad is None


class TestFeatureDistillationLoss:
    def test_identical_features_near_zero_loss(self):
        """Same features should yield SmoothL1 loss near zero."""
        loss_fn = FeatureDistillationLoss()
        feat = torch.randn(2, 64, 8, 8)
        feat = F.normalize(feat, dim=1)
        loss = loss_fn([feat], [feat])
        assert loss.item() < 1e-5

    def test_different_features_positive_loss(self):
        """Different features should yield positive SmoothL1 loss."""
        loss_fn = FeatureDistillationLoss()
        s_feat = F.normalize(torch.randn(2, 64, 8, 8), dim=1)
        t_feat = F.normalize(torch.randn(2, 64, 8, 8), dim=1)
        loss = loss_fn([s_feat], [t_feat])
        assert loss.item() > 0
