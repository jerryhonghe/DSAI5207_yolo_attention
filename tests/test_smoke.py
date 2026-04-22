"""
End-to-end smoke tests verifying components integrate correctly.
Uses random tensors only — no real model weights needed.
"""
import torch
import torch.nn.functional as F

from models.relation_constructor import EfficientRelationConstructor
from losses.relation_loss import RelationDistillationLoss
from losses.feature_loss import FeatureDistillationLoss
from models.d2_head import D2ProjectionHead


class TestRelationDistillPipeline:
    """Simulate E2 full pipeline: teacher relations + student relations -> KL -> backward."""

    def test_full_pipeline_gradient_flow(self):
        B = 2
        temperature = 0.07
        constructor = EfficientRelationConstructor(temperature=temperature, max_tokens=1600)
        loss_fn = RelationDistillationLoss()

        # Simulated neck features
        p3 = torch.randn(B, 256, 80, 80, requires_grad=True)
        p4 = torch.randn(B, 512, 40, 40, requires_grad=True)
        p5 = torch.randn(B, 512, 20, 20, requires_grad=True)

        # Student relation matrices
        s_shallow = constructor(p3)  # [B, 1600, 1600]
        s_middle = constructor(p4)   # [B, 1600, 1600]
        s_deep = constructor(p5)     # [B, 400, 400]

        # Simulated teacher relation matrices
        t_shallow = F.softmax(torch.randn(B, 1600, 1600) / temperature, dim=-1)
        t_middle = F.softmax(torch.randn(B, 1600, 1600) / temperature, dim=-1)
        t_deep = F.softmax(torch.randn(B, 400, 400) / temperature, dim=-1)

        # Loss
        loss = (loss_fn(s_shallow, t_shallow) +
                loss_fn(s_middle, t_middle) +
                loss_fn(s_deep, t_deep))

        assert loss.item() > 0

        # Gradient flows back to neck features
        loss.backward()
        assert p3.grad is not None
        assert p4.grad is not None
        assert p5.grad is not None

    def test_output_shapes(self):
        B = 2
        constructor = EfficientRelationConstructor(temperature=0.07, max_tokens=1600)

        p3 = torch.randn(B, 256, 80, 80)
        p4 = torch.randn(B, 512, 40, 40)
        p5 = torch.randn(B, 512, 20, 20)

        assert constructor(p3).shape == (B, 1600, 1600)
        assert constructor(p4).shape == (B, 1600, 1600)
        assert constructor(p5).shape == (B, 400, 400)


class TestD2DistillPipeline:
    """Simulate E3 full pipeline: teacher tokens -> resize -> student proj -> SmoothL1 -> backward."""

    def test_full_pipeline_gradient_flow(self):
        B = 2
        dinov2_dim = 1024
        student_channels = [256, 512, 512]

        d2_head = D2ProjectionHead(student_channels, dinov2_dim=dinov2_dim)
        loss_fn = FeatureDistillationLoss()

        features = [
            torch.randn(B, 256, 80, 80),
            torch.randn(B, 512, 40, 40),
            torch.randn(B, 512, 20, 20),
        ]

        student_embs = d2_head(features)

        # Simulated teacher patch tokens
        teacher_tokens = torch.randn(B, 2025, dinov2_dim)  # 45*45
        H_t = W_t = 45
        teacher_2d = teacher_tokens.reshape(B, H_t, W_t, dinov2_dim).permute(0, 3, 1, 2)

        teacher_resized = []
        for feat in features:
            H, W = feat.shape[2], feat.shape[3]
            t = F.interpolate(teacher_2d, size=(H, W), mode='bilinear', align_corners=False)
            t = F.normalize(t, p=2, dim=1)
            teacher_resized.append(t)

        loss = loss_fn(student_embs, teacher_resized)
        assert loss.item() > 0

        loss.backward()
        for proj in d2_head.projectors:
            assert proj.weight.grad is not None

    def test_projection_output_shapes(self):
        B = 2
        d2_head = D2ProjectionHead([256, 512, 512], dinov2_dim=1024)

        features = [
            torch.randn(B, 256, 80, 80),
            torch.randn(B, 512, 40, 40),
            torch.randn(B, 512, 20, 20),
        ]

        embs = d2_head(features)
        assert embs[0].shape == (B, 1024, 80, 80)
        assert embs[1].shape == (B, 1024, 40, 40)
        assert embs[2].shape == (B, 1024, 20, 20)
