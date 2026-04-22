import torch
import pytest
from models.d2_head import D2ProjectionHead


@pytest.fixture
def head():
    return D2ProjectionHead(student_channels=[256, 512, 512], dinov2_dim=1024)


@pytest.fixture
def dummy_features():
    B = 2
    return [
        torch.randn(B, 256, 80, 80),
        torch.randn(B, 512, 40, 40),
        torch.randn(B, 512, 20, 20),
    ]


def test_output_shapes(head, dummy_features):
    outputs = head(dummy_features)
    assert len(outputs) == 3
    assert outputs[0].shape == (2, 1024, 80, 80)
    assert outputs[1].shape == (2, 1024, 40, 40)
    assert outputs[2].shape == (2, 1024, 20, 20)


def test_output_is_l2_normalized(head, dummy_features):
    outputs = head(dummy_features)
    for emb in outputs:
        norms = torch.norm(emb, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_has_learnable_parameters(head):
    params = list(head.parameters())
    assert len(params) > 0
