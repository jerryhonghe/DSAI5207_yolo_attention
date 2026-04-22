import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.relation_constructor import EfficientRelationConstructor


@pytest.fixture
def constructor():
    return EfficientRelationConstructor(temperature=0.07, max_tokens=1600)


def test_output_shape_p5(constructor):
    """P5 20x20 = 400 tokens, no downsampling needed."""
    x = torch.randn(2, 256, 20, 20)
    out = constructor(x)
    assert out.shape == (2, 400, 400)


def test_output_shape_p4(constructor):
    """P4 40x40 = 1600 tokens, exactly at max_tokens, no downsampling."""
    x = torch.randn(2, 128, 40, 40)
    out = constructor(x)
    assert out.shape == (2, 1600, 1600)


def test_output_shape_p3_downsampled(constructor):
    """P3 80x80 = 6400 tokens > max_tokens, downsampled to 40x40 = 1600."""
    x = torch.randn(2, 64, 80, 80)
    out = constructor(x)
    assert out.shape == (2, 1600, 1600)


def test_output_is_probability_distribution(constructor):
    """Each row of the relation matrix should sum to 1 (softmax output)."""
    x = torch.randn(2, 256, 20, 20)
    out = constructor(x)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_no_learnable_parameters(constructor):
    """The module should have zero learnable parameters."""
    assert len(list(constructor.parameters())) == 0
