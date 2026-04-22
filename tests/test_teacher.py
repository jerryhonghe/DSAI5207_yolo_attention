"""
Unit tests for the DINOv2 Teacher module.

These tests verify the mathematical operations (cosine similarity, softmax,
spatial resize, prefix token removal) without requiring real model weights.
Integration tests that need the actual DINOv2 checkpoint are marked with
``@pytest.mark.slow``.
"""

import pytest
import torch
import torch.nn.functional as F

from models.teacher import (
    DINOV2_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PATCH_SIZE,
    DINOv2Teacher,
    prepare_for_dinov2,
)


# -----------------------------------------------------------------------
# Unit tests (no model weights required)
# -----------------------------------------------------------------------


class TestCosineSimRelationMatrixShape:
    """Verify that cosine-similarity -> softmax produces [B, N, N]."""

    @pytest.mark.parametrize("B,N,D", [(2, 400, 256), (1, 1600, 512)])
    def test_cosine_similarity_relation_matrix_shape(self, B, N, D):
        feat = torch.randn(B, N, D)
        feat_norm = F.normalize(feat, p=2, dim=-1)
        sim = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
        relation = F.softmax(sim / 0.07, dim=-1)

        assert relation.shape == (B, N, N)


class TestRelationMatrixIsProbabilityDistribution:
    """Each row of the relation matrix after softmax must sum to 1."""

    @pytest.mark.parametrize("B,N,D", [(2, 100, 64), (1, 400, 128)])
    def test_relation_matrix_is_probability_distribution(self, B, N, D):
        feat = torch.randn(B, N, D)
        feat_norm = F.normalize(feat, p=2, dim=-1)
        sim = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
        relation = F.softmax(sim / 0.07, dim=-1)

        row_sums = relation.sum(dim=-1)  # [B, N]
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5,
        )

    def test_all_values_non_negative(self):
        feat = torch.randn(2, 50, 32)
        feat_norm = F.normalize(feat, p=2, dim=-1)
        sim = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
        relation = F.softmax(sim / 0.07, dim=-1)
        assert (relation >= 0).all()


class TestSpatialResizePreservesBatch:
    """Bilinear interpolation must preserve batch dimension."""

    @pytest.mark.parametrize(
        "B,D,H_in,H_out",
        [(2, 1024, 45, 40), (4, 1024, 45, 20), (1, 256, 80, 40)],
    )
    def test_spatial_resize_preserves_batch(self, B, D, H_in, H_out):
        feat_2d = torch.randn(B, D, H_in, H_in)
        resized = F.interpolate(
            feat_2d, size=(H_out, H_out), mode="bilinear", align_corners=False,
        )
        assert resized.shape == (B, D, H_out, H_out)


class TestPrefixTokenRemoval:
    """CLS + 4 register tokens must be correctly stripped.

    DINOv2-Large with registers at 630x630:
      grid = 630 / 14 = 45   ->  45*45 = 2025 patch tokens
      total sequence length = 2025 + 5 = 2030  (CLS + 4 registers)
    """

    def test_prefix_token_removal(self):
        num_prefix = 5  # 1 CLS + 4 registers
        num_patches = 45 * 45  # 2025
        total_tokens = num_patches + num_prefix  # 2030
        B, D = 2, 1024

        tokens = torch.randn(B, total_tokens, D)
        patch_tokens = tokens[:, num_prefix:, :]

        assert patch_tokens.shape == (B, num_patches, D)
        assert patch_tokens.shape[1] == 2025

    def test_prefix_token_count_matches_class(self):
        assert DINOv2Teacher.NUM_PREFIX_TOKENS == 5
        assert DINOv2Teacher.NUM_REGISTER_TOKENS == 4


class TestPrepareForDINOv2:
    """Verify image preprocessing pipeline."""

    def test_output_size(self):
        imgs = torch.rand(2, 3, 640, 640)
        out = prepare_for_dinov2(imgs)
        assert out.shape == (2, 3, 630, 630)

    def test_normalisation_changes_values(self):
        imgs = torch.rand(2, 3, 640, 640)
        out = prepare_for_dinov2(imgs)
        # After ImageNet normalisation values should not remain in [0, 1]
        assert out.min() < 0.0 or out.max() > 1.0

    def test_dinov2_input_is_patch_aligned(self):
        assert DINOV2_INPUT_SIZE % PATCH_SIZE == 0
        assert DINOV2_INPUT_SIZE // PATCH_SIZE == 45


# -----------------------------------------------------------------------
# Integration tests (require real DINOv2 weights + GPU)
# -----------------------------------------------------------------------


@pytest.mark.slow
class TestDINOv2TeacherIntegration:
    """End-to-end tests that need a real checkpoint.

    Run with: ``pytest -m slow``
    """

    WEIGHTS = "weights/dinov2_vitl14_reg4_pretrain.pth"

    @pytest.fixture(autouse=True)
    def _setup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        import os
        if not os.path.exists(self.WEIGHTS):
            pytest.skip(f"Weights not found at {self.WEIGHTS}")
        self.teacher = DINOv2Teacher(
            weights_path=self.WEIGHTS,
            device="cuda",
            temperature=0.07,
        )

    def test_relation_matrices_shapes(self):
        imgs = torch.rand(2, 3, 640, 640, device="cuda")
        rels = self.teacher.get_relation_matrices(imgs)

        assert rels["shallow"].shape == (2, 1600, 1600)
        assert rels["middle"].shape == (2, 1600, 1600)
        assert rels["deep"].shape == (2, 400, 400)

    def test_relation_matrices_are_distributions(self):
        imgs = torch.rand(1, 3, 640, 640, device="cuda")
        rels = self.teacher.get_relation_matrices(imgs)

        for level in ("shallow", "middle", "deep"):
            row_sums = rels[level].sum(dim=-1)
            torch.testing.assert_close(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-4,
                rtol=1e-4,
            )

    def test_patch_features_shape(self):
        imgs = torch.rand(1, 3, 640, 640, device="cuda")
        feats = self.teacher.get_patch_features(imgs)
        assert feats.shape == (1, 2025, 1024)
