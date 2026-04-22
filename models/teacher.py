"""
DINOv2 Teacher module for attention relation distillation.

Loads a frozen DINOv2-Large (with registers) from a local .pth file,
extracts intermediate transformer block features via forward hooks,
and constructs multi-level cosine-similarity relation matrices as
teacher signals for YOLO neck distillation.
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ImageNet normalization constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# DINOv2 patch size
PATCH_SIZE = 14
# Target input size: largest multiple of 14 that fits in 640 → 630 = 14 * 45
DINOV2_INPUT_SIZE = 630


def prepare_for_dinov2(yolo_images: torch.Tensor) -> torch.Tensor:
    """Convert YOLO-format images to DINOv2-ready inputs.

    Args:
        yolo_images: [B, 3, 640, 640] float tensor in [0, 1] RGB.

    Returns:
        [B, 3, 630, 630] ImageNet-normalised tensor on the same device.
    """
    mean = IMAGENET_MEAN.to(yolo_images.device, yolo_images.dtype)
    std  = IMAGENET_STD.to(yolo_images.device, yolo_images.dtype)

    # Resize 640 -> 630 (must be multiple of patch_size=14)
    x = F.interpolate(
        yolo_images, size=(DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE),
        mode="bilinear", align_corners=False,
    )

    # ImageNet normalisation
    x = (x - mean) / std
    return x


class DINOv2Teacher:
    """Frozen DINOv2 teacher that produces multi-level relation matrices.

    The model is loaded entirely from a local ``.pth`` checkpoint so that
    training can proceed on an offline server.  Model definitions come from
    the ``third_party/dinov2`` source tree.

    Parameters
    ----------
    weights_path : str
        Path to ``dinov2_vitl14_reg4_pretrain.pth``.
    device : str
        Target device (default ``'cuda'``).
    temperature : float
        Softmax temperature for relation matrices (default ``0.07``).
    target_sizes : dict | None
        Spatial size for each level after interpolation.
        Defaults to ``{'shallow': 40, 'middle': 40, 'deep': 20}``.
    teacher_layers : dict | None
        Transformer block indices per level.
        Defaults to ``{'shallow': [6,7,8], 'middle': [12,13,14], 'deep': [20,21,22]}``.
    """

    # DINOv2-Large with registers architecture constants
    NUM_REGISTER_TOKENS = 4
    NUM_PREFIX_TOKENS = 1 + NUM_REGISTER_TOKENS  # CLS + 4 registers = 5
    EMBED_DIM = 1024
    NUM_HEADS = 16
    NUM_BLOCKS = 24

    def __init__(
        self,
        weights_path: str,
        device: str = "cuda",
        temperature: float = 0.07,
        target_sizes: dict | None = None,
        teacher_layers: dict | None = None,
    ):
        self.device = device
        self.temperature = temperature

        self.target_sizes = target_sizes or {
            "shallow": 40,
            "middle": 40,
            "deep": 20,
        }
        self.teacher_layers = teacher_layers or {
            "shallow": [6, 7, 8],
            "middle": [12, 13, 14],
            "deep": [20, 21, 22],
        }

        # Feature cache filled by hooks
        self.feature_maps: dict[str, torch.Tensor] = {}

        # Load model
        self.model = self._load_model(weights_path)
        self._register_hooks()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, weights_path: str) -> torch.nn.Module:
        """Build DINOv2-Large-reg from third_party source and load weights."""
        # Add third_party/dinov2 to sys.path so we can import its modules
        project_root = Path(__file__).resolve().parent.parent
        dinov2_src = str(project_root / "third_party" / "dinov2")
        if dinov2_src not in sys.path:
            sys.path.insert(0, dinov2_src)

        # Import the model builder from the DINOv2 source tree
        from dinov2.models.vision_transformer import vit_large  # type: ignore

        model = vit_large(
            patch_size=PATCH_SIZE,
            num_register_tokens=self.NUM_REGISTER_TOKENS,
            img_size=526,  # default; actual resolution can differ at runtime
            block_chunks=0,
        )

        # Load checkpoint
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)

        model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        return model

    # ------------------------------------------------------------------
    # Forward hooks
    # ------------------------------------------------------------------

    def _register_hooks(self) -> None:
        """Register forward hooks on transformer blocks to capture features."""
        for level_name, layer_indices in self.teacher_layers.items():
            for idx in layer_indices:
                block = self.model.blocks[idx]
                block.register_forward_hook(
                    self._make_feature_hook(f"{level_name}_{idx}")
                )

    def _make_feature_hook(self, name: str):
        def hook(module, input, output):
            # Block output shape: [B, N, D]  (N = num_prefix + num_patches)
            self.feature_maps[name] = output
        return hook

    # ------------------------------------------------------------------
    # Relation matrix extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_relation_matrices(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute multi-level cosine-similarity relation matrices.

        Args:
            images: [B, 3, 640, 640] YOLO-format (float [0, 1] RGB).

        Returns:
            Dictionary with keys ``'shallow'``, ``'middle'``, ``'deep'``,
            each mapping to a ``[B, tgt*tgt, tgt*tgt]`` softmax-normalised
            relation matrix.
        """
        # Preprocess
        x = prepare_for_dinov2(images)

        # Forward pass with float16 autocast
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _ = self.model(x)

        # Grid size for DINOv2 at 630: 630 / 14 = 45
        grid_size = DINOV2_INPUT_SIZE // PATCH_SIZE  # 45

        relations: dict[str, torch.Tensor] = {}
        for level_name, layer_indices in self.teacher_layers.items():
            # Collect features from the relevant layers and average
            feats = []
            for idx in layer_indices:
                key = f"{level_name}_{idx}"
                # Remove CLS + register prefix tokens
                feat = self.feature_maps[key][:, self.NUM_PREFIX_TOKENS:, :]
                feats.append(feat.float())
            avg_feat = torch.stack(feats).mean(dim=0)  # [B, num_patches, D]

            B, N, D = avg_feat.shape
            # Reshape to 2D spatial grid
            feat_2d = avg_feat.reshape(B, grid_size, grid_size, D).permute(0, 3, 1, 2)
            # [B, D, 45, 45]

            # Bilinear interpolate to target spatial size
            tgt = self.target_sizes[level_name]
            feat_resized = F.interpolate(
                feat_2d, size=(tgt, tgt), mode="bilinear", align_corners=False,
            )  # [B, D, tgt, tgt]

            # Flatten spatial dims
            feat_flat = feat_resized.reshape(B, D, tgt * tgt).permute(0, 2, 1)
            # [B, tgt*tgt, D]

            # L2 normalise along feature dim
            feat_norm = F.normalize(feat_flat, p=2, dim=-1)

            # Cosine similarity matrix
            sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
            # [B, tgt*tgt, tgt*tgt]

            # Temperature-scaled softmax -> probability distribution
            relation = F.softmax(sim_matrix / self.temperature, dim=-1)
            relations[level_name] = relation

        # Clear cached feature maps
        self.feature_maps.clear()

        return relations

    # ------------------------------------------------------------------
    # Patch feature extraction (for d2 feature-distillation baseline)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract last-layer patch token features for feature distillation.

        Args:
            images: [B, 3, 640, 640] YOLO-format.

        Returns:
            [B, num_patches, 1024] patch features from the last block.
        """
        x = prepare_for_dinov2(images)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            _ = self.model(x)

        # Last block features (always captured if layer 22 is in deep)
        # For safety, directly grab from the model's last block output
        # We use the hook data if available, else run get_intermediate_layers
        last_key = None
        for level_name, layer_indices in self.teacher_layers.items():
            for idx in layer_indices:
                key = f"{level_name}_{idx}"
                if key in self.feature_maps:
                    last_key = key  # keep updating; the highest idx wins

        if last_key is not None:
            feat = self.feature_maps[last_key][:, self.NUM_PREFIX_TOKENS:, :]
        else:
            # Fallback: run the model again and grab output of last block
            # Register a temporary hook
            output_holder = {}

            def _hook(module, input, output):
                output_holder["feat"] = output

            handle = self.model.blocks[-1].register_forward_hook(_hook)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _ = self.model(x)
            handle.remove()
            feat = output_holder["feat"][:, self.NUM_PREFIX_TOKENS:, :]

        self.feature_maps.clear()
        return feat.float()  # [B, num_patches, D=1024]
