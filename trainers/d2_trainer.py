"""
E3 Experiment: Feature Distillation Trainer (d2 method reproduction).

Projects YOLO neck features to DINOv2 dimension via 1x1 conv,
aligns with DINOv2 patch tokens using SmoothL1 loss.
"""

import torch
import torch.nn.functional as F
from ultralytics.utils.torch_utils import de_parallel

from trainers.base_distill_trainer import BaseDistillTrainer
from models.teacher import DINOv2Teacher
from models.d2_head import D2ProjectionHead
from losses.feature_loss import FeatureDistillationLoss


class D2DistillTrainer(BaseDistillTrainer):
    """Trainer for feature distillation d2 method (E3)."""

    def setup_teacher(self):
        """Initialize frozen DINOv2 teacher."""
        teacher_weights = self.custom_cfg.get(
            'teacher_weights', 'weights/dinov2_vitl14_reg4_pretrain.pth'
        )
        self.teacher = DINOv2Teacher(
            weights_path=teacher_weights,
            device=self.device,
        )

    def setup_distillation(self):
        """Initialize d2 projection head, feature loss, and wrap criterion."""
        dinov2_dim = self.custom_cfg.get('dinov2_dim', 1024)
        self.lambda_distill = self.custom_cfg.get('lambda_distill', 1.0)

        # Get YOLO neck channel sizes from Detect head
        model = de_parallel(self.model)
        detect_head = model.model[-1]
        student_channels = list(detect_head.ch)

        # Create projection head and move to device
        self.d2_head = D2ProjectionHead(student_channels, dinov2_dim=dinov2_dim)
        self.d2_head.to(self.device)
        self.d2_head.train()

        self.feature_loss_fn = FeatureDistillationLoss()

        # Add d2_head parameters to optimizer
        self.optimizer.add_param_group({
            'params': list(self.d2_head.parameters()),
            'lr': self.optimizer.param_groups[0]['lr'],
            'momentum': self.optimizer.param_groups[0].get('momentum', 0.9),
            'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 5e-4),
        })

        # Wrap criterion
        original_criterion = model.criterion
        model.criterion = _D2DistillCriterion(self, original_criterion)

    def get_validator(self):
        """Extend loss_names to include distillation loss."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "distill_loss"
        return super().get_validator()

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Label extended loss items for logging."""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = loss_items[:len(keys)] if len(loss_items) > len(keys) else loss_items
            if len(loss_items) < len(keys):
                keys = keys[:len(loss_items)]
            return dict(zip(keys, loss_items))
        return keys


class _D2DistillCriterion:
    """Wraps original v8DetectionLoss and adds feature distillation loss."""

    def __init__(self, trainer, original_criterion):
        self.trainer = trainer
        self.original = original_criterion

    def __call__(self, preds, batch):
        det_loss, det_loss_items = self.original(preds, batch)

        images = batch['img']

        # Teacher: extract last-layer patch features
        teacher_patch_tokens = self.trainer.teacher.get_patch_features(images)

        neck_feats = self.trainer.neck_features
        if len(neck_feats) < 3:
            return det_loss, det_loss_items

        # Reshape teacher tokens to 2D spatial grid
        B = images.shape[0]
        N, D = teacher_patch_tokens.shape[1], teacher_patch_tokens.shape[2]
        H_t = W_t = int(N ** 0.5)
        teacher_2d = teacher_patch_tokens.reshape(B, H_t, W_t, D).permute(0, 3, 1, 2)

        # Resize teacher features to match each neck level's spatial size
        teacher_resized = []
        for feat in neck_feats:
            H, W = feat.shape[2], feat.shape[3]
            t_feat = F.interpolate(teacher_2d, size=(H, W),
                                   mode='bilinear', align_corners=False)
            t_feat = F.normalize(t_feat, p=2, dim=1)
            teacher_resized.append(t_feat)

        # Student: project neck features to DINOv2 dimension
        student_embeddings = self.trainer.d2_head(neck_feats)

        # Feature loss
        feature_loss = self.trainer.feature_loss_fn(student_embeddings, teacher_resized)

        total_loss = det_loss + self.trainer.lambda_distill * feature_loss

        distill_item = (self.trainer.lambda_distill * feature_loss).detach().unsqueeze(0)
        extended_items = torch.cat([det_loss_items, distill_item])

        return total_loss, extended_items
