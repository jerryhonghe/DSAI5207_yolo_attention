"""
E2 Experiment: Attention Relation Distillation Trainer.

Constructs cosine-similarity relation matrices on YOLO neck P3/P4/P5 features
and aligns them with DINOv2 teacher relation matrices via KL divergence.
"""

import torch
from ultralytics.utils.torch_utils import de_parallel

from trainers.base_distill_trainer import BaseDistillTrainer
from models.teacher import DINOv2Teacher
from models.relation_constructor import EfficientRelationConstructor
from losses.relation_loss import RelationDistillationLoss


class RelationDistillTrainer(BaseDistillTrainer):
    """Trainer for attention relation distillation (E2)."""

    def setup_teacher(self):
        """Initialize frozen DINOv2 teacher."""
        teacher_weights = self.custom_cfg.get(
            'teacher_weights', 'weights/dinov2_vitl14_reg4_pretrain.pth'
        )
        temperature = self.custom_cfg.get('temperature', 0.07)
        teacher_layers = self.custom_cfg.get('teacher_layers', None)

        self.teacher = DINOv2Teacher(
            weights_path=teacher_weights,
            device=self.device,
            temperature=temperature,
            teacher_layers=teacher_layers,
        )

    def setup_distillation(self):
        """Initialize relation constructor, loss, and wrap criterion."""
        temperature = self.custom_cfg.get('temperature', 0.07)
        max_tokens = self.custom_cfg.get('max_relation_tokens', 1600)

        self.relation_constructor = EfficientRelationConstructor(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.relation_loss_fn = RelationDistillationLoss()
        self.lambda_distill = self.custom_cfg.get('lambda_distill', 1.0)
        self.level_weights = self.custom_cfg.get('level_weights', {
            'shallow': 1.0, 'middle': 1.0, 'deep': 1.0
        })

        # Wrap the original detection criterion with distillation loss
        model = de_parallel(self.model)
        original_criterion = model.criterion
        model.criterion = _RelationDistillCriterion(self, original_criterion)

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


class _RelationDistillCriterion:
    """Wraps original v8DetectionLoss and adds relation distillation loss."""

    def __init__(self, trainer, original_criterion):
        self.trainer = trainer
        self.original = original_criterion

    def __call__(self, preds, batch):
        # Original detection loss
        det_loss, det_loss_items = self.original(preds, batch)

        # Teacher forward
        images = batch['img']
        teacher_relations = self.trainer.teacher.get_relation_matrices(images)

        # Student: build relation matrices from neck features
        neck_feats = self.trainer.neck_features
        if len(neck_feats) < 3:
            return det_loss, det_loss_items

        level_names = ['shallow', 'middle', 'deep']
        relation_loss = torch.tensor(0.0, device=images.device)

        for i, level in enumerate(level_names):
            student_rel = self.trainer.relation_constructor(neck_feats[i])
            teacher_rel = teacher_relations[level]
            level_loss = self.trainer.relation_loss_fn(student_rel, teacher_rel)
            relation_loss = relation_loss + self.trainer.level_weights[level] * level_loss

        # Total loss
        total_loss = det_loss + self.trainer.lambda_distill * relation_loss

        # Extend loss items for logging
        distill_item = (self.trainer.lambda_distill * relation_loss).detach().unsqueeze(0)
        extended_items = torch.cat([det_loss_items, distill_item])

        return total_loss, extended_items
