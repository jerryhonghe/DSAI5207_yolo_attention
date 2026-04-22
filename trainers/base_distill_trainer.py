"""
Base distillation trainer — common infrastructure for all distillation experiments.

Inherits from Ultralytics DetectionTrainer and adds:
- Custom config extraction (teacher_weights, lambda, temperature, etc.)
- Neck feature hook on Detect head to capture P3/P4/P5
- Validation frequency control (every val_interval epochs)
- Abstract interface for teacher setup and distillation components
"""

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import de_parallel


class BaseDistillTrainer(DetectionTrainer):
    """Base trainer for distillation experiments.

    Subclasses must implement:
    - setup_teacher(): initialize the frozen DINOv2 teacher
    - setup_distillation(): initialize distillation components and wrap criterion
    """

    # Keys that are custom to our distillation config, not known by Ultralytics
    CUSTOM_KEYS = [
        'experiment', 'teacher_weights', 'lambda_distill', 'temperature',
        'max_relation_tokens', 'teacher_layers', 'level_weights',
        'val_interval', 'dinov2_dim',
    ]

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # Extract custom params before Ultralytics sees them
        self.custom_cfg = {}
        if overrides:
            for key in list(overrides.keys()):
                if key in self.CUSTOM_KEYS:
                    self.custom_cfg[key] = overrides.pop(key)

        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        self.val_interval = self.custom_cfg.get('val_interval', 10)
        self.neck_features = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def setup_teacher(self):
        """Initialize frozen DINOv2 teacher. Override in subclass."""
        raise NotImplementedError

    def setup_distillation(self):
        """Initialize distillation components and wrap model criterion. Override in subclass."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training loop override
    # ------------------------------------------------------------------

    def _do_train(self, world_size=1):
        """Inject teacher setup and neck hooks before training loop."""
        self.setup_teacher()
        self.setup_distillation()
        self._register_neck_hook()
        super()._do_train(world_size)

    # ------------------------------------------------------------------
    # Neck feature hook
    # ------------------------------------------------------------------

    def _register_neck_hook(self):
        """Register forward hook on Detect head to capture its input (P3/P4/P5)."""
        neck_features = self.neck_features

        def hook_fn(module, input, output):
            neck_features.clear()
            if isinstance(input, tuple) and len(input) > 0:
                feats = input[0] if isinstance(input[0], (list, tuple)) else input
                for feat in feats:
                    neck_features.append(feat)

        detect_head = de_parallel(self.model).model[-1]
        detect_head.register_forward_hook(hook_fn)

    # ------------------------------------------------------------------
    # Validation frequency control
    # ------------------------------------------------------------------

    def validate(self, *args, **kwargs):
        """Only run full validation every val_interval epochs."""
        current_epoch = self.epoch + 1
        final_epoch = current_epoch >= self.epochs

        if (current_epoch % self.val_interval == 0) or final_epoch:
            return super().validate(*args, **kwargs)

        # Non-validation epoch: return cached metrics
        return self.metrics, self.fitness
