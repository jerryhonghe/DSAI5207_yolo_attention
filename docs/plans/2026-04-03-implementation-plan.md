# Attention Relation Distillation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 DINOv2→YOLOv8m 注意力关系蒸馏，包含三组实验（E1 Baseline, E2 Ours, E3 d2）和三种评估方式。

**Architecture:** 深度集成 Ultralytics，继承 DetectionTrainer 注入蒸馏 loss。通过 `preds["feats"]` 获取 neck P3/P4/P5 特征，用自定义 criterion 包装原始 v8DetectionLoss 并添加蒸馏 loss。DINOv2 权重离线加载。

**Tech Stack:** PyTorch, Ultralytics (YOLOv8), DINOv2 (facebookresearch), torchvision, matplotlib, pytorch-grad-cam

---

## Task 1: 项目骨架与依赖

**Files:**
- Create: `requirements.txt`
- Create: `configs/baseline.yaml`
- Create: `configs/relation_distill.yaml`
- Create: `configs/feature_distill_d2.yaml`
- Create: `configs/data.yaml`

**Step 1: 创建 requirements.txt**

```
ultralytics>=8.3.0
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytorch-grad-cam>=1.5.0
pyyaml>=6.0
tqdm>=4.65.0
```

**Step 2: 创建 COCO 数据配置 configs/data.yaml**

```yaml
# COCO 数据集配置 - 指向服务器本地路径
path: /root/autodl-tmp/hh/attention_yolo/data
train: images/train2017
val: images/val2017

names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush

nc: 80
```

**Step 3: 创建 configs/baseline.yaml**

```yaml
# E1: YOLOv8m Baseline - 标准训练
experiment: baseline
model: weights/yolov8m.pt
data: configs/data.yaml
epochs: 100
batch: 16
imgsz: 640
optimizer: SGD
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
val_interval: 10
device: 0
project: runs
name: E1_baseline
```

**Step 4: 创建 configs/relation_distill.yaml**

```yaml
# E2: Attention Relation Distillation (我们的方法)
experiment: relation_distill
model: weights/yolov8m.pt
data: configs/data.yaml
epochs: 100
batch: 8
imgsz: 640
optimizer: SGD
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
val_interval: 10
device: 0
project: runs
name: E2_relation_distill

# 蒸馏配置
teacher_weights: weights/dinov2_vitl14_reg4_pretrain.pth
lambda_distill: 1.0
temperature: 0.07
max_relation_tokens: 1600
teacher_layers:
  shallow: [6, 7, 8]
  middle: [12, 13, 14]
  deep: [20, 21, 22]
level_weights:
  shallow: 1.0
  middle: 1.0
  deep: 1.0
```

**Step 5: 创建 configs/feature_distill_d2.yaml**

```yaml
# E3: Feature Distillation (d2 复现)
experiment: feature_distill_d2
model: weights/yolov8m.pt
data: configs/data.yaml
epochs: 100
batch: 8
imgsz: 640
optimizer: SGD
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
val_interval: 10
device: 0
project: runs
name: E3_feature_distill_d2

# 蒸馏配置
teacher_weights: weights/dinov2_vitl14_reg4_pretrain.pth
dinov2_dim: 1024
lambda_distill: 1.0
```

**Step 6: Commit**

```bash
git add requirements.txt configs/
git commit -m "feat: add project configs and requirements"
```

---

## Task 2: DINOv2 Teacher 模块

**Files:**
- Create: `models/__init__.py`
- Create: `models/teacher.py`
- Create: `tests/test_teacher.py`

**Step 1: 创建 models/__init__.py**

```python
```

**Step 2: 编写 teacher 单元测试 tests/test_teacher.py**

```python
import pytest
import torch
import torch.nn.functional as F


class TestDINOv2TeacherUnit:
    """不依赖真实模型权重的单元测试，验证关系矩阵计算逻辑。"""

    def test_cosine_similarity_relation_matrix_shape(self):
        """验证余弦相似度关系矩阵的形状。"""
        B, N, D = 2, 400, 1024  # 模拟 20x20 patch grid
        feat = torch.randn(B, N, D)
        feat_norm = F.normalize(feat, p=2, dim=-1)
        sim = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
        assert sim.shape == (B, N, N)

    def test_relation_matrix_is_probability_distribution(self):
        """验证 softmax 后的关系矩阵每行和为1。"""
        B, N, D = 2, 100, 256
        feat = torch.randn(B, N, D)
        feat_norm = F.normalize(feat, p=2, dim=-1)
        sim = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
        relation = F.softmax(sim / 0.07, dim=-1)

        row_sums = relation.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_spatial_resize_preserves_batch(self):
        """验证 feature 的空间 resize 保持 batch 维度。"""
        B, D, H, W = 2, 1024, 45, 45
        feat_2d = torch.randn(B, D, H, W)
        resized = F.interpolate(feat_2d, size=(20, 20), mode='bilinear', align_corners=False)
        assert resized.shape == (B, D, 20, 20)

    def test_prefix_token_removal(self):
        """验证去掉 CLS + register tokens 后的 token 数量。"""
        num_prefix = 1 + 4  # CLS + 4 registers
        total_tokens = 2025 + num_prefix  # 45*45 patches + prefix
        patch_tokens = total_tokens - num_prefix
        assert patch_tokens == 2025
        assert int(patch_tokens ** 0.5) == 45


class TestDINOv2TeacherIntegration:
    """需要真实模型权重的集成测试，标记为 slow。"""

    @pytest.mark.slow
    def test_teacher_forward_output_shapes(self):
        from models.teacher import DINOv2Teacher

        teacher = DINOv2Teacher(
            weights_path='weights/dinov2_vitl14_reg4_pretrain.pth',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        images = torch.randn(2, 3, 640, 640).to(teacher.device)
        relations = teacher.get_relation_matrices(images)

        assert 'shallow' in relations
        assert 'middle' in relations
        assert 'deep' in relations
        # P3 对应 shallow: 下采样到 40x40 = 1600
        assert relations['shallow'].shape == (2, 1600, 1600)
        # P4 对应 middle: 40x40 = 1600
        assert relations['middle'].shape == (2, 1600, 1600)
        # P5 对应 deep: 20x20 = 400
        assert relations['deep'].shape == (2, 400, 400)
```

**Step 3: 运行测试，确认失败**

```bash
cd /root/autodl-tmp/hh/attention_yolo
pytest tests/test_teacher.py::TestDINOv2TeacherUnit -v
```

Expected: FAIL (models.teacher 不存在)

**Step 4: 实现 models/teacher.py**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# DINOv2 预处理常量
DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINOV2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DINOV2_INPUT_SIZE = 630  # 14 * 45，必须是 patch_size=14 的倍数


def prepare_for_dinov2(yolo_images):
    """
    将 YOLO 格式图像 ([0,1] RGB) 转换为 DINOv2 格式 (ImageNet normalized)。

    Args:
        yolo_images: [B, 3, 640, 640], RGB, [0, 1]
    Returns:
        [B, 3, 630, 630], ImageNet normalized
    """
    mean = DINOV2_MEAN.to(yolo_images.device)
    std = DINOV2_STD.to(yolo_images.device)
    normalized = (yolo_images - mean) / std
    resized = F.interpolate(normalized, size=(DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE),
                            mode='bilinear', align_corners=False)
    return resized


def _build_dinov2_model(weights_path):
    """
    从本地权重文件构建 DINOv2-Large with registers 模型。
    不使用 torch.hub，完全离线。

    Args:
        weights_path: 本地 .pth 文件路径
    Returns:
        DINOv2 模型实例
    """
    try:
        # 方式一：尝试用 torch.hub 的本地缓存机制
        # 如果已缓存 dinov2 repo 代码则直接用
        import dinov2
        model = dinov2.vision_transformer.vit_large(
            patch_size=14,
            num_register_tokens=4,
            block_chunks=0,
        )
    except ImportError:
        # 方式二：直接用 timm 或手动构建
        # DINOv2-Large 配置: embed_dim=1024, depth=24, num_heads=16
        from functools import partial
        model = _build_vit_large_reg4()

    state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model


def _build_vit_large_reg4():
    """
    手动构建 DINOv2 ViT-Large with 4 registers。
    使用 torch.hub.load 的源码作为参考，从本地加载。
    """
    # 使用 dinov2 源码中的 vision_transformer 模块
    # 需要将 dinov2 源码放在项目目录下或安装
    # 这里使用一个兼容的实现
    import sys
    import os

    # 尝试从本地 dinov2 源码加载
    dinov2_src = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'dinov2')
    if os.path.exists(dinov2_src):
        sys.path.insert(0, dinov2_src)

    from dinov2.models.vision_transformer import vit_large
    model = vit_large(
        patch_size=14,
        num_register_tokens=4,
        block_chunks=0,
    )
    return model


class DINOv2Teacher:
    """
    Frozen DINOv2 teacher。
    提取多层 patch token features，在目标空间分辨率上计算余弦相似度关系矩阵。

    使用方式:
        teacher = DINOv2Teacher('weights/dinov2_vitl14_reg4_pretrain.pth')
        relations = teacher.get_relation_matrices(images)
    """

    def __init__(self, weights_path, device='cuda', temperature=0.07,
                 target_sizes=None, teacher_layers=None):
        """
        Args:
            weights_path: DINOv2 权重文件路径
            device: 设备
            temperature: softmax 温度参数
            target_sizes: 各层级目标空间大小，默认 {'shallow': 40, 'middle': 40, 'deep': 20}
            teacher_layers: 各层级使用的 transformer block 索引
        """
        self.device = device
        self.temperature = temperature
        self.target_sizes = target_sizes or {
            'shallow': 40,  # P3 下采样到 40x40
            'middle': 40,   # P4: 40x40
            'deep': 20      # P5: 20x20
        }
        self.teacher_layers = teacher_layers or {
            'shallow': [6, 7, 8],
            'middle': [12, 13, 14],
            'deep': [20, 21, 22]
        }

        self.model = _build_dinov2_model(weights_path)
        self.model.eval()
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False

        self.feature_maps = {}
        self._register_hooks()

    def _register_hooks(self):
        """在指定 transformer block 注册 forward hook 提取中间层输出。"""
        for level_name, layer_indices in self.teacher_layers.items():
            for idx in layer_indices:
                self.model.blocks[idx].register_forward_hook(
                    self._make_hook(f"{level_name}_{idx}")
                )

    def _make_hook(self, name):
        def hook(module, input, output):
            self.feature_maps[name] = output  # [B, N, D]
        return hook

    @torch.no_grad()
    def get_relation_matrices(self, images):
        """
        从输入图像提取三个层级的关系矩阵。

        Args:
            images: [B, 3, 640, 640], RGB, [0, 1] (YOLO 格式)
        Returns:
            dict: {
                'shallow': [B, 1600, 1600],  # 40*40
                'middle':  [B, 1600, 1600],
                'deep':    [B, 400, 400]     # 20*20
            }
        """
        dinov2_images = prepare_for_dinov2(images)

        # 使用 float16 减少显存
        with torch.amp.autocast('cuda'):
            _ = self.model(dinov2_images)

        num_prefix = 1 + 4  # CLS + 4 register tokens
        relations = {}

        for level_name, layer_indices in self.teacher_layers.items():
            # 收集并平均多层 features
            feats = []
            for idx in layer_indices:
                key = f"{level_name}_{idx}"
                feat = self.feature_maps[key][:, num_prefix:, :]  # [B, num_patches, D]
                feats.append(feat.float())  # 转回 float32 用于关系计算
            avg_feat = torch.stack(feats).mean(dim=0)  # [B, num_patches, D]

            # Reshape 到 2D grid: [B, 45, 45, D] → [B, D, 45, 45]
            B, N, D = avg_feat.shape
            H = W = int(N ** 0.5)
            feat_2d = avg_feat.reshape(B, H, W, D).permute(0, 3, 1, 2)

            # Resize 到目标空间分辨率
            tgt = self.target_sizes[level_name]
            feat_resized = F.interpolate(feat_2d, size=(tgt, tgt),
                                         mode='bilinear', align_corners=False)

            # 展平: [B, D, tgt, tgt] → [B, tgt*tgt, D]
            feat_flat = feat_resized.reshape(B, D, tgt * tgt).permute(0, 2, 1)

            # L2 归一化 + 余弦相似度
            feat_norm = F.normalize(feat_flat, p=2, dim=-1)
            sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))

            # Temperature-scaled softmax → 概率分布
            relation = F.softmax(sim_matrix / self.temperature, dim=-1)
            relations[level_name] = relation

        # 清理缓存
        self.feature_maps.clear()

        return relations

    def get_patch_features(self, images):
        """
        提取 DINOv2 最后一层的 patch token features（用于 d2 方法）。

        Args:
            images: [B, 3, 640, 640]
        Returns:
            [B, num_patches, D] 其中 D=1024
        """
        dinov2_images = prepare_for_dinov2(images)

        with torch.amp.autocast('cuda'):
            output = self.model(dinov2_images, is_training=True)

        num_prefix = 1 + 4
        # output["x_norm_patchtokens"] 或直接用最后一层 hook
        # 使用最后一层 block 的输出
        last_key = f"deep_{self.teacher_layers['deep'][-1]}"
        if last_key in self.feature_maps:
            patch_tokens = self.feature_maps[last_key][:, num_prefix:, :]
        else:
            # fallback: 重新 forward 并获取
            _ = self.model(dinov2_images)
            patch_tokens = self.feature_maps[last_key][:, num_prefix:, :]

        self.feature_maps.clear()
        return patch_tokens.float()
```

**Step 5: 运行单元测试**

```bash
pytest tests/test_teacher.py::TestDINOv2TeacherUnit -v
```

Expected: PASS (单元测试不依赖模型文件)

**Step 6: Commit**

```bash
git add models/ tests/test_teacher.py
git commit -m "feat: add DINOv2 teacher module with offline weight loading"
```

---

## Task 3: Student 端关系矩阵构造器

**Files:**
- Create: `models/relation_constructor.py`
- Create: `tests/test_relation_constructor.py`

**Step 1: 编写测试 tests/test_relation_constructor.py**

```python
import pytest
import torch


class TestRelationConstructor:
    def test_output_shape_p5(self):
        """P5: 20x20, 直接计算，输出 [B, 400, 400]。"""
        from models.relation_constructor import EfficientRelationConstructor

        constructor = EfficientRelationConstructor(temperature=0.07, max_tokens=1600)
        feat = torch.randn(2, 512, 20, 20)
        relation = constructor(feat)
        assert relation.shape == (2, 400, 400)

    def test_output_shape_p4(self):
        """P4: 40x40=1600, 等于 max_tokens，直接计算。"""
        from models.relation_constructor import EfficientRelationConstructor

        constructor = EfficientRelationConstructor(temperature=0.07, max_tokens=1600)
        feat = torch.randn(2, 512, 40, 40)
        relation = constructor(feat)
        assert relation.shape == (2, 1600, 1600)

    def test_output_shape_p3_downsampled(self):
        """P3: 80x80=6400 > max_tokens，下采样到 40x40=1600。"""
        from models.relation_constructor import EfficientRelationConstructor

        constructor = EfficientRelationConstructor(temperature=0.07, max_tokens=1600)
        feat = torch.randn(2, 256, 80, 80)
        relation = constructor(feat)
        assert relation.shape == (2, 1600, 1600)

    def test_output_is_probability_distribution(self):
        """验证输出每行和为1。"""
        from models.relation_constructor import EfficientRelationConstructor

        constructor = EfficientRelationConstructor(temperature=0.07, max_tokens=1600)
        feat = torch.randn(2, 256, 20, 20)
        relation = constructor(feat)

        row_sums = relation.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_no_learnable_parameters(self):
        """确认没有可学习参数。"""
        from models.relation_constructor import EfficientRelationConstructor

        constructor = EfficientRelationConstructor()
        params = list(constructor.parameters())
        assert len(params) == 0
```

**Step 2: 运行测试，确认失败**

```bash
pytest tests/test_relation_constructor.py -v
```

**Step 3: 实现 models/relation_constructor.py**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientRelationConstructor(nn.Module):
    """
    在特征图上构造余弦相似度关系矩阵。
    当空间 token 数超过 max_tokens 时自动下采样。
    不引入可学习参数，推理时不需要此模块。
    """

    def __init__(self, temperature=0.07, max_tokens=1600):
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def forward(self, feature_map):
        """
        Args:
            feature_map: [B, C, H, W]
        Returns:
            relation: [B, N, N] softmax 归一化的关系矩阵
                      N = min(H*W, max_tokens)
        """
        B, C, H, W = feature_map.shape
        num_tokens = H * W

        if num_tokens > self.max_tokens:
            target_size = int(self.max_tokens ** 0.5)
            feature_map = F.adaptive_avg_pool2d(feature_map, (target_size, target_size))
            H, W = target_size, target_size

        # 展平: [B, C, H*W] → [B, H*W, C]
        feat_flat = feature_map.reshape(B, C, H * W).permute(0, 2, 1)

        # L2 归一化
        feat_norm = F.normalize(feat_flat, p=2, dim=-1)

        # 余弦相似度: [B, H*W, H*W]
        sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))

        # Temperature-scaled softmax
        relation = F.softmax(sim_matrix / self.temperature, dim=-1)

        return relation
```

**Step 4: 运行测试**

```bash
pytest tests/test_relation_constructor.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add models/relation_constructor.py tests/test_relation_constructor.py
git commit -m "feat: add efficient relation constructor with auto-downsampling"
```

---

## Task 4: Loss 函数

**Files:**
- Create: `losses/__init__.py`
- Create: `losses/relation_loss.py`
- Create: `losses/feature_loss.py`
- Create: `tests/test_losses.py`

**Step 1: 编写测试 tests/test_losses.py**

```python
import pytest
import torch
import torch.nn.functional as F


class TestRelationDistillationLoss:
    def test_identical_distributions_near_zero_loss(self):
        """相同分布的 KL divergence 应接近 0。"""
        from losses.relation_loss import RelationDistillationLoss

        loss_fn = RelationDistillationLoss()
        B, N = 2, 100
        feat = torch.randn(B, N, 256)
        feat_norm = F.normalize(feat, p=2, dim=-1)
        sim = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
        relation = F.softmax(sim / 0.07, dim=-1)

        loss = loss_fn(relation, relation)
        assert loss.item() < 1e-5

    def test_different_distributions_positive_loss(self):
        """不同分布应产生正的 loss。"""
        from losses.relation_loss import RelationDistillationLoss

        loss_fn = RelationDistillationLoss()
        B, N = 2, 100
        student = F.softmax(torch.randn(B, N, N), dim=-1)
        teacher = F.softmax(torch.randn(B, N, N), dim=-1)

        loss = loss_fn(student, teacher)
        assert loss.item() > 0

    def test_loss_gradient_flows_to_student(self):
        """梯度应流向 student 而非 teacher。"""
        from losses.relation_loss import RelationDistillationLoss

        loss_fn = RelationDistillationLoss()
        student = F.softmax(torch.randn(2, 50, 50, requires_grad=True), dim=-1)
        teacher = F.softmax(torch.randn(2, 50, 50), dim=-1)

        loss = loss_fn(student, teacher)
        loss.backward()
        assert student.grad is not None


class TestFeatureDistillationLoss:
    def test_identical_features_near_zero_loss(self):
        """相同特征的 SmoothL1 loss 应为 0。"""
        from losses.feature_loss import FeatureDistillationLoss

        loss_fn = FeatureDistillationLoss()
        feat = torch.randn(2, 1024, 20, 20)
        feat_norm = F.normalize(feat, p=2, dim=1)

        loss = loss_fn([feat_norm], [feat_norm])
        assert loss.item() < 1e-5

    def test_different_features_positive_loss(self):
        """不同特征应产生正的 loss。"""
        from losses.feature_loss import FeatureDistillationLoss

        loss_fn = FeatureDistillationLoss()
        student = [F.normalize(torch.randn(2, 1024, 20, 20), p=2, dim=1)]
        teacher = [F.normalize(torch.randn(2, 1024, 20, 20), p=2, dim=1)]

        loss = loss_fn(student, teacher)
        assert loss.item() > 0
```

**Step 2: 运行测试，确认失败**

```bash
pytest tests/test_losses.py -v
```

**Step 3: 实现 losses/relation_loss.py**

```python
import torch
import torch.nn as nn


class RelationDistillationLoss(nn.Module):
    """
    KL Divergence loss 对齐 teacher 和 student 的关系分布。
    KLDivLoss 期望 input 是 log-probability，target 是 probability。
    """

    def __init__(self):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_relation, teacher_relation):
        """
        Args:
            student_relation: [B, N, N] softmax 后的 student 关系矩阵
            teacher_relation: [B, N, N] softmax 后的 teacher 关系矩阵
        Returns:
            loss: scalar
        """
        return self.kl_loss(
            student_relation.log(),
            teacher_relation.detach()
        )
```

**Step 4: 实现 losses/feature_loss.py**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillationLoss(nn.Module):
    """
    d2 方法的 SmoothL1 Loss。
    对齐 student embedding 和 teacher patch token features。
    """

    def forward(self, student_embeddings, teacher_embeddings):
        """
        Args:
            student_embeddings: list of [B, D, H, W] (L2 normalized)
            teacher_embeddings: list of [B, D, H, W] (L2 normalized, detached)
        Returns:
            loss: scalar
        """
        total_loss = 0
        for s_emb, t_emb in zip(student_embeddings, teacher_embeddings):
            total_loss += F.smooth_l1_loss(s_emb, t_emb.detach())
        return total_loss / len(student_embeddings)
```

**Step 5: 创建 losses/__init__.py**

```python
```

**Step 6: 运行测试**

```bash
pytest tests/test_losses.py -v
```

Expected: PASS

**Step 7: Commit**

```bash
git add losses/ tests/test_losses.py
git commit -m "feat: add relation KL loss and feature SmoothL1 loss"
```

---

## Task 5: d2 投影头

**Files:**
- Create: `models/d2_head.py`
- Create: `tests/test_d2_head.py`

**Step 1: 编写测试 tests/test_d2_head.py**

```python
import pytest
import torch


class TestD2ProjectionHead:
    def test_output_shapes(self):
        """验证投影头输出维度为 dinov2_dim。"""
        from models.d2_head import D2ProjectionHead

        student_channels = [256, 512, 512]
        head = D2ProjectionHead(student_channels, dinov2_dim=1024)

        features = [
            torch.randn(2, 256, 80, 80),  # P3
            torch.randn(2, 512, 40, 40),  # P4
            torch.randn(2, 512, 20, 20),  # P5
        ]
        embeddings = head(features)

        assert len(embeddings) == 3
        assert embeddings[0].shape == (2, 1024, 80, 80)
        assert embeddings[1].shape == (2, 1024, 40, 40)
        assert embeddings[2].shape == (2, 1024, 20, 20)

    def test_output_is_l2_normalized(self):
        """验证输出是 L2 归一化的。"""
        from models.d2_head import D2ProjectionHead

        head = D2ProjectionHead([256], dinov2_dim=1024)
        feat = [torch.randn(2, 256, 10, 10)]
        emb = head(feat)[0]

        norms = emb.norm(p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_has_learnable_parameters(self):
        """确认有可学习的 1x1 conv 参数。"""
        from models.d2_head import D2ProjectionHead

        head = D2ProjectionHead([256, 512], dinov2_dim=1024)
        params = list(head.parameters())
        assert len(params) > 0
```

**Step 2: 实现 models/d2_head.py**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class D2ProjectionHead(nn.Module):
    """
    d2 方法的投影头：1x1 conv 将 YOLO neck 特征投影到 DINOv2 维度。
    推理时可移除。
    """

    def __init__(self, student_channels, dinov2_dim=1024):
        """
        Args:
            student_channels: list, 各层级的通道数 [P3_ch, P4_ch, P5_ch]
            dinov2_dim: DINOv2 embedding 维度
        """
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Conv2d(ch, dinov2_dim, 1, bias=False)
            for ch in student_channels
        ])

    def forward(self, features):
        """
        Args:
            features: list of [B, C_i, H_i, W_i]
        Returns:
            list of [B, dinov2_dim, H_i, W_i] (L2 normalized)
        """
        embeddings = []
        for proj, feat in zip(self.projectors, features):
            emb = proj(feat)
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb)
        return embeddings
```

**Step 3: 运行测试**

```bash
pytest tests/test_d2_head.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add models/d2_head.py tests/test_d2_head.py
git commit -m "feat: add d2 projection head for feature distillation"
```

---

## Task 6: Base Distillation Trainer

**Files:**
- Create: `trainers/__init__.py`
- Create: `trainers/base_distill_trainer.py`

**Step 1: 实现 trainers/base_distill_trainer.py**

```python
import torch
from ultralytics.models.yolo.detect import DetectionTrainer


class BaseDistillTrainer(DetectionTrainer):
    """
    蒸馏训练器基类。
    - 管理 DINOv2 teacher 的加载
    - 在 Detect head 注册 hook 获取 neck features
    - 控制验证频率（每 val_interval epochs）
    - 支持断点恢复（复用 Ultralytics 机制）
    """

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # 从 overrides 提取自定义参数，避免传给 Ultralytics
        self.custom_cfg = {}
        if overrides:
            custom_keys = [
                'experiment', 'teacher_weights', 'lambda_distill', 'temperature',
                'max_relation_tokens', 'teacher_layers', 'level_weights',
                'val_interval', 'dinov2_dim',
            ]
            for key in custom_keys:
                if key in overrides:
                    self.custom_cfg[key] = overrides.pop(key)

        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

        self.val_interval = self.custom_cfg.get('val_interval', 10)
        self.neck_features = []

    def setup_teacher(self):
        """子类实现：初始化 teacher 模型。在 train() 开始时调用。"""
        raise NotImplementedError

    def setup_distillation(self):
        """子类实现：初始化蒸馏相关组件。"""
        raise NotImplementedError

    def _do_train(self, world_size=1):
        """
        重写训练循环，注入：
        1. Teacher 初始化
        2. Neck feature hook
        3. 自定义验证频率
        """
        # 初始化 teacher 和蒸馏组件
        self.setup_teacher()
        self.setup_distillation()

        # 注册 neck feature hook
        self._register_neck_hook()

        # 调用父类训练循环
        super()._do_train(world_size)

    def _register_neck_hook(self):
        """在 Detect head 注册 hook，捕获输入的 neck features (P3/P4/P5)。"""
        def hook_fn(module, input, output):
            self.neck_features.clear()
            if isinstance(input, tuple) and len(input) > 0:
                feats = input[0] if isinstance(input[0], (list, tuple)) else input
                for feat in feats:
                    self.neck_features.append(feat)

        # model.model[-1] 是 Detect head
        from ultralytics.utils.torch_utils import de_parallel
        detect_head = de_parallel(self.model).model[-1]
        detect_head.register_forward_hook(hook_fn)

    def validate(self, *args, **kwargs):
        """每 val_interval epochs 验证一次。"""
        current_epoch = self.epoch + 1
        final_epoch = current_epoch >= self.epochs

        if (current_epoch % self.val_interval == 0) or final_epoch:
            return super().validate(*args, **kwargs)

        # 非验证轮次返回空指标
        return self.metrics, self.fitness
```

**Step 2: 创建 trainers/__init__.py**

```python
```

**Step 3: Commit**

```bash
git add trainers/
git commit -m "feat: add base distillation trainer with hook and val interval"
```

---

## Task 7: Relation Distillation Trainer (E2)

**Files:**
- Create: `trainers/relation_trainer.py`

**Step 1: 实现 trainers/relation_trainer.py**

```python
import torch
from ultralytics.utils.torch_utils import de_parallel

from trainers.base_distill_trainer import BaseDistillTrainer
from models.teacher import DINOv2Teacher
from models.relation_constructor import EfficientRelationConstructor
from losses.relation_loss import RelationDistillationLoss


class RelationDistillTrainer(BaseDistillTrainer):
    """
    E2 实验：Attention Relation Distillation。
    在 YOLO neck 的 P3/P4/P5 构造余弦相似度关系矩阵，
    用 KL divergence 对齐 DINOv2 teacher 的关系矩阵。
    """

    def setup_teacher(self):
        """初始化 frozen DINOv2 teacher。"""
        teacher_weights = self.custom_cfg.get('teacher_weights', 'weights/dinov2_vitl14_reg4_pretrain.pth')
        temperature = self.custom_cfg.get('temperature', 0.07)
        teacher_layers = self.custom_cfg.get('teacher_layers', None)

        self.teacher = DINOv2Teacher(
            weights_path=teacher_weights,
            device=self.device,
            temperature=temperature,
            teacher_layers=teacher_layers,
        )

    def setup_distillation(self):
        """初始化关系构造器和蒸馏 loss。"""
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

        # 包装原始 criterion
        model = de_parallel(self.model)
        original_criterion = model.criterion
        model.criterion = self._make_distill_criterion(original_criterion)

    def _make_distill_criterion(self, original_criterion):
        """创建包装了蒸馏 loss 的 criterion。"""
        trainer = self

        class DistillCriterion:
            def __init__(self, original):
                self.original = original

            def __call__(self, preds, batch):
                # 原始检测 loss
                det_loss, det_loss_items = self.original(preds, batch)

                # Teacher forward
                images = batch['img']
                teacher_relations = trainer.teacher.get_relation_matrices(images)

                # Student: 从 hook 获取 neck features
                neck_feats = trainer.neck_features
                if len(neck_feats) < 3:
                    # fallback: 如果 hook 未触发，跳过蒸馏
                    return det_loss, det_loss_items

                level_names = ['shallow', 'middle', 'deep']
                relation_loss = torch.tensor(0.0, device=images.device)

                for i, level in enumerate(level_names):
                    student_rel = trainer.relation_constructor(neck_feats[i])
                    teacher_rel = teacher_relations[level]
                    level_loss = trainer.relation_loss_fn(student_rel, teacher_rel)
                    relation_loss = relation_loss + trainer.level_weights[level] * level_loss

                # 总 loss
                total_loss = det_loss + trainer.lambda_distill * relation_loss

                # 扩展 loss items 用于日志
                distill_item = (trainer.lambda_distill * relation_loss).detach().unsqueeze(0)
                extended_items = torch.cat([det_loss_items, distill_item])

                return total_loss, extended_items

        return DistillCriterion(original_criterion)

    def get_validator(self):
        """扩展 loss_names 以包含蒸馏 loss。"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "distill_loss"
        return super().get_validator()

    def label_loss_items(self, loss_items=None, prefix="train"):
        """为扩展的 loss items 添加标签。"""
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            # 确保长度匹配
            if len(loss_items) > len(keys):
                loss_items = loss_items[:len(keys)]
            elif len(loss_items) < len(keys):
                keys = keys[:len(loss_items)]
            return dict(zip(keys, loss_items))
        return keys
```

**Step 2: Commit**

```bash
git add trainers/relation_trainer.py
git commit -m "feat: add relation distillation trainer (E2)"
```

---

## Task 8: Feature Distillation Trainer (E3 - d2 复现)

**Files:**
- Create: `trainers/d2_trainer.py`

**Step 1: 实现 trainers/d2_trainer.py**

```python
import torch
import torch.nn.functional as F
from ultralytics.utils.torch_utils import de_parallel

from trainers.base_distill_trainer import BaseDistillTrainer
from models.teacher import DINOv2Teacher
from models.d2_head import D2ProjectionHead
from losses.feature_loss import FeatureDistillationLoss


class D2DistillTrainer(BaseDistillTrainer):
    """
    E3 实验：Feature Distillation (d2 复现)。
    用 1x1 conv 将 YOLO neck 特征投影到 DINOv2 维度，
    用 SmoothL1 对齐 DINOv2 patch tokens。
    """

    def setup_teacher(self):
        """初始化 frozen DINOv2 teacher。"""
        teacher_weights = self.custom_cfg.get('teacher_weights', 'weights/dinov2_vitl14_reg4_pretrain.pth')
        self.teacher = DINOv2Teacher(
            weights_path=teacher_weights,
            device=self.device,
        )

    def setup_distillation(self):
        """初始化投影头和 feature loss。"""
        dinov2_dim = self.custom_cfg.get('dinov2_dim', 1024)
        self.lambda_distill = self.custom_cfg.get('lambda_distill', 1.0)

        # 获取 YOLO neck 各层通道数
        model = de_parallel(self.model)
        detect_head = model.model[-1]
        student_channels = list(detect_head.ch)

        # 创建投影头并加入优化器
        self.d2_head = D2ProjectionHead(student_channels, dinov2_dim=dinov2_dim)
        self.d2_head.to(self.device)

        self.feature_loss_fn = FeatureDistillationLoss()

        # 将投影头参数加入优化器
        for param_group in self.optimizer.param_groups:
            pass  # 会在 build_optimizer 后处理

        # 包装 criterion
        original_criterion = model.criterion
        model.criterion = self._make_distill_criterion(original_criterion)

    def build_optimizer(self, model, name="auto", lr=0.01, momentum=0.9, decay=1e-5, iterations=1e5):
        """重写以在优化器中包含 d2_head 参数。"""
        optimizer = super().build_optimizer(model, name, lr, momentum, decay, iterations)
        # setup_distillation 在 _do_train 中调用，此时 optimizer 已建好
        # 所以在 setup_distillation 中手动添加参数组
        return optimizer

    def setup_distillation(self):
        """初始化投影头和 feature loss。"""
        dinov2_dim = self.custom_cfg.get('dinov2_dim', 1024)
        self.lambda_distill = self.custom_cfg.get('lambda_distill', 1.0)

        model = de_parallel(self.model)
        detect_head = model.model[-1]
        student_channels = list(detect_head.ch)

        self.d2_head = D2ProjectionHead(student_channels, dinov2_dim=dinov2_dim)
        self.d2_head.to(self.device)
        self.d2_head.train()

        self.feature_loss_fn = FeatureDistillationLoss()

        # 将 d2_head 参数加入优化器
        self.optimizer.add_param_group({
            'params': list(self.d2_head.parameters()),
            'lr': self.optimizer.param_groups[0]['lr'],
            'momentum': self.optimizer.param_groups[0].get('momentum', 0.9),
            'weight_decay': self.optimizer.param_groups[0].get('weight_decay', 5e-4),
        })

        # 包装 criterion
        original_criterion = model.criterion
        model.criterion = self._make_distill_criterion(original_criterion)

    def _make_distill_criterion(self, original_criterion):
        """创建包装了 feature 蒸馏 loss 的 criterion。"""
        trainer = self

        class DistillCriterion:
            def __init__(self, original):
                self.original = original

            def __call__(self, preds, batch):
                det_loss, det_loss_items = self.original(preds, batch)

                images = batch['img']

                # Teacher: 提取 patch features
                teacher_patch_tokens = trainer.teacher.get_patch_features(images)

                # Resize teacher features 到各层级空间分辨率
                neck_feats = trainer.neck_features
                if len(neck_feats) < 3:
                    return det_loss, det_loss_items

                B = images.shape[0]
                N, D = teacher_patch_tokens.shape[1], teacher_patch_tokens.shape[2]
                H_t = W_t = int(N ** 0.5)
                teacher_2d = teacher_patch_tokens.reshape(B, H_t, W_t, D).permute(0, 3, 1, 2)

                teacher_resized = []
                for feat in neck_feats:
                    H, W = feat.shape[2], feat.shape[3]
                    t_feat = F.interpolate(teacher_2d, size=(H, W),
                                           mode='bilinear', align_corners=False)
                    t_feat = F.normalize(t_feat, p=2, dim=1)
                    teacher_resized.append(t_feat)

                # Student: 投影
                student_embeddings = trainer.d2_head(neck_feats)

                # Feature loss
                feature_loss = trainer.feature_loss_fn(student_embeddings, teacher_resized)

                total_loss = det_loss + trainer.lambda_distill * feature_loss

                distill_item = (trainer.lambda_distill * feature_loss).detach().unsqueeze(0)
                extended_items = torch.cat([det_loss_items, distill_item])

                return total_loss, extended_items

        return DistillCriterion(original_criterion)

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "distill_loss"
        return super().get_validator()

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            if len(loss_items) > len(keys):
                loss_items = loss_items[:len(keys)]
            elif len(loss_items) < len(keys):
                keys = keys[:len(loss_items)]
            return dict(zip(keys, loss_items))
        return keys
```

**Step 2: Commit**

```bash
git add trainers/d2_trainer.py
git commit -m "feat: add d2 feature distillation trainer (E3)"
```

---

## Task 9: 统一训练入口

**Files:**
- Create: `train.py`

**Step 1: 实现 train.py**

```python
"""
统一训练入口。

用法:
    python train.py --config configs/baseline.yaml
    python train.py --config configs/relation_distill.yaml
    python train.py --config configs/feature_distill_d2.yaml
    python train.py --config configs/relation_distill.yaml --resume
"""
import argparse
import yaml
from pathlib import Path

from ultralytics import YOLO


def load_config(config_path):
    """加载 YAML 配置文件。"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_baseline(cfg):
    """E1: 标准 YOLOv8m 训练。"""
    model = YOLO(cfg['model'])
    model.train(
        data=cfg['data'],
        epochs=cfg['epochs'],
        batch=cfg['batch'],
        imgsz=cfg['imgsz'],
        optimizer=cfg.get('optimizer', 'SGD'),
        lr0=cfg.get('lr0', 0.01),
        lrf=cfg.get('lrf', 0.01),
        momentum=cfg.get('momentum', 0.937),
        weight_decay=cfg.get('weight_decay', 0.0005),
        warmup_epochs=cfg.get('warmup_epochs', 3),
        device=cfg.get('device', 0),
        project=cfg.get('project', 'runs'),
        name=cfg.get('name', 'baseline'),
        val=True,
    )


def train_relation_distill(cfg, resume=False):
    """E2: Attention Relation Distillation。"""
    from trainers.relation_trainer import RelationDistillTrainer

    overrides = {
        'model': cfg['model'],
        'data': cfg['data'],
        'epochs': cfg['epochs'],
        'batch': cfg['batch'],
        'imgsz': cfg['imgsz'],
        'optimizer': cfg.get('optimizer', 'SGD'),
        'lr0': cfg.get('lr0', 0.01),
        'lrf': cfg.get('lrf', 0.01),
        'momentum': cfg.get('momentum', 0.937),
        'weight_decay': cfg.get('weight_decay', 0.0005),
        'warmup_epochs': cfg.get('warmup_epochs', 3),
        'device': cfg.get('device', 0),
        'project': cfg.get('project', 'runs'),
        'name': cfg.get('name', 'relation_distill'),
        'val': True,
        # 自定义参数
        'teacher_weights': cfg.get('teacher_weights'),
        'lambda_distill': cfg.get('lambda_distill', 1.0),
        'temperature': cfg.get('temperature', 0.07),
        'max_relation_tokens': cfg.get('max_relation_tokens', 1600),
        'teacher_layers': cfg.get('teacher_layers'),
        'level_weights': cfg.get('level_weights'),
        'val_interval': cfg.get('val_interval', 10),
    }

    if resume:
        overrides['resume'] = True

    trainer = RelationDistillTrainer(overrides=overrides)
    trainer.train()


def train_d2_distill(cfg, resume=False):
    """E3: Feature Distillation (d2 复现)。"""
    from trainers.d2_trainer import D2DistillTrainer

    overrides = {
        'model': cfg['model'],
        'data': cfg['data'],
        'epochs': cfg['epochs'],
        'batch': cfg['batch'],
        'imgsz': cfg['imgsz'],
        'optimizer': cfg.get('optimizer', 'SGD'),
        'lr0': cfg.get('lr0', 0.01),
        'lrf': cfg.get('lrf', 0.01),
        'momentum': cfg.get('momentum', 0.937),
        'weight_decay': cfg.get('weight_decay', 0.0005),
        'warmup_epochs': cfg.get('warmup_epochs', 3),
        'device': cfg.get('device', 0),
        'project': cfg.get('project', 'runs'),
        'name': cfg.get('name', 'feature_distill_d2'),
        'val': True,
        # 自定义参数
        'teacher_weights': cfg.get('teacher_weights'),
        'lambda_distill': cfg.get('lambda_distill', 1.0),
        'dinov2_dim': cfg.get('dinov2_dim', 1024),
        'val_interval': cfg.get('val_interval', 10),
    }

    if resume:
        overrides['resume'] = True

    trainer = D2DistillTrainer(overrides=overrides)
    trainer.train()


EXPERIMENT_MAP = {
    'baseline': train_baseline,
    'relation_distill': train_relation_distill,
    'feature_distill_d2': train_d2_distill,
}


def main():
    parser = argparse.ArgumentParser(description='YOLO Attention Distillation Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = cfg.get('experiment', 'baseline')

    if experiment not in EXPERIMENT_MAP:
        raise ValueError(f"Unknown experiment: {experiment}. Choose from {list(EXPERIMENT_MAP.keys())}")

    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment}")
    print(f"  Config: {args.config}")
    print(f"  Resume: {args.resume}")
    print(f"{'='*60}\n")

    train_fn = EXPERIMENT_MAP[experiment]

    if experiment == 'baseline':
        train_fn(cfg)
    else:
        train_fn(cfg, resume=args.resume)


if __name__ == '__main__':
    main()
```

**Step 2: Commit**

```bash
git add train.py
git commit -m "feat: add unified training entry point"
```

---

## Task 10: DINOv2 离线加载 — third_party 源码准备

**Files:**
- Create: `scripts/download_weights.py`
- Create: `scripts/setup_dinov2.sh`

**Step 1: 创建权重下载脚本 scripts/download_weights.py**

```python
"""
在有网络的本机运行，下载所有需要的权重文件。

用法:
    python scripts/download_weights.py --output weights/
"""
import argparse
import os
import urllib.request


WEIGHTS = {
    'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt',
    'dinov2_vitl14_reg4_pretrain.pth': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth',
}


def download(url, output_path):
    if os.path.exists(output_path):
        print(f"  Already exists: {output_path}")
        return
    print(f"  Downloading: {url}")
    print(f"  To: {output_path}")
    urllib.request.urlretrieve(url, output_path)
    print(f"  Done!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='weights/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for filename, url in WEIGHTS.items():
        output_path = os.path.join(args.output, filename)
        download(url, output_path)

    print("\nAll weights downloaded. Upload the weights/ directory to your server.")


if __name__ == '__main__':
    main()
```

**Step 2: 创建 DINOv2 源码准备脚本 scripts/setup_dinov2.sh**

```bash
#!/bin/bash
# 在有网络的机器上运行，克隆 DINOv2 源码到 third_party/
# 这样服务器可以离线使用 DINOv2 的模型定义

set -e

THIRD_PARTY_DIR="third_party"
DINOV2_DIR="${THIRD_PARTY_DIR}/dinov2"

mkdir -p "${THIRD_PARTY_DIR}"

if [ -d "${DINOV2_DIR}" ]; then
    echo "DINOv2 source already exists at ${DINOV2_DIR}"
else
    echo "Cloning DINOv2 source code..."
    git clone https://github.com/facebookresearch/dinov2.git "${DINOV2_DIR}"
    echo "Done!"
fi

echo ""
echo "Next steps:"
echo "1. Run: python scripts/download_weights.py"
echo "2. Upload the entire project directory to your server"
```

**Step 3: Commit**

```bash
git add scripts/
git commit -m "feat: add weight download and DINOv2 setup scripts"
```

---

## Task 11: 评估工具 — GradCAM 热力图

**Files:**
- Create: `evaluation/__init__.py`
- Create: `evaluation/gradcam.py`

**Step 1: 实现 evaluation/gradcam.py**

```python
"""
GradCAM 热力图生成，对比三组实验的关注区域。

用法:
    from evaluation.gradcam import generate_gradcam_comparison
    generate_gradcam_comparison(
        image_paths=['img1.jpg', 'img2.jpg'],
        model_paths={
            'Baseline': 'runs/E1_baseline/weights/best.pt',
            'Ours': 'runs/E2_relation_distill/weights/best.pt',
            'd2': 'runs/E3_feature_distill_d2/weights/best.pt',
        },
        output_dir='results/gradcam/'
    )
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ultralytics import YOLO


class YOLOGradCAM:
    """针对 YOLOv8 的 GradCAM 实现。"""

    def __init__(self, model_path, target_layer_idx=-2, device='cuda'):
        """
        Args:
            model_path: YOLO 模型权重路径
            target_layer_idx: 目标层索引，-2 通常是 neck 的最后一层
            device: 设备
        """
        self.device = device
        self.model = YOLO(model_path)
        self.yolo_model = self.model.model.to(device).eval()

        # 获取目标层
        self.target_layer = self.yolo_model.model[target_layer_idx]
        self.gradients = None
        self.activations = None

        # 注册 hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_path, imgsz=640):
        """
        生成 GradCAM 热力图。

        Args:
            image_path: 输入图像路径
            imgsz: 输入尺寸
        Returns:
            heatmap: [H, W] numpy array, 0-1
            image: 原始图像 numpy array
        """
        # 读取和预处理图像
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (imgsz, imgsz))

        # 转为 tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True

        # Forward
        self.yolo_model.train()  # 需要 train 模式以获取 loss
        preds = self.yolo_model.predict(img_tensor)

        # 使用检测置信度作为目标进行反向传播
        if isinstance(preds, dict):
            scores = preds.get('scores', preds.get('cls', None))
        elif isinstance(preds, (list, tuple)):
            scores = preds[0] if len(preds) > 0 else None
        else:
            scores = preds

        if scores is not None:
            target = scores.max()
            self.yolo_model.zero_grad()
            target.backward(retain_graph=True)

        # 计算 GradCAM
        if self.gradients is not None and self.activations is not None:
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # GAP
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().detach().cpu().numpy()

            # 归一化到 0-1
            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            # Resize 到原图大小
            cam = cv2.resize(cam, (imgsz, imgsz))
        else:
            cam = np.zeros((imgsz, imgsz))

        self.yolo_model.eval()
        return cam, img_resized


def generate_gradcam_comparison(image_paths, model_paths, output_dir, imgsz=640):
    """
    生成三组实验的 GradCAM 对比图。

    Args:
        image_paths: list of str, 输入图像路径
        model_paths: dict, {name: weight_path}
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        n_models = len(model_paths)

        fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))

        # 原图
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (imgsz, imgsz))
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original', fontsize=14)
        axes[0].axis('off')

        # 各模型的 GradCAM
        for i, (name, weight_path) in enumerate(model_paths.items()):
            gradcam = YOLOGradCAM(weight_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            heatmap, _ = gradcam.generate(img_path, imgsz=imgsz)

            # 叠加热力图
            heatmap_colored = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = (0.6 * img_rgb + 0.4 * heatmap_colored).astype(np.uint8)

            axes[i + 1].imshow(overlay)
            axes[i + 1].set_title(name, fontsize=14)
            axes[i + 1].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'gradcam_{img_name}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
```

**Step 2: Commit**

```bash
git add evaluation/
git commit -m "feat: add GradCAM heatmap comparison tool"
```

---

## Task 12: 评估工具 — 关系矩阵可视化

**Files:**
- Create: `evaluation/relation_vis.py`

**Step 1: 实现 evaluation/relation_vis.py**

```python
"""
关系矩阵可视化：对比 teacher 和 student 的关系矩阵。

用法:
    from evaluation.relation_vis import visualize_relation_matrices
    visualize_relation_matrices(
        image_path='test.jpg',
        student_model_path='runs/E2_relation_distill/weights/best.pt',
        teacher_weights='weights/dinov2_vitl14_reg4_pretrain.pth',
        output_dir='results/relation_vis/'
    )
"""
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

from models.teacher import DINOv2Teacher
from models.relation_constructor import EfficientRelationConstructor


def _get_neck_features(model_path, image_tensor, device):
    """从 YOLO 模型获取 neck features。"""
    model = YOLO(model_path)
    yolo_model = model.model.to(device).eval()

    neck_features = []

    def hook_fn(module, input, output):
        neck_features.clear()
        if isinstance(input, tuple) and len(input) > 0:
            feats = input[0] if isinstance(input[0], (list, tuple)) else input
            for feat in feats:
                neck_features.append(feat.detach())

    detect_head = yolo_model.model[-1]
    handle = detect_head.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = yolo_model.predict(image_tensor)

    handle.remove()
    return neck_features


def visualize_relation_matrices(image_path, student_model_path, teacher_weights,
                                 output_dir, query_pos=None, temperature=0.07):
    """
    可视化 teacher 和 student 的关系矩阵。

    选取一个 query position，展示它与所有其他位置的关系强度（作为热力图覆盖在原图上）。

    Args:
        image_path: 输入图像路径
        student_model_path: YOLO 模型权重路径
        teacher_weights: DINOv2 权重路径
        output_dir: 输出目录
        query_pos: (row, col) 查询位置，默认选图像中心
        temperature: softmax 温度
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取图像
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_640 = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.from_numpy(img_640).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Teacher 关系矩阵
    teacher = DINOv2Teacher(teacher_weights, device=device, temperature=temperature)
    teacher_relations = teacher.get_relation_matrices(img_tensor)

    # Student 关系矩阵
    neck_features = _get_neck_features(student_model_path, img_tensor, device)
    constructor = EfficientRelationConstructor(temperature=temperature, max_tokens=1600)

    student_relations = {}
    level_names = ['shallow', 'middle', 'deep']
    for i, level in enumerate(level_names):
        if i < len(neck_features):
            student_relations[level] = constructor(neck_features[i])

    # 可视化各层级
    level_sizes = {'shallow': 40, 'middle': 40, 'deep': 20}
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    for level in level_names:
        if level not in student_relations or level not in teacher_relations:
            continue

        size = level_sizes[level]
        if query_pos is None:
            qr, qc = size // 2, size // 2
        else:
            qr, qc = query_pos

        query_idx = qr * size + qc

        t_rel = teacher_relations[level][0, query_idx].cpu().numpy().reshape(size, size)
        s_rel = student_relations[level][0, query_idx].cpu().numpy().reshape(size, size)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 原图 + query 位置标记
        axes[0].imshow(img_640)
        # 将 query 位置映射到 640x640
        scale = 640 / size
        axes[0].plot(qc * scale + scale / 2, qr * scale + scale / 2,
                     'r*', markersize=15)
        axes[0].set_title(f'Query Position ({level})', fontsize=14)
        axes[0].axis('off')

        # Teacher 关系热力图
        t_heatmap = cv2.resize(t_rel, (640, 640))
        t_overlay = img_640.copy().astype(float)
        t_colored = plt.cm.hot(t_heatmap)[:, :, :3] * 255
        t_overlay = (0.5 * t_overlay + 0.5 * t_colored).astype(np.uint8)
        axes[1].imshow(t_overlay)
        axes[1].set_title(f'Teacher ({level})', fontsize=14)
        axes[1].axis('off')

        # Student 关系热力图
        s_heatmap = cv2.resize(s_rel, (640, 640))
        s_overlay = img_640.copy().astype(float)
        s_colored = plt.cm.hot(s_heatmap)[:, :, :3] * 255
        s_overlay = (0.5 * s_overlay + 0.5 * s_colored).astype(np.uint8)
        axes[2].imshow(s_overlay)
        axes[2].set_title(f'Student ({level})', fontsize=14)
        axes[2].axis('off')

        plt.suptitle(f'Relation Matrix Visualization - {level}', fontsize=16)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'relation_{img_name}_{level}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
```

**Step 2: Commit**

```bash
git add evaluation/relation_vis.py
git commit -m "feat: add relation matrix visualization tool"
```

---

## Task 13: 评估工具 — mAP 对比与统一评估入口

**Files:**
- Create: `evaluation/compare.py`
- Create: `evaluate.py`

**Step 1: 实现 evaluation/compare.py**

```python
"""
三组实验的 mAP 指标对比。

用法:
    from evaluation.compare import compare_experiments
    compare_experiments(
        model_paths={
            'E1 Baseline': 'runs/E1_baseline/weights/best.pt',
            'E2 Ours': 'runs/E2_relation_distill/weights/best.pt',
            'E3 d2': 'runs/E3_feature_distill_d2/weights/best.pt',
        },
        data_yaml='configs/data.yaml',
        output_dir='results/'
    )
"""
import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def compare_experiments(model_paths, data_yaml, output_dir, imgsz=640):
    """
    对比多组实验的 mAP 指标。

    Args:
        model_paths: dict, {experiment_name: weight_path}
        data_yaml: 数据集配置路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for name, weight_path in model_paths.items():
        print(f"\nEvaluating: {name}")
        print(f"  Weights: {weight_path}")

        model = YOLO(weight_path)
        metrics = model.val(data=data_yaml, imgsz=imgsz, verbose=False)

        results[name] = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }

        # 尝试提取按尺度的 AP（如果可用）
        if hasattr(metrics.box, 'maps'):
            results[name]['per_class_ap50'] = [float(x) for x in metrics.box.maps]

        print(f"  mAP50: {results[name]['mAP50']:.4f}")
        print(f"  mAP50-95: {results[name]['mAP50-95']:.4f}")

    # 保存结果
    results_path = os.path.join(output_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # 生成对比表格图
    _plot_comparison(results, output_dir)

    return results


def _plot_comparison(results, output_dir):
    """生成 mAP 对比柱状图。"""
    names = list(results.keys())
    metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for i, metric in enumerate(metrics):
        values = [results[name].get(metric, 0) for name in names]
        bars = axes[i].bar(range(len(names)), values, color=['#2196F3', '#4CAF50', '#FF9800'][:len(names)])

        axes[i].set_title(metric, fontsize=14, fontweight='bold')
        axes[i].set_xticks(range(len(names)))
        axes[i].set_xticklabels(names, rotation=30, ha='right', fontsize=10)
        axes[i].set_ylim(0, 1)

        # 在柱子上方标注数值
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Experiment Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'comparison_chart.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to: {save_path}")
```

**Step 2: 实现 evaluate.py**

```python
"""
统一评估入口。

用法:
    # 对比 mAP
    python evaluate.py --mode compare

    # 生成 GradCAM 热力图
    python evaluate.py --mode gradcam --images img1.jpg img2.jpg

    # 生成关系矩阵可视化
    python evaluate.py --mode relation --images img1.jpg

    # 全部评估
    python evaluate.py --mode all --images img1.jpg img2.jpg
"""
import argparse
import os
import glob


DEFAULT_MODELS = {
    'E1 Baseline': 'runs/E1_baseline/weights/best.pt',
    'E2 Ours': 'runs/E2_relation_distill/weights/best.pt',
    'E3 d2': 'runs/E3_feature_distill_d2/weights/best.pt',
}

DEFAULT_TEACHER = 'weights/dinov2_vitl14_reg4_pretrain.pth'
DEFAULT_DATA = 'configs/data.yaml'
DEFAULT_OUTPUT = 'results'


def run_compare(model_paths, data_yaml, output_dir):
    from evaluation.compare import compare_experiments
    compare_experiments(model_paths, data_yaml, os.path.join(output_dir, 'compare'))


def run_gradcam(image_paths, model_paths, output_dir):
    from evaluation.gradcam import generate_gradcam_comparison
    generate_gradcam_comparison(image_paths, model_paths, os.path.join(output_dir, 'gradcam'))


def run_relation_vis(image_paths, student_model_path, teacher_weights, output_dir):
    from evaluation.relation_vis import visualize_relation_matrices
    for img_path in image_paths:
        visualize_relation_matrices(
            img_path, student_model_path, teacher_weights,
            os.path.join(output_dir, 'relation_vis')
        )


def main():
    parser = argparse.ArgumentParser(description='Evaluation Tools')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['compare', 'gradcam', 'relation', 'all'])
    parser.add_argument('--images', nargs='+', help='Image paths for visualization')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA)
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--teacher', type=str, default=DEFAULT_TEACHER)
    args = parser.parse_args()

    # 检查实际存在的模型
    model_paths = {}
    for name, path in DEFAULT_MODELS.items():
        if os.path.exists(path):
            model_paths[name] = path
        else:
            print(f"Warning: {name} model not found at {path}, skipping")

    if not model_paths:
        print("Error: No model weights found. Train models first.")
        return

    if args.mode in ('compare', 'all'):
        print("\n=== mAP Comparison ===")
        run_compare(model_paths, args.data, args.output)

    if args.mode in ('gradcam', 'all'):
        if not args.images:
            print("Error: --images required for GradCAM mode")
            return
        print("\n=== GradCAM Heatmaps ===")
        run_gradcam(args.images, model_paths, args.output)

    if args.mode in ('relation', 'all'):
        if not args.images:
            print("Error: --images required for relation visualization mode")
            return
        ours_path = model_paths.get('E2 Ours')
        if not ours_path:
            print("Error: E2 Ours model required for relation visualization")
            return
        print("\n=== Relation Matrix Visualization ===")
        run_relation_vis(args.images, ours_path, args.teacher, args.output)

    print(f"\nAll results saved to: {args.output}/")


if __name__ == '__main__':
    main()
```

**Step 3: Commit**

```bash
git add evaluation/compare.py evaluate.py
git commit -m "feat: add mAP comparison and unified evaluation entry"
```

---

## Task 14: 创建测试 __init__ 文件和 conftest

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: 创建 tests/conftest.py**

```python
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (needs real model weights)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
```

**Step 2: 创建空 tests/__init__.py**

```python
```

**Step 3: 运行所有单元测试确认通过**

```bash
pytest tests/ -v --ignore=tests/test_teacher.py -k "not slow"
```

Expected: ALL PASS

**Step 4: Commit**

```bash
git add tests/
git commit -m "feat: add test configuration and conftest"
```

---

## Task 15: 端到端冒烟测试

**Files:**
- Create: `tests/test_smoke.py`

**Step 1: 编写冒烟测试**

```python
"""
端到端冒烟测试，验证各组件可以正确组合。
不需要真实模型权重，使用随机初始化。
"""
import pytest
import torch
import torch.nn.functional as F

from models.relation_constructor import EfficientRelationConstructor
from losses.relation_loss import RelationDistillationLoss
from losses.feature_loss import FeatureDistillationLoss
from models.d2_head import D2ProjectionHead


class TestEndToEnd:
    def test_relation_distill_pipeline(self):
        """
        模拟 E2 的完整蒸馏管线：
        teacher 关系矩阵 + student 关系矩阵 → KL loss → backward
        """
        B = 2
        temperature = 0.07
        constructor = EfficientRelationConstructor(temperature=temperature, max_tokens=1600)
        loss_fn = RelationDistillationLoss()

        # 模拟 neck features
        p3 = torch.randn(B, 256, 80, 80, requires_grad=True)
        p4 = torch.randn(B, 512, 40, 40, requires_grad=True)
        p5 = torch.randn(B, 512, 20, 20, requires_grad=True)

        # Student 关系矩阵
        s_shallow = constructor(p3)  # [B, 1600, 1600]
        s_middle = constructor(p4)   # [B, 1600, 1600]
        s_deep = constructor(p5)     # [B, 400, 400]

        # 模拟 teacher 关系矩阵（随机概率分布）
        t_shallow = F.softmax(torch.randn(B, 1600, 1600) / temperature, dim=-1)
        t_middle = F.softmax(torch.randn(B, 1600, 1600) / temperature, dim=-1)
        t_deep = F.softmax(torch.randn(B, 400, 400) / temperature, dim=-1)

        # 计算 loss
        loss = (loss_fn(s_shallow, t_shallow) +
                loss_fn(s_middle, t_middle) +
                loss_fn(s_deep, t_deep))

        assert loss.item() > 0

        # 验证梯度可以流回 neck features
        loss.backward()
        assert p3.grad is not None
        assert p4.grad is not None
        assert p5.grad is not None

    def test_d2_distill_pipeline(self):
        """
        模拟 E3 的完整蒸馏管线：
        teacher patch tokens → resize → student projection → SmoothL1 → backward
        """
        B = 2
        dinov2_dim = 1024
        student_channels = [256, 512, 512]

        d2_head = D2ProjectionHead(student_channels, dinov2_dim=dinov2_dim)
        loss_fn = FeatureDistillationLoss()

        # 模拟 neck features
        features = [
            torch.randn(B, 256, 80, 80),
            torch.randn(B, 512, 40, 40),
            torch.randn(B, 512, 20, 20),
        ]

        # Student embeddings
        student_embs = d2_head(features)

        # 模拟 teacher patch tokens → resize
        teacher_tokens = torch.randn(B, 2025, dinov2_dim)  # 45*45 patches
        H_t = W_t = 45
        teacher_2d = teacher_tokens.reshape(B, H_t, W_t, dinov2_dim).permute(0, 3, 1, 2)

        teacher_resized = []
        for feat in features:
            H, W = feat.shape[2], feat.shape[3]
            t = F.interpolate(teacher_2d, size=(H, W), mode='bilinear', align_corners=False)
            t = F.normalize(t, p=2, dim=1)
            teacher_resized.append(t)

        # 计算 loss
        loss = loss_fn(student_embs, teacher_resized)
        assert loss.item() > 0

        # 验证梯度可以流回投影头
        loss.backward()
        for proj in d2_head.projectors:
            assert proj.weight.grad is not None
```

**Step 2: 运行冒烟测试**

```bash
pytest tests/test_smoke.py -v
```

Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: add end-to-end smoke tests for distillation pipelines"
```

---

## 执行顺序总结

| Task | 内容 | 依赖 |
|------|------|------|
| 1 | 项目骨架与配置 | 无 |
| 2 | DINOv2 Teacher 模块 | Task 1 |
| 3 | Student 关系矩阵构造器 | Task 1 |
| 4 | Loss 函数 | Task 1 |
| 5 | d2 投影头 | Task 1 |
| 6 | Base Distillation Trainer | Tasks 2,3,4 |
| 7 | Relation Trainer (E2) | Tasks 2,3,4,6 |
| 8 | d2 Trainer (E3) | Tasks 2,4,5,6 |
| 9 | 统一训练入口 | Tasks 7,8 |
| 10 | 离线加载脚本 | Task 2 |
| 11 | GradCAM 热力图 | Task 1 |
| 12 | 关系矩阵可视化 | Tasks 2,3 |
| 13 | mAP 对比与评估入口 | Task 11,12 |
| 14 | 测试配置 | Task 1 |
| 15 | 端到端冒烟测试 | Tasks 2-5 |

Tasks 2/3/4/5 可并行执行。Tasks 11/12 可并行执行。
