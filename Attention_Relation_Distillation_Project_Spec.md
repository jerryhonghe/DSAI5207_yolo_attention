# Attention Relation Distillation from DINOv2 to YOLO

## 项目技术规格文档 - Claude Code 实现指南

---

## 1. 项目概述

### 1.1 核心思想

YOLO 推理快但缺乏全局上下文建模能力，DINOv2 通过 self-attention 天然具备这种能力但推理成本高。本项目在训练阶段用 frozen DINOv2 的 self-attention 矩阵（描述 patch 间关注关系）作为 teacher signal，在 YOLO neck 的 P3/P4/P5 特征图上构造**余弦相似度关系矩阵**，用 KL divergence loss 让 YOLO 学到类似的全局关系 pattern。推理时 teacher 完全移除，YOLO 结构和速度不变。

### 1.2 对比方法

| 方法 | 角色 | 说明 |
|------|------|------|
| YOLOv8m vanilla | Baseline | 标准训练，无蒸馏 |
| YOLOv8m + Feature Distillation (d2复现) | 对比组 | 论文方法，分类分支扩展1024通道，SmoothL1对齐DINOv2 patch tokens |
| YOLOv8m + Attention Relation Distillation | **我们的方法** | Neck层P3/P4/P5余弦相似度关系矩阵，KL divergence对齐DINOv2注意力 |

### 1.3 评估指标

- mAP50, mAP50:95（整体）
- AP_small, AP_medium, AP_large（按目标尺度）
- FPS 和参数量（确认推理时无开销）

---

## 2. 整体架构

### 2.1 训练时架构（我们的方法）

```
输入图像 (640x640)
    │
    ├──────────────────────────────┐
    ▼                              ▼
  YOLOv8m                    DINOv2-Large (frozen)
    │                              │
    ▼                              ▼
  Backbone (CSPDarknet)      ViT Encoder (24 layers)
    │                              │
    ▼                              ▼
  Neck (PANet/FPN)           提取多层 self-attention maps
    │                              │
    ├── P3 (80x80, 浅层特征)       ├── 浅层注意力 (layers 6-8 平均)
    ├── P4 (40x40, 中层特征)       ├── 中层注意力 (layers 12-14 平均)
    ├── P5 (20x20, 深层特征)       ├── 深层注意力 (layers 20-22 平均)
    │                              │
    ▼                              ▼
  构造余弦相似度               提取注意力关系矩阵
  关系矩阵 (S_student)        (S_teacher)
    │                              │
    └──────────┬───────────────────┘
               ▼
        KL Divergence Loss (L_relation)
               +
        原始检测 Loss (L_det)
               =
        总 Loss = L_det + λ * L_relation
```

### 2.2 推理时架构

```
输入图像 (640x640)
    │
    ▼
  YOLOv8m（原始结构，无任何修改）
    │
    ▼
  检测输出
```

**关键：推理时完全不需要 DINOv2，YOLO 的参数量、结构、FPS 与 vanilla 完全一致。**

---

## 3. 核心模块详细设计

### 3.1 Teacher 端：DINOv2 注意力关系矩阵提取

#### 3.1.1 模型选择

- 使用 `dinov2_vitl14_reg`（Large版本 + registers）
- 输入分辨率：与 YOLO 一致 (640x640)
- DINOv2 patch size = 14，所以 patch grid = 640/14 ≈ 45x45（实际46x46=2116 patches）
- 全程 frozen，不计算梯度

#### 3.1.2 注意力矩阵提取

```python
import torch
import torch.nn.functional as F

class DINOv2Teacher:
    """
    Frozen DINOv2 teacher，提取多层注意力关系矩阵。
    """
    def __init__(self, model_name='dinov2_vitl14_reg', device='cuda'):
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 存储注意力图的 hooks
        self.attention_maps = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """
        在指定层注册 forward hook 提取注意力权重。
        DINOv2-Large 有 24 层 transformer blocks。
        
        层级对应关系：
        - 浅层 (layers 6,7,8)  → 对应 YOLO P3 (局部纹理特征)
        - 中层 (layers 12,13,14) → 对应 YOLO P4 (中等语义特征)
        - 深层 (layers 20,21,22) → 对应 YOLO P5 (全局语义特征)
        """
        self.target_layers = {
            'shallow': [6, 7, 8],
            'middle': [12, 13, 14],
            'deep': [20, 21, 22]
        }
        
        for name, layer_indices in self.target_layers.items():
            for idx in layer_indices:
                block = self.model.blocks[idx]
                block.attn.register_forward_hook(
                    self._make_hook(f"{name}_{idx}")
                )
    
    def _make_hook(self, name):
        def hook(module, input, output):
            # DINOv2 的 attention 模块内部可以通过
            # 修改 forward 或使用 attn_weights 获取注意力权重
            # 具体实现取决于 DINOv2 源码结构
            # 需要获取 softmax 后的 attention weights: [B, num_heads, N, N]
            self.attention_maps[name] = module.attn_weights
        return hook
    
    @torch.no_grad()
    def extract_attention_relations(self, images):
        """
        提取三个层级的注意力关系矩阵。
        
        Args:
            images: [B, 3, 640, 640]
        
        Returns:
            dict: {
                'shallow': [B, H_p3*W_p3, H_p3*W_p3],  # resize到P3空间分辨率
                'middle':  [B, H_p4*W_p4, H_p4*W_p4],
                'deep':    [B, H_p5*W_p5, H_p5*W_p5]
            }
        """
        _ = self.model(images)
        
        relations = {}
        target_sizes = {
            'shallow': 80,  # P3: 80x80
            'middle': 40,   # P4: 40x40
            'deep': 20      # P5: 20x20
        }
        
        for level_name, layer_indices in self.target_layers.items():
            # 收集该层级所有层的注意力，取平均
            attn_list = []
            for idx in layer_indices:
                key = f"{level_name}_{idx}"
                attn = self.attention_maps[key]  # [B, num_heads, N, N]
                attn_list.append(attn)
            
            # 平均多层注意力
            avg_attn = torch.stack(attn_list).mean(dim=0)  # [B, num_heads, N, N]
            
            # 平均多头注意力 → [B, N, N]
            avg_attn = avg_attn.mean(dim=1)
            
            # 去掉 CLS token 和 register tokens
            # DINOv2 with registers: 前 1+num_register 个 token 不是 patch token
            # 假设 num_registers = 4, 则 patch tokens 从 index 5 开始
            num_prefix = 1 + 4  # CLS + 4 registers
            patch_attn = avg_attn[:, num_prefix:, num_prefix:]  # [B, num_patches, num_patches]
            
            # Reshape 到 2D grid
            H_dino = W_dino = int(patch_attn.shape[1] ** 0.5)
            patch_attn_2d = patch_attn.reshape(
                patch_attn.shape[0], H_dino, W_dino, H_dino, W_dino
            )
            
            # Resize 到目标空间分辨率
            target_size = target_sizes[level_name]
            # 需要将 [B, H, W, H, W] resize 到 [B, target, target, target, target]
            # 分步 resize
            relation_matrix = self._resize_relation_matrix(
                patch_attn, H_dino, target_size
            )
            
            # Softmax 归一化（沿最后一维），使其成为概率分布
            relation_matrix = F.softmax(relation_matrix / self.temperature, dim=-1)
            
            relations[level_name] = relation_matrix
        
        return relations
    
    def _resize_relation_matrix(self, attn, src_size, tgt_size):
        """
        将 attention 关系矩阵从 src_size² x src_size² 
        resize 到 tgt_size² x tgt_size²。
        
        简化方案：对 patch tokens 做空间池化后重新计算关系。
        
        更高效的替代方案（推荐）：
        直接从 DINOv2 提取 patch token features，
        空间 resize 后在 student 端计算余弦相似度。
        这样避免了对 NxN 矩阵的 resize。
        详见 3.1.3 替代实现。
        """
        # 方案：先 resize patch features 再算关系
        # 见 3.1.3
        pass
```

#### 3.1.3 推荐实现：提取 Patch Features 后在目标分辨率计算关系

**直接 resize 注意力矩阵在实现上很复杂，推荐以下替代方案：**

```python
class DINOv2TeacherV2:
    """
    推荐实现：提取 DINOv2 多层 patch token features，
    空间 resize 到 P3/P4/P5 分辨率后，
    计算余弦相似度关系矩阵作为 teacher signal。
    
    优点：
    1. 避免对 NxN attention 矩阵做 resize
    2. Teacher 和 Student 使用相同的关系构造方式（余弦相似度）
    3. 实现更简洁
    """
    
    def __init__(self, model_name='dinov2_vitl14_reg', device='cuda', temperature=0.07):
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.model.to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.temperature = temperature
        self.feature_maps = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """注册 hook 提取中间层输出（block 的 output）"""
        self.target_layers = {
            'shallow': [6, 7, 8],
            'middle': [12, 13, 14],
            'deep': [20, 21, 22]
        }
        
        for name, layer_indices in self.target_layers.items():
            for idx in layer_indices:
                self.model.blocks[idx].register_forward_hook(
                    self._make_feature_hook(f"{name}_{idx}")
                )
    
    def _make_feature_hook(self, name):
        def hook(module, input, output):
            self.feature_maps[name] = output  # [B, N, D]
        return hook
    
    @torch.no_grad()
    def get_relation_matrices(self, images):
        """
        Args:
            images: [B, 3, 640, 640]
        Returns:
            dict: {
                'shallow': [B, 80*80, 80*80] softmax归一化的关系矩阵,
                'middle':  [B, 40*40, 40*40],
                'deep':    [B, 20*20, 20*20]
            }
        """
        _ = self.model(images)
        
        target_sizes = {'shallow': 80, 'middle': 40, 'deep': 20}
        num_prefix = 1 + 4  # CLS + registers
        
        relations = {}
        for level_name, layer_indices in self.target_layers.items():
            # 收集并平均多层 features
            feats = []
            for idx in layer_indices:
                key = f"{level_name}_{idx}"
                feat = self.feature_maps[key][:, num_prefix:, :]  # [B, num_patches, D]
                feats.append(feat)
            avg_feat = torch.stack(feats).mean(dim=0)  # [B, num_patches, D]
            
            # Reshape 到 2D grid
            B, N, D = avg_feat.shape
            H = W = int(N ** 0.5)
            feat_2d = avg_feat.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
            
            # Resize 到目标空间分辨率
            tgt = target_sizes[level_name]
            feat_resized = F.interpolate(feat_2d, size=(tgt, tgt), mode='bilinear', align_corners=False)
            # [B, D, tgt, tgt]
            
            # 展平空间维度
            feat_flat = feat_resized.reshape(B, D, tgt * tgt).permute(0, 2, 1)
            # [B, tgt*tgt, D]
            
            # L2 归一化
            feat_norm = F.normalize(feat_flat, p=2, dim=-1)
            
            # 余弦相似度关系矩阵
            sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
            # [B, tgt*tgt, tgt*tgt]
            
            # Temperature-scaled softmax → 概率分布
            relation = F.softmax(sim_matrix / self.temperature, dim=-1)
            
            relations[level_name] = relation
        
        return relations
```

### 3.2 Student 端：YOLO Neck 特征关系矩阵构造

```python
class RelationConstructor(torch.nn.Module):
    """
    在 YOLO 的 P3/P4/P5 特征图上构造余弦相似度关系矩阵。
    
    注意：此模块不引入任何可学习参数，
    只是对现有特征做关系计算，
    因此推理时完全不需要这个模块。
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, feature_map):
        """
        Args:
            feature_map: [B, C, H, W] - YOLO neck 输出的特征图
                P3: [B, 256, 80, 80]  (YOLOv8m neck 输出通道数为256)
                P4: [B, 512, 40, 40]
                P5: [B, 512, 20, 20]  (具体通道数取决于YOLOv8m配置)
        
        Returns:
            relation_matrix: [B, H*W, H*W] - softmax归一化的关系矩阵
        """
        B, C, H, W = feature_map.shape
        
        # 展平空间维度: [B, C, H*W] → [B, H*W, C]
        feat_flat = feature_map.reshape(B, C, H * W).permute(0, 2, 1)
        
        # L2 归一化
        feat_norm = F.normalize(feat_flat, p=2, dim=-1)
        
        # 余弦相似度矩阵: [B, H*W, H*W]
        sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
        
        # Temperature-scaled softmax
        relation = F.softmax(sim_matrix / self.temperature, dim=-1)
        
        return relation
```

### 3.3 显存优化：关系矩阵的稀疏/分块计算

**问题：P3 的 80x80=6400 个位置，关系矩阵大小为 6400x6400，显存开销极大。**

```python
class EfficientRelationConstructor(torch.nn.Module):
    """
    显存优化版本。
    
    策略一：空间下采样（推荐用于 P3）
    策略二：局部窗口 + 全局采样混合关系
    """
    
    def __init__(self, temperature=0.07, max_tokens=1600):
        """
        Args:
            temperature: softmax 温度
            max_tokens: 关系矩阵最大 token 数。
                        1600 → 矩阵大小 1600x1600，约 10MB/sample (float32)
                        建议: P5(400) 直接算, P4(1600) 直接算, P3(6400) 需要下采样
        """
        super().__init__()
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def forward(self, feature_map):
        """
        Args:
            feature_map: [B, C, H, W]
        Returns:
            relation: [B, N, N] where N = min(H*W, max_tokens)
            若做了下采样，N < H*W
        """
        B, C, H, W = feature_map.shape
        num_tokens = H * W
        
        if num_tokens > self.max_tokens:
            # 空间下采样：用 adaptive_avg_pool2d
            target_size = int(self.max_tokens ** 0.5)  # e.g., 40x40=1600
            feature_map = F.adaptive_avg_pool2d(feature_map, (target_size, target_size))
            H, W = target_size, target_size
        
        # 同 RelationConstructor 的逻辑
        feat_flat = feature_map.reshape(B, C, H * W).permute(0, 2, 1)
        feat_norm = F.normalize(feat_flat, p=2, dim=-1)
        sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
        relation = F.softmax(sim_matrix / self.temperature, dim=-1)
        
        return relation
```

**显存预算分析：**

| 层级 | 原始空间 | Token数 | 关系矩阵大小 | 显存 (float32, B=16) |
|------|---------|---------|-------------|---------------------|
| P5 | 20x20 | 400 | 400x400 | ~10 MB |
| P4 | 40x40 | 1600 | 1600x1600 | ~164 MB |
| P3 | 80x80 | 6400 | 6400x6400 | ~2.6 GB |
| P3 下采样到 40x40 | 40x40 | 1600 | 1600x1600 | ~164 MB |

**建议：P3 下采样到 40x40 或 32x32 后再计算关系矩阵。Teacher 端也做相同的下采样以保持对齐。**

---

## 4. Loss 设计

### 4.1 Relation Distillation Loss

```python
class RelationDistillationLoss(torch.nn.Module):
    """
    KL Divergence loss 对齐 teacher 和 student 的关系分布。
    
    为什么用 KL 而不是 SmoothL1：
    - 关系矩阵经过 softmax 后是概率分布
    - KL divergence 是衡量两个概率分布差异的标准度量
    - SmoothL1 适合逐元素的特征回归，不适合分布对齐
    """
    
    def __init__(self):
        super().__init__()
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_relation, teacher_relation):
        """
        Args:
            student_relation: [B, N, N] - softmax 后的 student 关系矩阵
            teacher_relation: [B, N, N] - softmax 后的 teacher 关系矩阵
        
        Returns:
            loss: scalar
        
        注意：KLDivLoss 期望 input 是 log-probability，target 是 probability
        """
        loss = self.kl_loss(
            student_relation.log(),      # log(Q) - student
            teacher_relation.detach()    # P - teacher (stop gradient)
        )
        return loss
```

### 4.2 总 Loss

```python
class TotalLoss:
    """
    L_total = L_det + λ * L_relation
    
    其中：
    - L_det: YOLOv8 原始检测 loss (box_loss + cls_loss + dfl_loss)
    - L_relation: 三个层级的 KL divergence 之和
    - λ: 蒸馏 loss 权重，需要调参
    """
    
    def __init__(self, lambda_distill=1.0, level_weights=None):
        """
        Args:
            lambda_distill: 蒸馏 loss 的总权重
            level_weights: 各层级权重, 默认 {'shallow': 1.0, 'middle': 1.0, 'deep': 1.0}
        """
        self.lambda_distill = lambda_distill
        self.level_weights = level_weights or {
            'shallow': 1.0,  # P3
            'middle': 1.0,   # P4
            'deep': 1.0      # P5
        }
        self.relation_loss_fn = RelationDistillationLoss()
    
    def compute(self, det_loss, student_relations, teacher_relations):
        """
        Args:
            det_loss: scalar, YOLOv8 原始检测 loss
            student_relations: dict, {'shallow': [B,N,N], 'middle': ..., 'deep': ...}
            teacher_relations: dict, 同上
        Returns:
            total_loss, relation_loss (用于日志)
        """
        relation_loss = 0
        for level in ['shallow', 'middle', 'deep']:
            level_loss = self.relation_loss_fn(
                student_relations[level],
                teacher_relations[level]
            )
            relation_loss += self.level_weights[level] * level_loss
        
        total_loss = det_loss + self.lambda_distill * relation_loss
        return total_loss, relation_loss
```

---

## 5. 对比方法实现：Feature Distillation (d2 复现)

### 5.1 架构修改

```python
class YOLOv8_D2_Head(torch.nn.Module):
    """
    复现论文的 d2 方法：
    在分类分支的最后一层卷积扩展 1024 个通道，
    输出 embedding 与 DINOv2 patch tokens 做 SmoothL1 对齐。
    
    修改位置：YOLOv8 的 Detect head 中的 cls_conv
    """
    
    def __init__(self, original_detect_head, dinov2_dim=1024):
        super().__init__()
        self.original_head = original_detect_head
        self.dinov2_dim = dinov2_dim
        
        # 对每个检测层级，扩展分类分支最后的卷积
        self.distill_convs = torch.nn.ModuleList()
        for i, ch in enumerate(original_detect_head.ch):
            # 添加 1x1 conv: ch → dinov2_dim
            self.distill_convs.append(
                torch.nn.Conv2d(ch, dinov2_dim, 1, bias=False)
            )
    
    def forward(self, features):
        """
        Args:
            features: list of [B, C_i, H_i, W_i] from neck
        Returns:
            det_output: 原始检测输出
            embeddings: list of [B, 1024, H_i, W_i] 用于蒸馏
        """
        det_output = self.original_head(features)
        
        embeddings = []
        for i, feat in enumerate(features):
            emb = self.distill_convs[i](feat)  # [B, 1024, H_i, W_i]
            # L2 归一化
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb)
        
        return det_output, embeddings


class FeatureDistillationLoss(torch.nn.Module):
    """
    d2 方法的 SmoothL1 Loss。
    对齐 YOLO embedding 和 DINOv2 patch tokens。
    """
    
    def forward(self, student_embeddings, teacher_patch_tokens):
        """
        Args:
            student_embeddings: list of [B, D, H_i, W_i]
            teacher_patch_tokens: [B, num_patches, D] from DINOv2
        Returns:
            loss: scalar
        """
        total_loss = 0
        for emb in student_embeddings:
            B, D, H, W = emb.shape
            
            # Resize teacher tokens 到相同空间分辨率
            # teacher tokens: [B, N, D] → reshape → resize
            N = teacher_patch_tokens.shape[1]
            H_t = W_t = int(N ** 0.5)
            teacher_2d = teacher_patch_tokens.reshape(B, H_t, W_t, D).permute(0, 3, 1, 2)
            teacher_resized = F.interpolate(teacher_2d, size=(H, W), mode='bilinear', align_corners=False)
            teacher_resized = F.normalize(teacher_resized, p=2, dim=1)
            
            total_loss += F.smooth_l1_loss(emb, teacher_resized.detach())
        
        return total_loss / len(student_embeddings)
```

---

## 6. 训练流程

### 6.1 Ultralytics 集成方案

需要修改 Ultralytics 的训练流程，主要改动点：

```python
# 文件结构建议：
# project/
# ├── models/
# │   ├── relation_distiller.py      # DINOv2Teacher + RelationConstructor
# │   └── feature_distiller.py       # d2 复现
# ├── losses/
# │   ├── relation_loss.py           # KL divergence loss
# │   └── feature_loss.py            # SmoothL1 loss
# ├── trainers/
# │   └── distill_trainer.py         # 自定义 trainer，继承 ultralytics trainer
# ├── configs/
# │   ├── baseline.yaml              # vanilla YOLOv8m
# │   ├── relation_distill.yaml      # 我们的方法
# │   └── feature_distill_d2.yaml    # d2 复现
# └── train.py                       # 入口
```

### 6.2 自定义 Trainer

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

class RelationDistillTrainer(DetectionTrainer):
    """
    继承 Ultralytics 的 DetectionTrainer，
    在训练 loop 中加入 DINOv2 teacher 的 forward 和蒸馏 loss。
    """
    
    def __init__(self, cfg, teacher_model, lambda_distill=1.0, temperature=0.07):
        super().__init__(cfg)
        self.teacher = teacher_model  # frozen DINOv2
        self.lambda_distill = lambda_distill
        self.relation_constructor = EfficientRelationConstructor(temperature=temperature)
        self.relation_loss_fn = RelationDistillationLoss()
    
    def compute_loss(self, batch, preds):
        """
        重写 loss 计算，加入蒸馏 loss。
        
        关键：需要在此处同时拿到：
        1. YOLO 的 neck 输出 (P3, P4, P5)
        2. YOLO 的检测输出 (用于原始 det loss)
        3. DINOv2 的特征 (用于 teacher 关系矩阵)
        """
        # 原始检测 loss
        det_loss = super().compute_loss(batch, preds)
        
        # 获取 images
        images = batch['img']
        
        # Teacher forward
        teacher_relations = self.teacher.get_relation_matrices(images)
        
        # Student: 从 YOLO neck 提取 P3/P4/P5
        # 需要在 model forward 时保存 neck 输出
        neck_features = self.model.neck_features  # 需要 hook 或修改 forward
        
        student_relations = {}
        level_map = {'shallow': 0, 'middle': 1, 'deep': 2}  # P3, P4, P5
        for level_name, idx in level_map.items():
            student_relations[level_name] = self.relation_constructor(neck_features[idx])
        
        # 蒸馏 loss
        relation_loss = 0
        for level in ['shallow', 'middle', 'deep']:
            relation_loss += self.relation_loss_fn(
                student_relations[level],
                teacher_relations[level]
            )
        
        total_loss = det_loss + self.lambda_distill * relation_loss
        return total_loss
```

### 6.3 提取 YOLO Neck Features 的 Hook

```python
def register_neck_hooks(model):
    """
    在 YOLOv8 的 neck 输出层注册 hook，提取 P3/P4/P5 特征。
    
    YOLOv8m 的网络结构中，neck 的输出对应 model.model 的特定层。
    具体层号需要查看 model.model.yaml 或 print(model.model)。
    
    一般来说：
    - P3 (stride 8, 80x80)  → 检测头的输入 features[0]
    - P4 (stride 16, 40x40) → 检测头的输入 features[1]  
    - P5 (stride 32, 20x20) → 检测头的输入 features[2]
    
    最简单的方式：修改 Detect head 的 forward，返回输入 features。
    """
    neck_features = []
    
    def hook_fn(module, input, output):
        # Detect head 的 input 就是 neck 的输出
        # input 是 tuple，第一个元素是 features list
        if isinstance(input, tuple):
            neck_features.clear()
            for feat in input[0]:
                neck_features.append(feat)
    
    # 注册到 Detect head
    model.model.model[-1].register_forward_hook(hook_fn)
    
    return neck_features
```

### 6.4 训练超参数

```yaml
# relation_distill.yaml

# === 模型 ===
model: yolov8m.pt  # 预训练权重
teacher: dinov2_vitl14_reg

# === 蒸馏超参 ===
lambda_distill: 1.0        # 蒸馏 loss 权重, 需要调参 [0.1, 0.5, 1.0, 2.0]
temperature: 0.07           # 余弦相似度 softmax 温度, 参考 CLIP 的默认值
level_weights:
  shallow: 1.0              # P3 权重
  middle: 1.0               # P4 权重
  deep: 1.0                 # P5 权重
max_relation_tokens: 1600   # P3 下采样后的最大 token 数

# === DINOv2 层级对应 ===
teacher_layers:
  shallow: [6, 7, 8]        # → P3
  middle: [12, 13, 14]      # → P4
  deep: [20, 21, 22]        # → P5

# === 训练配置 ===
data: coco.yaml
epochs: 300
batch: 16                   # 受 DINOv2 显存影响，可能需要减小
imgsz: 640
optimizer: SGD
lr0: 0.01
lrf: 0.01                  # cosine annealing 最终 lr 比例
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
device: 0                   # 单卡或多卡

# === 数据增强 ===
# 保持 YOLOv8 默认增强
mosaic: 1.0
mixup: 0.0
copy_paste: 0.0
```

---

## 7. Ablation 实验设计

### 7.1 实验列表

| 实验编号 | 实验名称 | 修改内容 | 目的 |
|---------|---------|---------|------|
| E1 | Baseline | YOLOv8m 标准训练 | 下界 |
| E2 | Ours-Full | P3+P4+P5 relation distill, KL, cosine sim | 完整方法 |
| E3 | d2-Reproduce | 分类分支扩展1024通道, SmoothL1 | 论文方法复现 |
| A1 | Ours-P5-only | 只在 P5 做 relation distill | 验证多尺度必要性 |
| A2 | Ours-P4-only | 只在 P4 | 同上 |
| A3 | Ours-P3-only | 只在 P3 | 同上 |
| A4 | Ours-SmoothL1 | 把 KL 换成 SmoothL1 | 验证 KL 优势 |
| A5 | Ours-MSE | 把 KL 换成 MSE | 同上 |
| A6 | Feature@Neck | 在 P3/P4/P5 做 feature mimicking (非 relation) | **最关键 ablation**: relation vs. feature |
| A7 | Ours-LastLayer | Teacher 统一只用 DINOv2 最后一层 | 验证多层级对应价值 |
| A8 | λ=0.1 | 蒸馏权重调参 | 超参敏感性 |
| A9 | λ=0.5 | 同上 | 同上 |
| A10 | λ=2.0 | 同上 | 同上 |

### 7.2 A6 实验详细设计 (Feature@Neck)

```python
class FeatureAtNeckDistiller(torch.nn.Module):
    """
    在 P3/P4/P5 位置做 feature-level distillation（非 relation-level）。
    作为 ablation，与我们的 relation distillation 做直接对比。
    
    唯一区别：用 projection + SmoothL1 对齐特征向量，
    而不是构造关系矩阵 + KL divergence。
    """
    
    def __init__(self, student_channels, teacher_dim=1024):
        """
        Args:
            student_channels: list, [P3_ch, P4_ch, P5_ch], e.g. [256, 512, 512]
            teacher_dim: DINOv2 embedding 维度
        """
        super().__init__()
        self.projectors = torch.nn.ModuleList([
            torch.nn.Conv2d(ch, teacher_dim, 1, bias=False)
            for ch in student_channels
        ])
    
    def forward(self, student_features, teacher_features_2d):
        """
        Args:
            student_features: list of [B, C_i, H_i, W_i]
            teacher_features_2d: list of [B, D, H_i, W_i] (已 resize)
        Returns:
            loss: scalar
        """
        total_loss = 0
        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features_2d)):
            projected = self.projectors[i](s_feat)  # [B, D, H, W]
            projected = F.normalize(projected, p=2, dim=1)
            t_feat = F.normalize(t_feat, p=2, dim=1)
            total_loss += F.smooth_l1_loss(projected, t_feat.detach())
        
        return total_loss / len(student_features)
```

**这个 ablation 控制了所有变量（同样的蒸馏位置、同样的 teacher 信号来源），只改变了"对齐什么"——feature vs. relation。**

---

## 8. 关键实现注意事项

### 8.1 DINOv2 输入预处理

```python
# DINOv2 和 YOLO 的预处理不同！
# YOLO: [0, 1] 归一化, BGR→RGB
# DINOv2: ImageNet 标准化

DINOV2_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINOV2_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def prepare_for_dinov2(yolo_images):
    """
    将 YOLO 格式的图像转换为 DINOv2 格式。
    
    Args:
        yolo_images: [B, 3, 640, 640], RGB, [0, 1]
    Returns:
        dinov2_images: [B, 3, 640, 640], RGB, ImageNet normalized
    
    注意：确认 YOLO 输出是 RGB 还是 BGR。
    Ultralytics 默认输出 RGB [0,1]。
    """
    mean = DINOV2_MEAN.to(yolo_images.device)
    std = DINOV2_STD.to(yolo_images.device)
    return (yolo_images - mean) / std
```

### 8.2 DINOv2 输入尺寸对齐

```python
# DINOv2 patch_size=14，输入需要是 14 的倍数
# 640 / 14 = 45.71 → 不整除！
# 
# 方案一：将 DINOv2 输入 resize 到 644 (14*46) 或 630 (14*45)
# 方案二：使用 DINOv2 的 interpolate_pos_encoding 支持任意分辨率
#
# 推荐方案一，resize 到 630x630：
# patch_grid = 630/14 = 45x45 = 2025 patch tokens

DINOV2_INPUT_SIZE = 630  # 14 * 45

def resize_for_dinov2(images):
    """
    Args:
        images: [B, 3, 640, 640]
    Returns:
        resized: [B, 3, 630, 630]
    """
    return F.interpolate(images, size=(DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), 
                         mode='bilinear', align_corners=False)
```

### 8.3 梯度管理

```python
# 关键：DINOv2 不需要梯度，但它的输出要参与 loss 计算
# 正确做法：

with torch.no_grad():
    teacher_relations = teacher.get_relation_matrices(dinov2_images)
    # teacher_relations 中的张量已经 detached

# Student 的关系矩阵需要梯度
student_relations = {}
for level, feat in zip(['shallow', 'middle', 'deep'], neck_features):
    student_relations[level] = relation_constructor(feat)  # 保留梯度

# Loss 计算时，teacher 端用 .detach()
loss = kl_loss(student_relations[level].log(), teacher_relations[level].detach())
```

### 8.4 训练日志

```python
# 建议记录的指标：
# - det_loss: 原始检测 loss
# - relation_loss: 蒸馏 loss 总和
# - relation_loss_shallow: P3 层蒸馏 loss
# - relation_loss_middle:  P4 层蒸馏 loss  
# - relation_loss_deep:    P5 层蒸馏 loss
# - mAP50, mAP50:95 (验证集)
# - AP_small, AP_medium, AP_large (验证集)
```

---

## 9. 实验执行优先级

```
Phase 1（核心实验，必须完成）:
  1. E1: Baseline (YOLOv8m vanilla)
  2. E2: Ours-Full (完整方法)
  3. E3: d2-Reproduce (论文复现)
  
Phase 2（关键 ablation）:
  4. A6: Feature@Neck (relation vs. feature, 最重要的 ablation)
  5. A4/A5: Loss 函数对比 (KL vs. SmoothL1 vs. MSE)
  
Phase 3（补充 ablation）:
  6. A1-A3: 单层级 ablation
  7. A7: 多层级对应 vs. 只用最后一层
  8. A8-A10: λ 调参

Phase 4（分析）:
  9. 可视化 (GradCAM 热图对比)
  10. AP_small/medium/large 分析
```

---

## 10. 预期算力需求

| 实验 | GPU 显存 | 预估训练时间 (300 epochs, COCO) |
|------|---------|------|
| E1 Baseline | ~8 GB | ~24h (单卡 A100) |
| E2 Ours | ~20-24 GB (DINOv2-L 约 1.2GB + 关系矩阵) | ~48-72h |
| E3 d2 | ~20-24 GB (DINOv2-L + embedding 计算) | ~48-72h |

**显存优化建议：**
- 使用 `torch.cuda.amp` 混合精度训练
- Teacher forward 全程 float16
- P3 关系矩阵下采样到 40x40
- batch size 可能需要从 16 降到 8

---

## 11. 环境部署与模型离线下载

### 11.1 前提

远程训练服务器（如 AutoDL）可能无法访问外网或访问很慢。所有模型权重需要**提前在本地或有网络的机器上下载好**，然后上传到服务器。数据集（COCO）已在服务器上准备好。

### 11.2 DINOv2 模型下载

DINOv2 通过 `torch.hub` 加载，底层是从 GitHub releases 下载 `.pth` 权重文件。

**方法一：直接下载权重文件（推荐）**

在有网络的机器上下载以下文件：

```bash
# DINOv2-Large with registers（我们使用的版本）
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth

# 备选：DINOv2-Large without registers
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth

# 备选：DINOv2-Base with registers（显存不够时可用小模型调试）
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.
