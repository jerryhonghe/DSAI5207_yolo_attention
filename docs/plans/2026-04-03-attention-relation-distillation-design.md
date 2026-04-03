# Attention Relation Distillation from DINOv2 to YOLO - 设计文档

## 项目目标

用 frozen DINOv2 的多层 patch features 构造余弦相似度关系矩阵作为 teacher signal，通过 KL divergence 蒸馏到 YOLOv8m neck 层 P3/P4/P5 特征上。推理时 teacher 完全移除，YOLO 结构和速度不变。

## 实验范围 (Phase 1)

| 实验 | 说明 |
|------|------|
| E1 Baseline | YOLOv8m 标准训练 |
| E2 Ours-Full | P3+P4+P5 relation distillation, KL divergence |
| E3 d2-Reproduce | 分类分支扩展1024通道, SmoothL1对齐DINOv2 patch tokens |

- 数据集：标准 COCO，位于服务器 `/root/autodl-tmp/hh/attention_yolo/data`
- 训练轮数：100 epochs（快速验证）
- 验证频率：每 10 epochs
- GPU：RTX 5090 32GB

## 技术方案：深度集成 Ultralytics

继承 `DetectionTrainer`，复用数据加载、验证、日志、checkpoint/resume 等基础设施，仅重写 loss 计算注入蒸馏逻辑。

## 项目结构

```
yolo_attention/
├── configs/
│   ├── baseline.yaml
│   ├── relation_distill.yaml
│   └── feature_distill_d2.yaml
├── models/
│   ├── teacher.py                 # DINOv2Teacher 离线加载
│   ├── relation_constructor.py    # Student端余弦相似度关系矩阵
│   └── d2_head.py                 # d2方法投影头
├── losses/
│   ├── relation_loss.py           # KL divergence loss
│   └── feature_loss.py            # SmoothL1 loss
├── trainers/
│   ├── base_distill_trainer.py    # 基类：teacher加载、neck hook、每10轮验证
│   ├── relation_trainer.py        # E2 关系蒸馏trainer
│   └── d2_trainer.py              # E3 feature蒸馏trainer
├── evaluation/
│   ├── gradcam.py                 # GradCAM热力图
│   ├── relation_vis.py            # 关系矩阵可视化
│   └── compare.py                 # 三组实验对比
├── weights/
│   ├── yolov8m.pt
│   └── dinov2_vitl14_reg4_pretrain.pth
├── train.py                       # 统一训练入口
├── evaluate.py                    # 评估入口
└── requirements.txt
```

## 核心数据流 (E2)

```
输入图像 [B,3,640,640]
    │
    ├─► YOLOv8m ──► Neck P3/P4/P5
    │                   │
    │                   ├─► P3 下采样40x40 ──► 余弦相似度 ──► softmax ──► S_student
    │                   ├─► P4 直接算 ──────► 余弦相似度 ──► softmax ──► S_student
    │                   └─► P5 直接算 ──────► 余弦相似度 ──► softmax ──► S_student
    │                                                                       │
    │                                                                 KL Divergence
    │                                                                       │
    └─► resize 630x630 ──► DINOv2 frozen                                    │
              ├─► 浅层(L6-8) ──► resize 40x40 ──► 余弦相似度 ──► softmax ──► S_teacher
              ├─► 中层(L12-14) ► resize 40x40 ──► 余弦相似度 ──► softmax ──► S_teacher
              └─► 深层(L20-22) ► resize 20x20 ──► 余弦相似度 ──► softmax ──► S_teacher

L_total = L_det + λ * (KL_shallow + KL_middle + KL_deep)
```

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| P3 下采样 | 40x40 | 避免 6400² 关系矩阵爆显存 |
| DINOv2 输入 | 630x630 | patch_size=14 的倍数 |
| 温度 τ | 0.07 | 参考 CLIP |
| 蒸馏权重 λ | 1.0 | 默认值 |
| batch size | 8 | 32GB 显存约束 |
| 混合精度 | 是 | teacher forward 用 float16 |

## 评估方法

1. **mAP 指标**：mAP50, mAP50:95, AP_small/medium/large
2. **GradCAM 热力图**：三组方法并排对比关注区域
3. **关系矩阵热力图**：teacher vs student 关系矩阵可视化

## 离线部署

- YOLOv8m 权重：本机下载后上传到 `weights/yolov8m.pt`
- DINOv2-L 权重：本机下载后上传到 `weights/dinov2_vitl14_reg4_pretrain.pth`
- Python 依赖：本机下载 wheel 包，离线安装
- 服务器无需联网
