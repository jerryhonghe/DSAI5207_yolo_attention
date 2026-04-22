"""
Relation matrix visualization: compare teacher and student relation patterns.

For a selected query position, visualizes how strongly it attends to all other
positions — overlaid on the original image as a heatmap.

Usage:
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
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.torch_utils import de_parallel

from models.teacher import DINOv2Teacher
from models.relation_constructor import EfficientRelationConstructor


def _get_neck_features(model_path, image_tensor, device):
    """Extract neck features from a YOLO model."""
    model = YOLO(model_path)
    yolo_model = model.model.to(device).eval()

    neck_features = []

    def hook_fn(module, input, output):
        neck_features.clear()
        if isinstance(input, tuple) and len(input) > 0:
            feats = input[0] if isinstance(input[0], (list, tuple)) else input
            for feat in feats:
                neck_features.append(feat.detach())

    detect_head = de_parallel(yolo_model).model[-1]
    handle = detect_head.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = yolo_model.predict(image_tensor)

    handle.remove()
    return neck_features


def visualize_relation_matrices(image_path, student_model_path, teacher_weights,
                                 output_dir, query_pos=None, temperature=0.07):
    """Visualize teacher vs student relation matrices as heatmaps.

    Selects a query position and shows its attention to all other positions.

    Args:
        image_path: input image path
        student_model_path: YOLO model weight path (E2)
        teacher_weights: DINOv2 weight path
        output_dir: output directory
        query_pos: (row, col) query position, defaults to center
        temperature: softmax temperature
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read and preprocess image
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_640 = cv2.resize(img_rgb, (640, 640))
    img_tensor = torch.from_numpy(img_640).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Teacher relation matrices
    teacher = DINOv2Teacher(teacher_weights, device=device, temperature=temperature)
    teacher_relations = teacher.get_relation_matrices(img_tensor)

    # Student relation matrices
    neck_features = _get_neck_features(student_model_path, img_tensor, device)
    constructor = EfficientRelationConstructor(temperature=temperature, max_tokens=1600)

    student_relations = {}
    level_names = ['shallow', 'middle', 'deep']
    for i, level in enumerate(level_names):
        if i < len(neck_features):
            student_relations[level] = constructor(neck_features[i])

    # Visualize each level
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

        # Original + query marker
        axes[0].imshow(img_640)
        scale = 640 / size
        axes[0].plot(qc * scale + scale / 2, qr * scale + scale / 2, 'r*', markersize=15)
        axes[0].set_title(f'Query Position ({level})', fontsize=14)
        axes[0].axis('off')

        # Teacher heatmap
        t_heatmap = cv2.resize(t_rel, (640, 640))
        t_colored = plt.cm.hot(t_heatmap / (t_heatmap.max() + 1e-8))[:, :, :3] * 255
        t_overlay = (0.5 * img_640 + 0.5 * t_colored).astype(np.uint8)
        axes[1].imshow(t_overlay)
        axes[1].set_title(f'Teacher ({level})', fontsize=14)
        axes[1].axis('off')

        # Student heatmap
        s_heatmap = cv2.resize(s_rel, (640, 640))
        s_colored = plt.cm.hot(s_heatmap / (s_heatmap.max() + 1e-8))[:, :, :3] * 255
        s_overlay = (0.5 * img_640 + 0.5 * s_colored).astype(np.uint8)
        axes[2].imshow(s_overlay)
        axes[2].set_title(f'Student ({level})', fontsize=14)
        axes[2].axis('off')

        plt.suptitle(f'Relation Matrix Visualization - {level}', fontsize=16)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'relation_{img_name}_{level}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
