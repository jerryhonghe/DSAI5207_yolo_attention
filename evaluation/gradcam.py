"""
GradCAM heatmap generation for comparing attention patterns across experiments.

Usage:
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
    """GradCAM implementation for YOLOv8."""

    def __init__(self, model_path, target_layer_idx=-2, device='cuda'):
        self.device = device
        self.model = YOLO(model_path)
        self.yolo_model = self.model.model.to(device).eval()

        self.target_layer = self.yolo_model.model[target_layer_idx]
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, image_path, imgsz=640):
        """Generate GradCAM heatmap.

        Args:
            image_path: input image path
            imgsz: input size

        Returns:
            heatmap: [H, W] numpy array in [0, 1]
            image: resized original image as numpy array
        """
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (imgsz, imgsz))

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        img_tensor.requires_grad = True

        # Forward
        self.yolo_model.train()
        preds = self.yolo_model.predict(img_tensor)

        # Use max detection score as target for backprop
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

        # Compute GradCAM
        if self.gradients is not None and self.activations is not None:
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = cam.squeeze().detach().cpu().numpy()

            cam = cam - cam.min()
            if cam.max() > 0:
                cam = cam / cam.max()

            cam = cv2.resize(cam, (imgsz, imgsz))
        else:
            cam = np.zeros((imgsz, imgsz))

        self.yolo_model.eval()
        return cam, img_resized


def generate_gradcam_comparison(image_paths, model_paths, output_dir, imgsz=640):
    """Generate GradCAM comparison figure for multiple models.

    Args:
        image_paths: list of image file paths
        model_paths: dict {name: weight_path}
        output_dir: output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        n_models = len(model_paths)

        fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))

        # Original image
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (imgsz, imgsz))
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original', fontsize=14)
        axes[0].axis('off')

        # GradCAM per model
        colors = ['#2196F3', '#4CAF50', '#FF9800']
        for i, (name, weight_path) in enumerate(model_paths.items()):
            gradcam = YOLOGradCAM(weight_path, device=device)
            heatmap, _ = gradcam.generate(img_path, imgsz=imgsz)

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
