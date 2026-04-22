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
            relation: [B, N, N] softmax normalized relation matrix
                      N = min(H*W, max_tokens)
        """
        B, C, H, W = feature_map.shape
        num_tokens = H * W

        if num_tokens > self.max_tokens:
            target_size = int(self.max_tokens ** 0.5)
            feature_map = F.adaptive_avg_pool2d(feature_map, (target_size, target_size))
            H, W = target_size, target_size

        feat_flat = feature_map.reshape(B, C, H * W).permute(0, 2, 1)
        feat_norm = F.normalize(feat_flat, p=2, dim=-1)
        sim_matrix = torch.bmm(feat_norm, feat_norm.transpose(1, 2))
        relation = F.softmax(sim_matrix / self.temperature, dim=-1)

        return relation
