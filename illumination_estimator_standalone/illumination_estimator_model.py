"""
Standalone illumination estimator model.

这个文件是从 RetinexFormer 中“抠出来”的光照估计子网络，做了最小化改造：
- 只依赖 PyTorch（torch, torch.nn）
- 不依赖 basicsr / RetinexFormer 代码仓库

搭配导出的精简权重 illumination_estimator_only.pth 使用：
    from illumination_estimator_model import load_estimator
    model = load_estimator("illumination_estimator_only.pth").to(device)
    illu_fea, illu_map = model(img)   # img: [B,3,H,W] in [0,1]
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IlluminationEstimator(nn.Module):
    """
    与原仓库中 basicsr.models.archs.RetinexFormer_arch.Illumination_Estimator 结构一致，
    以保证能无缝加载裁剪后的权重。
    """

    def __init__(self, n_fea_middle: int, n_fea_in: int = 4, n_fea_out: int = 3) -> None:
        """
        Args:
            n_fea_middle: 中间特征通道数（原 RetinexFormer 中等于 n_feat）
            n_fea_in: 输入特征通道数（RGB 3 通道 + mean_c 1 通道 = 4）
            n_fea_out: 输出光照图通道数（3 通道）
        """
        super().__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        # 注意：groups = n_fea_in，这一点要与原模型保持完全一致，否则无法加载权重
        self.depth_conv = nn.Conv2d(
            n_fea_middle,
            n_fea_middle,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=n_fea_in,
        )

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            img: [B, 3, H, W]，范围通常为 [0,1]

        Returns:
            illu_fea: [B, C_mid, H, W] 光照特征
            illu_map: [B, 3, H, W]     光照图
        """
        # mean_c: B,1,H,W
        mean_c = img.mean(dim=1, keepdim=True)

        # 拼接 RGB 与亮度均值
        x = torch.cat([img, mean_c], dim=1)  # B,4,H,W

        x_1 = self.conv1(x)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map


def _load_state_dict_and_nfeat(weights_path: str) -> tuple[dict, int]:
    """
    读取 illumination_estimator_only.pth 中的 state_dict 和 n_feat。

    兼容两种格式：
    - 导出脚本 export_illumination_estimator.py 生成的：
        {"type": "...", "n_feat": int, "params": {...}}
    - 直接保存的 state_dict：
        {"conv1.weight": ..., "depth_conv.weight": ..., ...}
    """
    ckpt = torch.load(weights_path, map_location="cpu")

    if isinstance(ckpt, dict) and "params" in ckpt and isinstance(ckpt["params"], dict):
        sd = ckpt["params"]
        n_feat = int(ckpt.get("n_feat", 31))
        return sd, n_feat

    if isinstance(ckpt, dict):
        # 纯 state_dict，尽力从 meta 里取 n_feat，否则用默认值
        n_feat = int(ckpt.get("n_feat", 31)) if "n_feat" in ckpt else 31
        return ckpt, n_feat

    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


def load_estimator(weights_path: str, strict: bool = True) -> IlluminationEstimator:
    """
    从精简权重文件创建并加载一个独立的 IlluminationEstimator 模型。

    Args:
        weights_path: illumination_estimator_only.pth 路径
        strict: 是否严格匹配 state_dict
    """
    sd, n_feat = _load_state_dict_and_nfeat(weights_path)
    model = IlluminationEstimator(n_fea_middle=n_feat)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if not strict and (missing or unexpected):
        print("[Warn] load_state_dict not strictly matched:")
        if missing:
            print("  - missing keys:", missing)
        if unexpected:
            print("  - unexpected keys:", unexpected)
    return model


__all__ = ["IlluminationEstimator", "load_estimator"]


