"""
使用光照估计器模型进行光照分量提取。
原论文GitHub地址https://github.com/caiyuanhao1998/Retinexformer
依赖：
- Python 3.x
- torch
- numpy
- opencv-python (cv2)
- natsort
- tqdm

"""

import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from natsort import natsorted
from tqdm import tqdm

from illumination_estimator_model import load_estimator


def load_img(path: str) -> np.ndarray:
    """读取图像并返回 HWC, uint8, BGR->RGB."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def save_img(path: str, img: np.ndarray) -> None:
    """保存 HWC, RGB, [0,255] 到指定路径（写成 PNG/JPEG 等）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def main():
    parser = argparse.ArgumentParser(description="Standalone illumination estimator inference")
    # 注意：如果提供了 default，就不要再设置 required=True，否则不传参数仍会报“required”错误
    parser.add_argument("--input_dir", default=r'D:\train_data\test\vi', type=str, help="输入图像目录")
    parser.add_argument("--output_dir", default=r'D:\train_data\test', type=str, help="输出目录")
    # 默认权重文件：同目录下的 illumination_estimator_only.pth
    parser.add_argument("--weights", default="./illumination_estimator_only.pth", type=str, help="illumination_estimator_only.pth 路径")
    parser.add_argument("--gpus", type=str, default="0", help="使用的 GPU，如 '0' 或 '0,1'")
    parser.add_argument(
        "--input_extensions",
        nargs="+",
        default=["*.png", "*.jpg", "*.jpeg"],
        help="要读取的图像扩展名",
    )
    # 按你的需求：只保存 illumination_features，并基于该图计算反射分量图
    # 所以不再提供 illu_map 的保存开关。
    args = parser.parse_args()

    # 设置 GPU 环境
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载独立 estimator 模型
    print(f"Loading estimator from: {args.weights}")
    model = load_estimator(args.weights).to(device)
    model.eval()

    factor = 4  # 保持与原模型一致：尺寸尽量是 4 的倍数

    os.makedirs(args.output_dir, exist_ok=True)
    illu_fea_dir = os.path.join(args.output_dir, "illumination_features")
    refl_dir = os.path.join(args.output_dir, "reflectance")
    os.makedirs(illu_fea_dir, exist_ok=True)
    os.makedirs(refl_dir, exist_ok=True)

    # 收集输入图像
    input_paths = []
    for ext in args.input_extensions:
        input_paths.extend(glob(os.path.join(args.input_dir, ext)))
        input_paths.extend(glob(os.path.join(args.input_dir, ext.upper())))
    input_paths = natsorted(input_paths)
    if not input_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir} with {args.input_extensions}")

    print(f"Found {len(input_paths)} images in {args.input_dir}")

    with torch.inference_mode():
        for path in tqdm(input_paths, desc="Extracting illumination"):
            # 原始输入图像（用于后续反射分量计算）
            img_uint8 = load_img(path)  # H,W,3 RGB uint8 [0,255]
            img = img_uint8.astype(np.float32) / 255.0  # H,W,3 float32 [0,1]
            h, w, _ = img.shape

            # HWC -> BCHW
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

            # pad 到 4 的倍数
            H = (h + factor - 1) // factor * factor
            W = (w + factor - 1) // factor * factor
            padh = H - h
            padw = W - w
            tensor_pad = F.pad(tensor, (0, padw, 0, padh), mode="reflect")

            illu_fea, illu_map = model(tensor_pad)
            illu_fea = illu_fea[:, :, :h, :w]

            illu_fea_np = illu_fea.cpu().numpy()

            base = os.path.splitext(os.path.basename(path))[0]

            # 1) 保存 illumination_features（灰度可视化图：illumination_feature 本身）
            fea = illu_fea_np[0]  # (C,H,W)
            fea_vis = fea.mean(axis=0) if fea.shape[0] > 1 else fea[0]  # (H,W)
            mn, mx = float(fea_vis.min()), float(fea_vis.max())
            if mx > mn:
                fea_vis = (fea_vis - mn) / (mx - mn)
            else:
                fea_vis = np.zeros_like(fea_vis)
            fea_vis = (fea_vis * 255).astype(np.uint8)  # (H,W) uint8

            # 保存：illumination_features/<name>.png  （这里保存的是 illumination_feature 的灰度图）
            cv2.imwrite(os.path.join(illu_fea_dir, base + ".png"), fea_vis)

            # 2) 反射分量：按“除法”计算，然后只保存亮度单通道
            # 这里反射分量依然使用 255 - illumination_feature 作为分母：
            # reflectance = input_rgb / (255 - illumination_feature)
            # 注意分母可能为 0，这里用 epsilon 防止除零
            illu_inv = (255 - fea_vis).astype(np.uint8)  # (H,W)，仅用于计算，不再单独保存
            denom = illu_inv.astype(np.float32)
            eps = 1.0  # 以 8-bit 量纲为单位的最小分母，避免除零；你也可调成 1e-6
            denom = np.maximum(denom, eps)  # (H,W)

            img_f = img_uint8.astype(np.float32)  # (H,W,3) in [0,255]
            denom3 = denom[:, :, None]  # (H,W,1) broadcast 到 RGB
            refl = img_f / denom3  # (H,W,3)

            # 为了保存成 8-bit，可视化亮度图，这里把 refl 线性映射到 [0,255]
            refl = np.clip(refl * 255.0, 0, 255).astype(np.float32)  # (H,W,3)

            # 亮度单通道：y = R*0.299 + G*0.587 + B*0.114
            # 这里 refl[...,0/1/2] 分别对应 R,G,B
            y = (
                refl[:, :, 0:1] * 0.299000
                + refl[:, :, 1:2] * 0.587000
                + refl[:, :, 2:2 + 1] * 0.114000
            )  # (H,W,1)
            y = np.clip(y, 0, 255).astype(np.uint8)
            y_gray = y[:, :, 0]  # (H,W) 单通道

            # 保存亮度图到反射分量目录
            cv2.imwrite(os.path.join(refl_dir, base + ".png"), y_gray)

    print(f"Done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()


