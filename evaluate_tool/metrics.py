"""全参考(FR)基础指标:PSNR / SSIM / LPIPS。

输入统一为 uint8 RGB numpy 数组(H, W, 3),PSNR/SSIM 在内部转 float 计算。
LPIPS 依赖 lpips 库(权重已在本地缓存),可选按最长边缩放以控制 4K 图的计算开销。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

_lpips_cache: dict = {}


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """RGB 全图 MSE → PSNR(dB),值域按 0-255。"""
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def _ssim_single(a: np.ndarray, b: np.ndarray, win: np.ndarray, c1: float, c2: float) -> float:
    """单通道 SSIM(高斯加权)。a/b 为 [0,1] float。"""
    mu1 = gaussian_filter(a, sigma=1.5, mode="reflect")
    mu2 = gaussian_filter(b, sigma=1.5, mode="reflect")
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    s1_sq = gaussian_filter(a * a, sigma=1.5, mode="reflect") - mu1_sq
    s2_sq = gaussian_filter(b * b, sigma=1.5, mode="reflect") - mu2_sq
    s12 = gaussian_filter(a * b, sigma=1.5, mode="reflect") - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (s1_sq + s2_sq + c2)
    )
    return float(ssim_map.mean())


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """标准 SSIM(11x11 高斯窗, σ=1.5),RGB 三通道均值。输入 uint8 [0,255]。"""
    a = img1.astype(np.float64) / 255.0
    b = img2.astype(np.float64) / 255.0
    c1 = 0.01**2
    c2 = 0.03**2
    vals = [_ssim_single(a[..., c], b[..., c], None, c1, c2) for c in range(3)]
    return float(np.mean(vals))


def lpips_score(
    img1: np.ndarray,
    img2: np.ndarray,
    net: str = "alex",
    max_side: int | None = 1024,
    device: str | None = None,
) -> float:
    """LPIPS 感知距离(越小越好)。4K 图默认缩放到最长边 max_side 内再算。"""
    import torch
    import lpips as lpips_lib

    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        if max_side is not None:
            h, w = img.shape[:2]
            scale = min(1.0, max_side / max(h, w))
            if scale < 1.0:
                from PIL import Image

                img = np.asarray(
                    Image.fromarray(img).resize(
                        (max(1, round(w * scale)), max(1, round(h * scale))),
                        Image.LANCZOS,
                    )
                )
        t = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1)) / 255.0
        return t.unsqueeze(0)

    device_ = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = (net, str(device_))
    if key not in _lpips_cache:
        _lpips_cache[key] = lpips_lib.LPIPS(net=net).to(device_)
    loss_fn = _lpips_cache[key]
    with torch.no_grad():
        d = loss_fn(_to_tensor(img1).to(device_), _to_tensor(img2).to(device_))
    return float(d.item())
