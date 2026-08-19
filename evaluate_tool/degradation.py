"""退化指纹分析:残差结构分析 + 退化构成打分。

方法复刻 doc/初步分析.md 实验 4:
- 残差 r = LQ − GT
- 残差RMS = sqrt(mean(r²))            → 整体退化程度
- 高频占比 = (r−Gaussian(r,1))² / r²  → 高=噪声/压缩主导,低=模糊主导
- 边缘掩码 = GT Sobel 梯度前 85% 分位 → 边缘残差 / 平坦残差
- 通道残差均值 → 偏色量(color cast)

退化构成打分与主导标签的阈值由 doc 实验 4 的五张验证图标定(见模块常量说明),
输出连续分数(0~1)便于跨图/跨轮次比较,同时给出主导退化标签。

输入统一为 uint8 RGB numpy 数组 (H, W, 3)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# ---- 退化严重度(RMS 分级,标定自 doc 实验 4: case1=6.4 轻 / case2=9.9 中 / case4=31.9 重) ----
SEVERITY_LIGHT = 8.0      # rms < 8 → 轻度
SEVERITY_MEDIUM = 20.0    # 8 ≤ rms < 20 → 中度; rms ≥ 20 → 重度

# ---- 成分判定阈值(各成分独立判定,不再"偏色一票否决") ----
BLUR_HF = 0.08            # hf ≤ 0.08 → 模糊成分(低频主导;case1=0.01/case2=0.06/case4=0.04/case5=0.07)
NOISE_HF = 0.12           # hf ≥ 0.12 → 噪声成分(case3=0.18)
NOISE_HF_LO = 0.05        # 中间带 0.05~0.12 且平坦残差高 → 模糊+噪声混合(case2=0.06,flat=4.7)
NOISE_FLAT_RES = 3.0      # 平坦残差阈值(case1 纯模糊 flat=1.5,case2/case5 flat>4 含噪声)
CAST_THRESHOLD = 2.0      # 通道残差均值 L2 范数 > 2.0 → 偏色成分(case4≈16.9,其余≤1.4)
EDGE_RATIO = 3.0          # 边缘/平坦残差比 ≥ 3.0 且边缘残差 ≥ EDGE_RES → 边缘伪影成分(case5=3.6)
EDGE_RES = 12.0           # 边缘残差绝对阈值(case5=18.7,case2=13.8 但比值不足)


@dataclass
class ResidualFingerprint:
    """残差结构指纹(对应 doc 实验 4 表的一行)。"""

    rms: float                       # 残差RMS
    hf_ratio: float                  # 高频占比
    edge_res: float                  # 边缘掩码处 |r| 均值
    flat_res: float                  # 平坦区 |r| 均值
    edge_flat_ratio: float           # 边缘残差 / 平坦残差
    channel_means: list              # 各通道残差均值 [R, G, B]
    cast_norm: float                 # 通道残差均值 L2 范数(偏色量)
    per_channel: dict = field(default_factory=dict)  # 保留原始逐项供扩展


def residual_fingerprint(lq: np.ndarray, gt: np.ndarray) -> ResidualFingerprint:
    """计算 LQ 相对 GT 的残差结构指纹。lq/gt 为 uint8 RGB (H,W,3),要求同尺寸。"""
    if lq.shape != gt.shape:
        raise ValueError(f"lq/gt shape mismatch: {lq.shape} vs {gt.shape}")
    a = lq.astype(np.float64)
    b = gt.astype(np.float64)
    r = a - b

    rms = float(np.sqrt(np.mean(r**2)))

    # 高频占比: 残差减去高斯平滑(σ=1)后能量 / 总能量
    r_smooth = gaussian_filter(r, sigma=1.0, axes=(0, 1), mode="reflect")
    hf = r - r_smooth
    hf_ratio = float(np.mean(hf**2) / max(np.mean(r**2), 1e-12))

    # 边缘掩码: GT 灰度 Sobel 梯度幅值前 85% 分位
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).astype(np.float64)
    gx = cv2.Sobel(gt_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gt_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    thr = np.percentile(grad_mag, 85)
    edge_mask = grad_mag > thr
    flat_mask = ~edge_mask

    edge_res = float(np.mean(np.abs(r)[edge_mask]))
    flat_res = float(np.mean(np.abs(r)[flat_mask]))
    edge_flat_ratio = float(edge_res / max(flat_res, 1e-12))

    # 通道残差均值(偏色)
    channel_means = [float(np.mean(r[..., c])) for c in range(3)]
    cast_norm = float(np.sqrt(sum(m * m for m in channel_means)))

    return ResidualFingerprint(
        rms=rms,
        hf_ratio=hf_ratio,
        edge_res=edge_res,
        flat_res=flat_res,
        edge_flat_ratio=edge_flat_ratio,
        channel_means=channel_means,
        cast_norm=cast_norm,
    )


def degradation_scores(fp: ResidualFingerprint) -> dict:
    """把指纹映射为 0~1 的退化构成分数(连续,便于比较与排序)。"""
    cast_score = float(min(1.0, fp.cast_norm / 5.0))
    noise_score = float(min(1.0, max(0.0, (fp.hf_ratio - 0.05) / 0.15)))
    blur_score = float(min(1.0, max(0.0, (0.15 - fp.hf_ratio) / 0.15)))
    # 边缘伪影: 边缘残差大且高频占比处于中等带(非纯模糊/非纯噪声)
    mid_band = float(np.exp(-((fp.hf_ratio - 0.09) ** 2) / (2 * 0.04**2)))
    edge_artifact_score = float(min(1.0, fp.edge_res / 25.0) * mid_band)
    return {
        "cast_score": cast_score,
        "noise_score": noise_score,
        "blur_score": blur_score,
        "edge_artifact_score": edge_artifact_score,
    }


def severity_of(rms: float) -> str:
    """整体退化严重度(基于残差RMS): 轻度/中度/重度。"""
    if rms < SEVERITY_LIGHT:
        return "轻度"
    if rms < SEVERITY_MEDIUM:
        return "中度"
    return "重度"


def degradation_profile(fp: ResidualFingerprint) -> dict:
    """退化构成画像: 严重度 + 成分列表 + 结构主导成分。

    与旧"单一主导标签"的区别:
    - 偏色不再一票否决,各成分独立判定(多类型混合可见);
    - 输出严重度,重度结构退化(如 case4 模糊+偏色)不会被偏色掩盖;
    - dominant 只在结构成分(模糊/噪声/边缘伪影)中取分最高者,
      因为对恢复任务而言结构退化比颜色退化更本质。
    """
    components: list[str] = []
    if fp.hf_ratio <= BLUR_HF:
        components.append("模糊")
    if fp.hf_ratio >= NOISE_HF or (
        NOISE_HF_LO <= fp.hf_ratio < NOISE_HF and fp.flat_res > NOISE_FLAT_RES
    ):
        components.append("噪声")
    if fp.edge_flat_ratio >= EDGE_RATIO and fp.edge_res >= EDGE_RES:
        components.append("边缘伪影")
    if fp.cast_norm > CAST_THRESHOLD:
        components.append("偏色")

    scores = degradation_scores(fp)
    structural = {
        name: scores[key]
        for name, key in [
            ("模糊", "blur_score"),
            ("噪声", "noise_score"),
            ("边缘伪影", "edge_artifact_score"),
        ]
        if name in components
    }
    dominant = max(structural, key=structural.get) if structural else None
    severity = severity_of(fp.rms)
    label = f"{severity}·{'+'.join(components)}" if components else f"{severity}·轻微退化"
    return {
        "severity": severity,
        "components": components,
        "dominant": dominant,
        "label": label,
    }


def classify_degradation(fp: ResidualFingerprint) -> str:
    """兼容入口:返回退化构成画像的标签(如 "重度·模糊+偏色")。"""
    return degradation_profile(fp)["label"]
