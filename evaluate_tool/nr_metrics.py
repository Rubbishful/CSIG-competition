"""无参考(NR)评估:pyiqa 指标适配层 + 单图启发式退化指纹。

指标(NIQE / MANIQA / MUSIQ)使用本地权重,避免运行时联网下载:
- NIQE : HYPIR_model/niqe_modelparameters.mat(pyiqa 支持 pretrained_model_path,纯 Python 实现)
- MANIQA: HYPIR_model/MANIQA.pt —— pyiqa 加载硬编码 weight_keys='params',
          而本地文件无该包装,首次使用时自动打包为 MANIQA_params.pt 缓存
- MUSIQ : HYPIR_model/MUSIQ.pth(格式与 pyiqa 默认一致,直接传路径)

注意: MANIQA 架构内部会尝试联网下载 timm 的 vit_base_patch8_224 预训练权重,
加载时通过 HF_HUB_OFFLINE=1 让其立即失败降级(零网络依赖)——MANIQA.pt
自带完整 vit 权重,strict 加载会整体覆盖,指标不受 timm 初始化影响;
若个别 timm 版本离线直接抛异常,会自动回退在线模式重试一次。
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "HYPIR_model"

NIQE_PARAM_PATH = MODEL_DIR / "niqe_modelparameters.mat"
MANIQA_PATH = MODEL_DIR / "MANIQA.pt"
MUSIQ_PATH = MODEL_DIR / "MUSIQ.pth"

_LOCK = threading.Lock()


@contextlib.contextmanager
def _offline_hf():
    """临时设 HF_HUB_OFFLINE=1,退出时恢复原值。

    用于 MANIQA 加载:timm 的 vit_base_patch8_224 预训练权重会被 MANIQA.pt
    完整覆盖,联网下载纯属浪费(~90s 重试噪音);离线让下载立即失败并降级为
    随机初始化,随后由 MANIQA.pt 覆盖,指标不受影响。
    """
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev


def _prepare_maniqa(path: Path) -> Path:
    """pyiqa 硬编码 weight_keys='params';本地 MANIQA.pt 无包装时打包一次缓存。"""
    packed = path.with_name(path.stem + "_params" + path.suffix)
    if packed.exists():
        return packed
    with _LOCK:
        if packed.exists():  # 双检锁,避免并发重复打包
            return packed
        import torch

        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "params" in sd and isinstance(sd["params"], dict):
            return path  # 已是包装格式,直接用
        torch.save({"params": sd}, packed)
        print(f"[nr] 本地 MANIQA.pt 无 'params' 包装,已生成 pyiqa 兼容权重: {packed}")
    return packed


class NRMetricSet:
    """pyiqa NR 指标集合(惰性加载,本地权重)。"""

    def __init__(self, models: list[str], device: str | None = None):
        import torch

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._metrics: dict[str, object] = {}
        self._input_size: dict[str, int | None] = {"niqe": None, "musiq": None}
        for name in models:
            self._load(name)
        if not self._metrics:
            raise ValueError("未成功加载任何 NR 指标")

    def _load(self, name: str) -> None:
        import pyiqa

        if name not in ("niqe", "maniqa", "musiq"):
            raise ValueError(f"不支持的 NR 指标: {name}(可选 niqe/maniqa/musiq)")

        try:
            if name == "niqe":
                if not NIQE_PARAM_PATH.exists():
                    raise FileNotFoundError(
                        f"缺少 NIQE 模型参数: {NIQE_PARAM_PATH}(需提前放置 niqe_modelparameters.mat)"
                    )
                self._metrics[name] = pyiqa.create_metric(
                    "niqe", device=self.device, pretrained_model_path=str(NIQE_PARAM_PATH)
                )
            elif name == "maniqa":
                if not MANIQA_PATH.exists():
                    raise FileNotFoundError(f"缺少 MANIQA 权重: {MANIQA_PATH}")
                path = _prepare_maniqa(MANIQA_PATH)
                try:
                    # 离线加载: 避免 timm vit 权重的联网下载重试(~90s)噪音;
                    # MANIQA.pt 自带完整 vit 权重,下载失败降级随机初始化后会被覆盖,不影响指标。
                    with _offline_hf():
                        self._metrics[name] = pyiqa.create_metric(
                            "maniqa", device=self.device,
                            pretrained_model_path=str(path),
                        )
                except Exception:
                    # 个别 timm 版本在 HF_HUB_OFFLINE=1 下直接抛异常而非降级,
                    # 此时回退在线模式重试一次(行为与优化前一致)。
                    print("[nr] MANIQA 离线加载失败,回退在线模式重试一次 ...")
                    self._metrics[name] = pyiqa.create_metric(
                        "maniqa", device=self.device,
                        pretrained_model_path=str(path),
                    )
                self._input_size["maniqa"] = 448  # MANIQA 固定输入
            elif name == "musiq":
                if not MUSIQ_PATH.exists():
                    raise FileNotFoundError(f"缺少 MUSIQ 权重: {MUSIQ_PATH}")
                self._metrics[name] = pyiqa.create_metric(
                    "musiq", device=self.device, pretrained_model_path=str(MUSIQ_PATH)
                )
        except Exception as e:
            raise RuntimeError(
                f"加载 NR 指标 '{name}' 失败: {type(e).__name__}: {e}\n"
                f"  若为网络下载错误(timm/HuggingFace),需先联网完成权重下载;"
                f"若为本地文件问题,请检查 HYPIR_model 下的权重文件。"
            ) from e

    def available(self) -> list[str]:
        return list(self._metrics)

    def score(self, name: str, img: np.ndarray) -> float:
        """对单张 uint8 RGB 图打分(越小越差/越大越好取决于指标,见注释)。"""
        import torch

        metric = self._metrics[name]
        t = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1)) / 255.0
        size = self._input_size.get(name)
        if size is not None:
            from PIL import Image

            h, w = img.shape[:2]
            scale = max(1, size // max(h, w))  # 放大到至少 size
            new_h, new_w = h * scale, w * scale
            if new_h != h or new_w != w:
                t = torch.from_numpy(
                    np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
                    .astype(np.float32).transpose(2, 0, 1)
                ) / 255.0
        t = t.unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = metric(t)
        return float(out.reshape(-1)[0].cpu().item())


def heuristic_fingerprint(img: np.ndarray) -> dict:
    """单图启发式退化指纹(无 GT 时粗估):模糊度/噪声/偏色/边缘振铃。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = gx * gx + gy * gy
    sharpness = float(np.sqrt(mag).mean())          # 越大越清晰(高频能量)

    flat = mag < np.percentile(mag, 10)             # 最平坦 10% 像素
    noise_est = float(gray[flat].std()) if flat.any() else 0.0  # 平坦区噪声估计

    means = img.reshape(-1, 3).mean(axis=0)         # 通道均值偏色
    gray_mean = float(means.mean())
    color_cast = float(np.sqrt(np.sum((means - gray_mean) ** 2)))

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge = mag > np.percentile(mag, 95)             # 强边缘处拉普拉斯响应(振铃振荡)
    edge_ringing = float(np.abs(lap)[edge].mean()) if edge.any() else 0.0

    return {
        "sharpness": round(sharpness, 4),
        "noise_est": round(noise_est, 4),
        "color_cast": round(color_cast, 4),
        "edge_ringing": round(edge_ringing, 4),
    }


__all__ = ["NRMetricSet", "heuristic_fingerprint", "NIQE_PARAM_PATH", "MANIQA_PATH", "MUSIQ_PATH"]
