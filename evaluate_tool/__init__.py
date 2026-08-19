"""evaluate_tool: HYPIR 赛题批量分析与评估工具链。

提供两类评估:
- paired(成对,有 GT):PSNR / SSIM / LPIPS + 残差结构指纹(模糊/噪声/偏色/边缘伪影)
- nr(无参考,无 GT):NIQE / MANIQA / MUSIQ + 单图启发式退化指纹
"""

__version__ = "0.1.0"
