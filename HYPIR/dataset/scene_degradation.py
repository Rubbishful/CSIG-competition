# -*- coding: utf-8 -*-
"""基线场景退化合成管线。

本模块实现设计书《设计书V3基线实现.md》所述的"最小可用退化合成管线"：

    离线:
      GT_sRGB -> inverse_isp(固定参考相机, smoothstep逆) -> linear_raw/{name}.npy
      运动模糊合法角度表 -> preprocessed/motion_blur_legal_angles.json
      Flare 模板 -> flare_templates/512/{id}.npy

    在线 (DataLoader worker):
      load GT_linear
      [1] 运动模糊 (可选): Bresenham 核卷积
      [2] 传感器噪声: 归一化 beta 域 + 可选彩色噪声 + 可选暗部增强
      [3] 正向 ISP (固定参考相机): WB -> CCM -> smoothstep -> [可选] 风格化
      [4] 数码变焦 + JPEG: Down-Up + Double-JPEG (无网格偏移)
      [5] 眩光叠加 (可选): Screen 混合

全局约定:
    - 所有 numpy 数组: RGB 通道顺序、float32、范围 [0,1]
    - cv2 默认 BGR，读写必须显式 cvtColor
    - 随机数统一使用 np.random.Generator (不使用旧 API/全局随机)

注意: 基线中 inverse_isp 不在线执行（已离线缓存为 linear_raw），
因此不出现在在线调度阶段中。
"""

import glob
import json
import os

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------- #
# 常量
# ---------------------------------------------------------------------------- #

# 在线调度默认阶段（不含 inverse_isp）
DEFAULT_STAGES = ['motion_blur', 'sensor_noise', 'forward_isp', 'zoom_jpeg', 'flare']

# 离线预处理用的固定基准（无扰动，区间中点）。
# 注意：preprocess.run_preprocess 现从配置取中点，不再依赖本常量。
# 保留用于向后兼容/参考；修改默认配置区间时应同步本常量或直接改用 cfg 中点。
WB_GAINS_FIXED = np.array([2.35, 1.0, 2.6], dtype=np.float32)


# ---------------------------------------------------------------------------- #
# 基础工具
# ---------------------------------------------------------------------------- #

def load_srgb(path: str) -> np.ndarray:
    """加载图像为 [H,W,3] float32 [0,1] RGB。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def deep_merge(a: dict, b: dict) -> dict:
    """深度合并两个 dict（b 优先）。"""
    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _project_root() -> str:
    """返回项目根目录（两个层级向上：HYPIR/dataset -> HYPIR）。"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve_path(manifest_dir: str, record: dict, key: str) -> str:
    """将 manifest 中的相对路径解析为绝对路径。返回 None 表示字段不存在。

    解析基准为 manifest 所在目录（与 run_preprocess 产出的相对路径一致）。
    若以 manifest_dir 拼接后不存在，回退到 cwd 相对路径再尝试（测试数据准备脚本
    以 cwd 为基准写 flare_bank_path / flare3d_* 路径），保证两种布局都能工作；
    对训练管线现有布局（manifest_dir 相对路径存在）无影响。
    """
    path = record.get(key)
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    rel = os.path.join(manifest_dir, path)
    if os.path.exists(rel):
        return rel
    # 回退：manifest_dir 相对解析不存在，尝试 cwd 相对路径
    if os.path.exists(path):
        return path
    return rel


# ---------------------------------------------------------------------------- #
# 数据契约：PreprocessManager
# ---------------------------------------------------------------------------- #

class PreprocessManager:
    def __init__(self, manifest_path: str, expected_version: str):
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        if self.manifest['preprocess_version'] != expected_version:
            raise RuntimeError(
                f"Version mismatch: manifest={self.manifest['preprocess_version']}, "
                f"expected={expected_version}"
            )
        # 字典索引
        self._index = {f['name']: f for f in self.manifest['files']}

    def get(self, name: str) -> dict:
        if name not in self._index:
            raise KeyError(f"Record not found: {name}")
        return self._index[name]


# ---------------------------------------------------------------------------- #
# Tone Mapping (smoothstep)
# ---------------------------------------------------------------------------- #

def smoothstep_forward(x: np.ndarray) -> np.ndarray:
    """正向 smoothstep: 输入 [0,1] -> 输出 [0,1]"""
    x = np.clip(x, 0.0, 1.0)
    return 3.0 * x ** 2 - 2.0 * x ** 3


def smoothstep_inverse(y: np.ndarray) -> np.ndarray:
    """逆向 smoothstep: 输入 [0,1] -> 输出 [0,1]"""
    y = np.clip(y, 0.0, 1.0)
    # 防止 arcsin 数值溢出
    arg = np.clip(1.0 - 2.0 * y, -1.0, 1.0)
    return 0.5 - np.sin(np.arcsin(arg) / 3.0)


# ---------------------------------------------------------------------------- #
# 逆 ISP（固定华为 P60 Pro 参考相机）
# ---------------------------------------------------------------------------- #

def apply_inverse_ccm(I: np.ndarray, M_ccm: np.ndarray) -> np.ndarray:
    """逆 CCM（预求逆矩阵乘，数学等价于 solve，速度约 20 倍）。

    I: [H,W,3] float32
    M_ccm: [3,3] float32, forward CCM
    Returns: [H,W,3] float32, M_ccm @ I 的逆映射
    """
    M_inv = np.linalg.inv(M_ccm)  # 3x3 正态矩阵，求逆代价可忽略
    I_flat = I.reshape(-1, 3).T  # [3, H*W]
    I_out = (M_inv @ I_flat).T  # [H*W, 3]
    return I_out.reshape(I.shape).astype(np.float32)


def highlight_preserving_inverse(x: np.ndarray, g: float, t: float = 0.9) -> np.ndarray:
    """正确的逆 WB：g=1 恒等，饱和保持，alpha 截断。

    Input:  x [H,W] float32 [0,1], g 标量增益, t 高光阈值
    Output: [H,W] float32 [0,1]
    """
    x = np.clip(x, 0.0, 1.0)

    # alpha 截断到 [0,1]
    alpha = np.clip((x - t) / max(1.0 - t, 1e-6), 0.0, 1.0)
    alpha = alpha ** 2

    # 线性部分
    linear = x / g

    # 高光部分: Hermite 插值, 保证连续性和恒等性
    # h(x) = [x²(g-1) + x(1-g·t)] / [g·(1-t)]
    denom = g * max(1.0 - t, 1e-6)
    cubic = (x ** 2 * (g - 1.0) + x * (1.0 - g * t)) / denom

    # float32 数值泄漏可能使结果略超 [0,1]（如 g=2.35 时 f(1)≈1.0000001），
    # 按输出契约严格裁剪到 [0,1]，保持 g=1 恒等 / 饱和 / 线性区 / 单调性不变。
    return np.clip(linear * (1.0 - alpha) + cubic * alpha, 0.0, 1.0).astype(np.float32)


def inverse_isp(I_srgb: np.ndarray, vc: dict, cfg: dict) -> np.ndarray:
    """完整逆 ISP。

    Input:  I_srgb [H,W,3] float32 [0,1] sRGB RGB
            vc: virtual camera params (含 wb_gains, ccm_matrix)
            cfg: config dict
    Output: I_raw [H,W,3] float32 [0,1] Linear pseudo-RAW RGB
    """
    # 1. 逆 Tone Mapping (smoothstep 逆)
    I_lin = smoothstep_inverse(I_srgb)

    # 2. 逆 CCM (固定矩阵, 预求逆矩阵乘)
    M_ccm = vc['ccm_matrix']
    I_lin_cam = apply_inverse_ccm(I_lin, M_ccm)
    I_lin_cam = np.clip(I_lin_cam, 0.0, 1.0)

    # 3. 逆 WB (highlight-preserving, 修复版)
    g_r, g_g, g_b = vc['wb_gains']
    I_raw = np.empty_like(I_lin_cam)
    I_raw[..., 0] = highlight_preserving_inverse(I_lin_cam[..., 0], g_r, t=0.9)
    I_raw[..., 1] = highlight_preserving_inverse(I_lin_cam[..., 1], g_g, t=0.9)
    I_raw[..., 2] = highlight_preserving_inverse(I_lin_cam[..., 2], g_b, t=0.9)

    return I_raw.astype(np.float32)


# ---------------------------------------------------------------------------- #
# 虚拟相机参数采样（参数一致性）
# ---------------------------------------------------------------------------- #

def sample_wb_gains(rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """以固定基准为中心的小扰动，不做全区间重采样。"""
    wb = cfg['forward_isp']['white_balance']
    # 固定基准 = 区间中点
    g_r_base = float(np.mean(wb['r_gain_range']))  # 2.35
    g_b_base = float(np.mean(wb['b_gain_range']))  # 2.6
    g_g = wb['g_gain']  # 1.0

    if rng.random() < wb['perturbation_prob']:
        delta = wb['perturbation_range']
        g_r = g_r_base * (1.0 + rng.uniform(*delta))
        g_b = g_b_base * (1.0 + rng.uniform(*delta))
    else:
        g_r, g_b = g_r_base, g_b_base

    return np.array([g_r, g_g, g_b], dtype=np.float32)


def sample_ccm_matrix(rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """采样 CCM 矩阵 (固定基准 + 可选扰动)。"""
    cc = cfg['forward_isp']['color_correction']
    M_base = np.array(cc['base_matrix'], dtype=np.float32)
    if rng.random() < cc['perturbation_prob']:
        perturb = rng.uniform(-cc['perturbation_range'],
                              cc['perturbation_range'],
                              size=(3, 3)).astype(np.float32)
        M = M_base * (1.0 + perturb)
    else:
        M = M_base
    return M


def sample_noise_params(rng: np.random.Generator, cfg: dict) -> tuple:
    """采样 (beta_1, beta_2)。"""
    s = cfg['sensor']
    f_lo, f_hi = s['beta_sample_factor']
    beta_1 = rng.uniform(f_lo * s['beta_1_ref'], f_hi * s['beta_1_ref'])
    beta_2 = rng.uniform(f_lo * s['beta_2_ref'], f_hi * s['beta_2_ref'])
    return float(beta_1), float(beta_2)


def sample_virtual_camera(rng: np.random.Generator, cfg: dict) -> dict:
    """采样虚拟相机参数（基线无光学参数）。"""
    vc = {}
    vc['wb_gains'] = sample_wb_gains(rng, cfg)
    vc['ccm_matrix'] = sample_ccm_matrix(rng, cfg)
    vc['beta_1'], vc['beta_2'] = sample_noise_params(rng, cfg)
    return vc


# ---------------------------------------------------------------------------- #
# 传感器噪声（归一化 beta 域）
# ---------------------------------------------------------------------------- #

def apply_sensor_noise(I_linear: np.ndarray, beta_1: float, beta_2,
                       rng: np.random.Generator) -> np.ndarray:
    """主噪声实现（支持逐像素 beta_2）。

    Input:  I_linear [H,W,3] float32 [0,1]
            beta_1: float, 散粒噪声尺度
            beta_2: float 或 [H,W] float32, 读出噪声方差
            rng: np.random.Generator
    Output: I_noisy [H,W,3] float32 [0,1]
    """
    if beta_1 <= 0:
        raise ValueError("beta_1 must be positive")

    I_linear = np.clip(I_linear, 0.0, 1.0)
    lam = I_linear / beta_1  # [H,W,3]

    # large lambda guard
    if lam.max() >= 1e6:
        I_shot = rng.normal(lam, np.sqrt(np.maximum(lam, 1e-12))).astype(np.float32) * beta_1
    else:
        I_shot = rng.poisson(lam).astype(np.float32) * beta_1

    # 读出噪声（支持标量或逐像素 beta_2）
    if np.isscalar(beta_2):
        beta_2_std = np.sqrt(beta_2)
    else:
        beta_2_std = np.sqrt(beta_2).astype(np.float32)  # [H,W]

    # 标准正态 * std（避免对逐像素 std 调 normal 的 shape 问题）
    I_read = rng.normal(0.0, 1.0, I_linear.shape).astype(np.float32) * beta_2_std[..., None]

    I_noisy = I_shot + I_read
    return np.clip(I_noisy, 0.0, 1.0).astype(np.float32)


def apply_chroma_noise(I_linear: np.ndarray, rng: np.random.Generator,
                       cfg: dict) -> np.ndarray:
    """彩色噪声：主要影响色度，不显著改变亮度，保留原始 G 通道。"""
    cn = cfg['sensor']['chroma_noise']
    if not cn['enable'] or rng.random() >= cn['prob']:
        return I_linear

    sigma = rng.uniform(*cn['sigma_range'])
    Y = 0.299 * I_linear[..., 0] + 0.587 * I_linear[..., 1] + 0.114 * I_linear[..., 2]
    R_Y = I_linear[..., 0] - Y
    B_Y = I_linear[..., 2] - Y

    R_Y_noisy = R_Y + rng.normal(0.0, sigma, R_Y.shape).astype(np.float32)
    B_Y_noisy = B_Y + rng.normal(0.0, sigma, B_Y.shape).astype(np.float32)

    I_noisy = np.empty_like(I_linear)
    I_noisy[..., 0] = Y + R_Y_noisy
    I_noisy[..., 1] = I_linear[..., 1]  # 保留原始 G，不替换为 Y
    I_noisy[..., 2] = Y + B_Y_noisy

    return np.clip(I_noisy, 0.0, 1.0).astype(np.float32)


def apply_dark_region_noise_boost(I_linear: np.ndarray, beta_2: float,
                                  rng: np.random.Generator, cfg: dict):
    """暗部区域提升 beta_2，返回逐像素有效 beta_2 [H,W,1]。"""
    dr = cfg['sensor']['dark_region_noise_boost']
    if not dr['enable'] or rng.random() >= dr['prob']:
        return np.full((*I_linear.shape[:2], 1), beta_2, dtype=np.float32)

    threshold = dr['luminance_threshold']
    boost = rng.uniform(*dr['beta_2_boost_range'])
    lum = 0.299 * I_linear[..., 0] + 0.587 * I_linear[..., 1] + 0.114 * I_linear[..., 2]
    w_dark = np.clip((threshold - lum) / max(threshold, 1e-6), 0.0, 1.0)
    beta_2_eff = beta_2 * (1.0 + boost * w_dark[..., None])
    return beta_2_eff.astype(np.float32)  # [H,W,1]


def apply_noise_chain(I_linear: np.ndarray, vc: dict, rng: np.random.Generator,
                      cfg: dict) -> np.ndarray:
    """完整噪声链：主噪声 + 彩色噪声 + 暗部增强。"""
    beta_1 = vc['beta_1']
    beta_2 = vc['beta_2']

    # 暗部增强：返回逐像素 beta_2_eff [H,W,1]
    beta_2_eff = apply_dark_region_noise_boost(I_linear, beta_2, rng, cfg)
    # 广播为 [H,W,3] -> 直接传给 apply_sensor_noise 时 squeeze 为 [H,W]
    beta_2_per_pixel = beta_2_eff[..., 0]  # [H,W]

    I_noisy = apply_sensor_noise(I_linear, beta_1, beta_2_per_pixel, rng)

    if cfg['sensor']['chroma_noise']['enable']:
        I_noisy = apply_chroma_noise(I_noisy, rng, cfg)

    return I_noisy


# ---------------------------------------------------------------------------- #
# 正向 ISP（固定华为 P60 Pro 参考相机）
# ---------------------------------------------------------------------------- #

def apply_vignetting(I_srgb: np.ndarray, rng: np.random.Generator,
                     cfg: dict) -> np.ndarray:
    """渐晕效应。"""
    vg = cfg['forward_isp']['vignetting']
    if rng.random() >= vg['prob']:
        return I_srgb
    strength = rng.uniform(*vg['strength_range'])
    H, W, _ = I_srgb.shape
    cy, cx = H / 2.0, W / 2.0
    y, x = np.ogrid[:H, :W]
    r = np.sqrt(((y - cy) / cy) ** 2 + ((x - cx) / cx) ** 2)
    r = np.clip(r / r.max(), 0.0, 1.0)
    vignette = 1.0 - strength * (r ** 2)
    return np.clip(I_srgb * vignette[..., None], 0.0, 1.0).astype(np.float32)


def apply_sharpening(I_srgb: np.ndarray, rng: np.random.Generator,
                     cfg: dict) -> np.ndarray:
    """Unsharp Masking 锐化。"""
    sh = cfg['forward_isp']['sharpening']
    if rng.random() >= sh['prob']:
        return I_srgb
    amount = rng.uniform(*sh['amount_range'])
    ksize = sh['radius'] * 2 + 1
    I_blur = cv2.GaussianBlur(I_srgb, (ksize, ksize), 0.0)
    I_sharp = I_srgb + amount * (I_srgb - I_blur)
    return np.clip(I_sharp, 0.0, 1.0).astype(np.float32)


def apply_color_cast(I_srgb: np.ndarray, rng: np.random.Generator,
                     cfg: dict) -> np.ndarray:
    """偏色：sRGB 域加通道偏移。"""
    cc = cfg['forward_isp']['color_cast']
    if rng.random() >= cc['prob']:
        return I_srgb
    offsets = rng.uniform(*cc['offset_range'], size=3).astype(np.float32)
    return np.clip(I_srgb + offsets[None, None, :], 0.0, 1.0).astype(np.float32)


def forward_isp(I_noisy_raw: np.ndarray, vc: dict, rng: np.random.Generator,
                cfg: dict) -> np.ndarray:
    """完整正向 ISP。

    Input:  I_noisy_raw [H,W,3] float32 [0,1]
            vc: virtual camera params (含 wb_gains, ccm_matrix)
            rng: np.random.Generator
    Output: I_srgb [H,W,3] float32 [0,1] RGB
    """
    # 1. 正向 WB
    g_r, g_g, g_b = vc['wb_gains']
    I_wb = I_noisy_raw * np.array([g_r, g_g, g_b], dtype=np.float32)
    I_wb = np.clip(I_wb, 0.0, 1.0)

    # 2. 正向 CCM
    M_ccm = vc['ccm_matrix']
    I_flat = I_wb.reshape(-1, 3)
    I_lin_srgb = (I_flat @ M_ccm.T).reshape(I_wb.shape)
    I_lin_srgb = np.clip(I_lin_srgb, 0.0, 1.0)

    # 3. 正向 Tone Mapping (smoothstep)
    I_srgb = smoothstep_forward(I_lin_srgb)
    I_srgb = np.clip(I_srgb, 0.0, 1.0)

    # 4. 可选风格化模块
    fi = cfg['forward_isp']
    if fi['vignetting']['prob'] > 0:
        I_srgb = apply_vignetting(I_srgb, rng, cfg)
    if fi['sharpening']['prob'] > 0:
        I_srgb = apply_sharpening(I_srgb, rng, cfg)
    if fi['color_cast']['prob'] > 0:
        I_srgb = apply_color_cast(I_srgb, rng, cfg)

    return np.clip(I_srgb, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------- #
# 数码变焦与 JPEG 压缩
# ---------------------------------------------------------------------------- #

def double_jpeg(I: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Double-JPEG。要求: 不使用 np.roll; 不做 8x8 网格偏移; 内存内编解码, 不写磁盘。"""
    jp = cfg['jpeg']

    # 转 uint8 BGR for cv2
    I_uint8 = np.clip(I * 255.0, 0, 255).astype(np.uint8)
    I_bgr = cv2.cvtColor(I_uint8, cv2.COLOR_RGB2BGR)

    # First JPEG
    qf1 = int(rng.uniform(*jp['qf_range']))
    _, buf1 = cv2.imencode('.jpg', I_bgr, [cv2.IMWRITE_JPEG_QUALITY, qf1])
    I_jpg1 = cv2.imdecode(buf1, cv2.IMREAD_COLOR)

    if rng.random() < jp['double_prob']:
        # Second JPEG (lower QF, no grid offset)
        qf2_ratio = rng.uniform(*jp['qf2_ratio'])
        qf2 = max(int(qf1 * qf2_ratio), 10)
        _, buf2 = cv2.imencode('.jpg', I_jpg1, [cv2.IMWRITE_JPEG_QUALITY, qf2])
        I_final = cv2.imdecode(buf2, cv2.IMREAD_COLOR)
    else:
        I_final = I_jpg1

    I_final_rgb = cv2.cvtColor(I_final, cv2.COLOR_BGR2RGB)
    return (I_final_rgb.astype(np.float32) / 255.0)


def zoom_and_jpeg(I_srgb: np.ndarray, rng: np.random.Generator, cfg: dict,
                  blur_strength: float = 0.0) -> np.ndarray:
    """Down-Up resize 与 JPEG 压缩（合并实现）。

    Input:  I_srgb [H,W,3] float32 [0,1] RGB
            blur_strength: 前置光学模糊强度（基线固定 0.0）
    Output: I_lq [H,W,3] float32 [0,1] RGB
    """
    z = cfg['zoom']
    jp = cfg['jpeg']

    # 缩放区间由配置接线（避免硬编码 0.5-0.6 / 0.6-0.9）
    lo, hi = z['scale_factor_range']
    th = float(min(max(z['severe_threshold'], lo), hi))

    # 1. Down-Up Resize
    if rng.random() < z['prob']:
        # 决定是否使用 severe zoom
        if blur_strength < 0.1:
            severe = rng.random() < z['severe_prob_when_no_blur']
        else:
            severe = rng.random() < z['severe_prob']

        if severe:
            a = rng.uniform(lo, th)
        else:
            a = rng.uniform(th, hi)

        H, W = I_srgb.shape[:2]
        interp = rng.choice(np.array([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
        I = cv2.resize(I_srgb, (int(W * a), int(H * a)), interpolation=interp)
        I = cv2.resize(I, (W, H), interpolation=interp)
    else:
        I = I_srgb

    # 2. Double-JPEG
    if rng.random() < jp['prob']:
        I = double_jpeg(I, rng, cfg)

    return I.astype(np.float32)


# ---------------------------------------------------------------------------- #
# 运动模糊（Bresenham，支持任意角度）
# ---------------------------------------------------------------------------- #

def generate_motion_kernel(r: float, theta: float) -> np.ndarray:
    """生成 Bresenham 运动模糊核，归一化到 sum=1。

    Input:  r 长度（像素），theta 角度（度，theta=0 为垂直方向，逆时针为正）
    Output: [h,w] float32 核

    几何修复：以核中心为锚点（= filter2D 锚点），线段端点取"中心 ± 半位移"，
    避免端点从"边中点→角"导致的：
      1. 角度失真（跨 45° 处两个分支端点公式切换产生 ~37° 跳变）
      2. 有效长度仅 0.5–0.8 r
      3. 线段质心偏离核中心 -> 卷积产生净平移（破坏 LQ/GT 像素对齐）
    """
    from skimage.draw import line
    tr = np.deg2rad(theta)
    dr = r * np.cos(tr)          # 总位移（行方向）
    dc = r * np.sin(tr)          # 总位移（列方向）
    h = int(np.ceil(abs(dr))) + 1
    w = int(np.ceil(abs(dc))) + 1
    r0 = (h - 1) / 2.0           # 核中心（= filter2D 锚点）
    c0 = (w - 1) / 2.0

    rr, cc = line(int(round(r0 - dr / 2.0)), int(round(c0 - dc / 2.0)),
                  int(round(r0 + dr / 2.0)), int(round(c0 + dc / 2.0)))
    kernel = np.zeros((h, w), dtype=np.float32)
    kernel[rr, cc] = 1.0
    return kernel / max(kernel.sum(), 1e-12)


def sample_motion_blur_params(rng: np.random.Generator, cfg: dict,
                              legal_angles_table: dict = None) -> tuple:
    """Returns: (r, theta)"""
    mb = cfg['motion_blur']
    r = rng.uniform(*mb['length_range'])

    if r < mb['short_length_threshold']:
        if legal_angles_table is not None:
            r_int = int(round(r))
            if r_int in legal_angles_table:
                legal_angles = legal_angles_table[r_int]
                theta = legal_angles[rng.integers(0, len(legal_angles))]
            else:
                theta = int(rng.integers(0, 91))
        else:
            theta = int(rng.integers(0, 91))
    elif r < mb['full_angle_length']:
        max_angle = int(90 * (r - mb['short_length_threshold']) /
                        (mb['full_angle_length'] - mb['short_length_threshold']))
        theta = int(rng.integers(0, max_angle + 1))
    else:
        theta = int(rng.integers(0, 91))

    if rng.random() < 0.5:
        theta = -theta
    return r, theta


def precompute_legal_angles(output_path: str):
    """预计算合法角度表，输出 JSON（不用 pickle）。

    直接复用 generate_motion_kernel 的核生成逻辑做去重判定，保证与在线
    采样使用的核完全一致（避免两套端点逻辑不一致）。
    """
    table = {}
    for r in range(2, 101):
        unique_kernels = set()
        legal_angles = []
        for theta in range(0, 91):
            kernel = generate_motion_kernel(float(r), float(theta))
            key = kernel.tobytes()
            if key not in unique_kernels:
                unique_kernels.add(key)
                legal_angles.append(theta)
        table[str(r)] = legal_angles
    with open(output_path, 'w') as f:
        json.dump(table, f)


def apply_motion_blur(I_linear: np.ndarray, rng: np.random.Generator, cfg: dict,
                      legal_angles_table: dict = None) -> np.ndarray:
    """运动模糊应用（概率门控在调度级完成，本函数只负责渲染）。

    I_linear: [H,W,3] float32 [0,1]
    说明：调度器已判定本阶段启用，因此这里不再内部判 prob，避免 0.3×0.3=0.09
    的重复门控（见设计书 §9.5）。
    """
    r, theta = sample_motion_blur_params(rng, cfg, legal_angles_table)
    kernel = generate_motion_kernel(r, theta)
    # 多通道卷积（单通道核）
    I_blur = cv2.filter2D(I_linear, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return np.clip(I_blur, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------- #
# 眩光叠加（Flare）
# ---------------------------------------------------------------------------- #

class FlareTemplateBank:
    def __init__(self, flare_dir: str, cfg: dict):
        """flare_dir: path to flare_templates/512/"""
        self.flare_dir = flare_dir
        self.cfg = cfg
        self.template_files = sorted(glob.glob(f"{flare_dir}/*.npy"))
        if len(self.template_files) == 0:
            raise RuntimeError(f"No flare templates found in {flare_dir}")

    def sample(self, rng: np.random.Generator, target_size: int = 512) -> np.ndarray:
        """随机采样一个预缩放模板 [target_size,target_size,4]"""
        idx = int(rng.integers(0, len(self.template_files)))
        template = np.load(self.template_files[idx])  # [512,512,4] float32

        # 若尺寸不匹配（理论不应发生），缩放
        if template.shape[0] != target_size or template.shape[1] != target_size:
            template = cv2.resize(template, (target_size, target_size))

        return template.astype(np.float32)


def should_apply_flare(I: np.ndarray, rng: np.random.Generator, cfg: dict) -> bool:
    """触发门控。应在 sRGB 域（zoom_jpeg 之后）判定高光。

    - flare.prob: 全局是否尝试本阶段（接线，不再死配置）
    - gate.type=prob: 直接按 gate.value 概率
    - gate.type=highlight: 高光比例超过 gate.value 时按 highlight_prob，
      否则按 default_prob
    """
    fl = cfg['flare']
    if rng.random() >= fl['prob']:
        return False
    gate = fl['gate']
    if gate['type'] == 'prob':
        return rng.random() < gate['value']
    elif gate['type'] == 'highlight':
        highlight_ratio = float(np.mean(np.max(I, axis=-1) > 0.95))
        if highlight_ratio > gate['value']:
            return rng.random() < gate.get('highlight_prob', 0.8)
        else:
            return rng.random() < gate.get('default_prob', 0.2)
    return False


def apply_flare(I_bg: np.ndarray, flare_template: np.ndarray,
                rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """Screen 混合叠加眩光。"""
    intensity = rng.uniform(*cfg['flare']['intensity_range'])
    # 模板尺寸与背景不一致时缩放（保证非 512×512 输入也能叠加，避免广播崩溃）
    if (flare_template.shape[0] != I_bg.shape[0] or flare_template.shape[1] != I_bg.shape[1]):
        flare_template = cv2.resize(
            flare_template, (I_bg.shape[1], I_bg.shape[0]),
            interpolation=cv2.INTER_LINEAR)
    F_rgb = flare_template[..., :3] * intensity
    F_alpha = flare_template[..., 3:4]
    I_out = 1.0 - (1.0 - I_bg) * (1.0 - F_rgb * F_alpha)
    return np.clip(I_out, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------- #
# 链路调度器
# ---------------------------------------------------------------------------- #

def validate_constraints(constraints: list, all_stages: list):
    """检测约束图是否有环。"""
    graph = {s: [] for s in all_stages}
    for c in constraints:
        graph[c['from']].append(c['before'])
    visited = {s: 0 for s in all_stages}
    cycle = []

    def dfs(node):
        if visited[node] == 1:
            return True
        if visited[node] == 2:
            return False
        visited[node] = 1
        cycle.append(node)
        for nxt in graph[node]:
            if dfs(nxt):
                return True
        cycle.pop()
        visited[node] = 2
        return False

    for s in all_stages:
        if dfs(s):
            raise ValueError(f"Constraints form cycle: {' -> '.join(cycle)}")


def random_topological_sort(stages: list, constraints: list,
                            rng: np.random.Generator) -> list:
    graph = {s: [] for s in stages}
    in_degree = {s: 0 for s in stages}
    for c in constraints:
        graph[c['from']].append(c['before'])
        in_degree[c['before']] += 1
    queue = [s for s in stages if in_degree[s] == 0]
    result = []
    while queue:
        idx = int(rng.integers(0, len(queue)))
        node = queue.pop(idx)
        result.append(node)
        for nxt in graph[node]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
    if len(result) != len(stages):
        raise ValueError("Constraints form a cycle")
    return result


def pregenerate_sequences(stages: list, constraints: list, num_sequences: int,
                          rng: np.random.Generator) -> list:
    validate_constraints(constraints, stages)
    sequences = set()
    max_attempts = num_sequences * 10
    attempts = 0
    while len(sequences) < num_sequences and attempts < max_attempts:
        seq = random_topological_sort(stages, constraints, rng)
        sequences.add(tuple(seq))
        attempts += 1
    sequences = sorted((list(s) for s in sequences))
    if len(sequences) == 0:
        sequences = [DEFAULT_STAGES.copy()]
    return sequences


def _select_sequence(sequences: list, enabled_stages: list,
                     rng: np.random.Generator) -> list:
    """任取一条缓存序列，删除未启用的阶段（删元素保序，约束自动满足）。"""
    idx = int(rng.integers(0, len(sequences)))
    full_seq = sequences[idx]
    return [s for s in full_seq if s in enabled_stages]


# ---------------------------------------------------------------------------- #
# 主入口：SceneDegradation
# ---------------------------------------------------------------------------- #

def serialize_vc(vc: dict) -> dict:
    """序列化虚拟相机参数为可 JSON 化的 dict。"""
    return {
        'wb_gains': vc['wb_gains'].tolist(),
        'ccm_matrix': vc['ccm_matrix'].tolist(),
        'beta_1': vc['beta_1'],
        'beta_2': vc['beta_2'],
    }


class SceneDegradation:
    def __init__(self, cfg: dict, manifest_path: str, base_seed: int = 42,
                 linear_raw_loader=None):
        """cfg: config dict (configs/degradation_baseline.yaml)
        manifest_path: str, path to preprocessed/manifest.json
        base_seed: int, master RNG 种子（兜底初始化）
        linear_raw_loader: 可选 callable(name)->np.ndarray，用于从 zip 批次等
            非标准路径加载 linear_raw（配合 zip_store.ZipBatchStore）；默认 None
            时沿用 manifest['linear_raw'] 相对路径（传统松散模式）。
        """
        self.cfg = cfg
        self._manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        # master_rng 在 __init__ 中兜底初始化
        self.master_rng = np.random.default_rng(base_seed)
        self.manifest = PreprocessManager(manifest_path, cfg['preprocess_version'])
        self.linear_raw_loader = linear_raw_loader

        # Flare 模板库（路径从 manifest 读取，不硬编码）
        flare_bank_path = resolve_path(self._manifest_dir, self.manifest.manifest, 'flare_bank_path')
        if flare_bank_path is not None:
            self.flare_bank = FlareTemplateBank(flare_bank_path, cfg)
        else:
            self.flare_bank = None

        # 固定参考相机配置
        isp_cfg = load_yaml(os.path.join(_project_root(), 'configs/isp_huawei_p60pro.yaml'))
        # 合并到主 cfg（基线使用统一 cfg）
        self.cfg = deep_merge(self.cfg, isp_cfg)

        # 运动模糊合法角度表（JSON 格式）
        # 兼容配置中带 "preprocessed/..." 前缀或纯文件名的写法；
        # 路径基准统一为 manifest 所在目录（与 resolve_path 一致）。
        angles_path = cfg.get('motion_blur', {}).get(
            'legal_angles_path', 'preprocessed/motion_blur_legal_angles.json')
        if not os.path.isabs(angles_path):
            candidate = os.path.join(self._manifest_dir, angles_path)
            if not os.path.exists(candidate):
                # 回退：若配置形如 "preprocessed/xx.json"，去掉前缀后相对 manifest 目录
                candidate = os.path.join(self._manifest_dir, os.path.basename(angles_path))
            angles_path = candidate
        if os.path.exists(angles_path):
            with open(angles_path, 'r') as f:
                self.legal_angles_table = {int(k): v for k, v in json.load(f).items()}
        else:
            self.legal_angles_table = None  # fallback: 所有角度可用

        # 预生成调度序列
        self.sequences = pregenerate_sequences(
            DEFAULT_STAGES, cfg['scheduler']['constraints'],
            cfg['scheduler']['num_cached_sequences'],
            np.random.default_rng(cfg['scheduler']['seed'])
        )

    def set_rng(self, seed: int):
        """供 worker_init_fn 或外部调用。"""
        self.master_rng = np.random.default_rng(seed)

    def __call__(self, gt_name: str, master_rng: np.random.Generator = None,
                 source: str = 'gt', debug: bool = False) -> dict:
        """合成单个样本。

        Output: {'lq', 'gt', 'sample_params', 'debug'}
        """
        if master_rng is None:
            master_rng = self.master_rng

        sample_seed = int(master_rng.integers(0, 2**63))
        sample_rng = np.random.default_rng(sample_seed)

        record = self.manifest.get(gt_name)
        vc = sample_virtual_camera(sample_rng, self.cfg)

        if source == 'gt':
            lq, gt, params, debug_data = self._full_pipeline(record, vc, sample_rng, debug)
        elif source == 'flare3d_lq':
            lq, gt, params, debug_data = self._flare3d_pipeline(record, vc, sample_rng, debug)
        else:
            raise ValueError(f"Unknown source: {source}")

        params['seed'] = sample_seed
        return {
            'lq': lq,
            'gt': gt,  # 必须返回 GT
            'sample_params': params,
            'debug': debug_data if debug else None
        }

    def _full_pipeline(self, record: dict, vc: dict, rng: np.random.Generator,
                       debug: bool) -> tuple:
        """Returns: (lq, gt, params, debug_data)"""
        # 加载离线产物（支持自定义 loader，如 zip 批次读取）
        if self.linear_raw_loader is not None:
            I = self.linear_raw_loader(record['name'])
        else:
            linear_raw_path = resolve_path(self._manifest_dir, record, 'linear_raw')
            I = np.load(linear_raw_path)  # [H,W,3] float32 [0,1]
        if I.ndim != 3 or I.shape[2] != 3:
            raise ValueError(f"linear_raw must be [H,W,3], got shape {I.shape}")

        # GT 用于训练；gt_path 缺失时回退到线性域经 smoothstep 正向转到 sRGB 域，
        # 保证返回 GT 处于 sRGB 域（不违反输出契约）。
        gt_path = resolve_path(self._manifest_dir, record, 'gt_path')
        gt = load_srgb(gt_path) if gt_path is not None else smoothstep_forward(I).astype(np.float32)

        params = {'vc': serialize_vc(vc), 'stages_run': []}
        debug_data = {} if debug else None

        # 1. 决定每个阶段是否启用（概率门控在这里统一完成）
        enabled_stages = []
        if rng.random() < self.cfg['motion_blur']['prob']:
            enabled_stages.append('motion_blur')
        enabled_stages.append('sensor_noise')  # 必选
        enabled_stages.append('forward_isp')   # 必选
        enabled_stages.append('zoom_jpeg')     # 必选
        # flare 允许被调度；是否真正触发延迟到执行阶段在 sRGB 域上判定
        enabled_stages.append('flare')

        # 2. 从预生成序列中选择满足约束的子序列
        seq = _select_sequence(self.sequences, enabled_stages, rng)

        # 3. 按序执行；仅当阶段真正改变图像时才写入 stages_run / debug
        for stage in seq:
            applied = False
            if stage == 'motion_blur':
                I = apply_motion_blur(I, rng, self.cfg, self.legal_angles_table)
                applied = True
            elif stage == 'sensor_noise':
                I = apply_noise_chain(I, vc, rng, self.cfg)
                applied = True
            elif stage == 'forward_isp':
                I = forward_isp(I, vc, rng, self.cfg)
                applied = True
            elif stage == 'zoom_jpeg':
                I = zoom_and_jpeg(I, rng, self.cfg, blur_strength=0.0)
                applied = True
            elif stage == 'flare':
                # 在 sRGB 域（zoom_jpeg 之后）实时门控
                if self.flare_bank is not None and should_apply_flare(I, rng, self.cfg):
                    flare_template = self.flare_bank.sample(rng)
                    I = apply_flare(I, flare_template, rng, self.cfg)
                    applied = True
            if applied:
                params['stages_run'].append(stage)
                if debug:
                    debug_data[f'after_{stage}'] = I.copy()

        return I.astype(np.float32), gt.astype(np.float32), params, debug_data

    def _flare3d_pipeline(self, record: dict, vc: dict, rng: np.random.Generator,
                          debug: bool) -> tuple:
        """Returns: (lq, gt, params, debug_data)"""
        input_path = resolve_path(self._manifest_dir, record, 'flare3d_input_path')
        gt_path = resolve_path(self._manifest_dir, record, 'flare3d_gt_path')
        if input_path is None or gt_path is None:
            raise ValueError(
                f"flare3d record '{record.get('name')}' lacks flare3d_input_path "
                f"or flare3d_gt_path")

        I_srgb = load_srgb(input_path)
        gt = load_srgb(gt_path)

        params = {'vc': serialize_vc(vc), 'stages_run': []}
        debug_data = {} if debug else None

        # 1. 轻量逆 ISP (仅反 smoothstep)
        I_linear = smoothstep_inverse(I_srgb)
        I_linear = np.clip(I_linear, 0.0, 1.0)
        params['stages_run'].append('inverse_isp_lightweight')
        if debug:
            debug_data['after_inverse_isp_lightweight'] = I_linear.copy()

        # 2. 传感器噪声
        I = apply_noise_chain(I_linear, vc, rng, self.cfg)
        params['stages_run'].append('sensor_noise')
        if debug:
            debug_data['after_sensor_noise'] = I.copy()

        # 3. 轻量正向 ISP (仅 smoothstep)
        I = smoothstep_forward(I)
        I = np.clip(I, 0.0, 1.0)
        params['stages_run'].append('forward_isp_lightweight')
        if debug:
            debug_data['after_forward_isp_lightweight'] = I.copy()

        # 4. zoom_jpeg
        I = zoom_and_jpeg(I, rng, self.cfg, blur_strength=0.0)
        params['stages_run'].append('zoom_jpeg')
        if debug:
            debug_data['after_zoom_jpeg'] = I.copy()

        return I.astype(np.float32), gt.astype(np.float32), params, debug_data


# ---------------------------------------------------------------------------- #
# worker_init_fn（多进程 DataLoader）
# ---------------------------------------------------------------------------- #

def worker_init_fn(worker_id: int):
    """每个 worker 的初始化（num_workers>0 时由 PyTorch 调用）。

    说明：
    - 若 worker.dataset 不是 SceneDegradation（或已包装），静默跳过，避免崩溃；
    - flare_bank_path 为 None 时置 flare_bank=None，不再用 None 构造 FlareTemplateBank。
    """
    import torch
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    worker = info.dataset
    if not hasattr(worker, 'set_rng'):
        return  # 被训练 Dataset 包装时，无法直接初始化 SceneDegradation
    # info.seed 已含 worker 区分，直接使用
    worker.set_rng(info.seed)
    flare_bank_path = resolve_path(worker._manifest_dir, worker.manifest.manifest, 'flare_bank_path')
    worker.flare_bank = FlareTemplateBank(flare_bank_path, worker.cfg) if flare_bank_path is not None else None
