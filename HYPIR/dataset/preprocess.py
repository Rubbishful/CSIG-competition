# -*- coding: utf-8 -*-
"""基线离线预处理入口。

职责：
- 生成 linear_raw/{name}.npy (固定参考相机逆 ISP)
- 生成 preprocessed/manifest.json
- 生成运动模糊合法角度表 JSON
- 生成 flare_templates/512/{id}.npy (当缺失时)

用法示例：
    from HYPIR.dataset.preprocess import run_preprocess
    run_preprocess(gt_dir="preprocessed/gt", output_dir="preprocessed", cfg=cfg)
"""

import glob
import json
import os
from pathlib import Path

import numpy as np

from HYPIR.dataset.scene_degradation import (
    inverse_isp,
    load_srgb,
    load_yaml,
    precompute_legal_angles,
    _project_root,
)


def run_preprocess(gt_dir: str, output_dir: str, cfg: dict):
    """基线离线预处理主入口。

    gt_dir: 原始 GT 输入目录（*.png / *.jpg）
    output_dir: 输出目录（linear_raw、manifest.json 将写于此）
    cfg: config dict (configs/degradation_baseline.yaml)
    """
    os.makedirs(os.path.join(output_dir, 'linear_raw'), exist_ok=True)

    # 全局产物：运动模糊合法角度表
    angles_path = os.path.join(output_dir, 'motion_blur_legal_angles.json')
    if not os.path.exists(angles_path):
        print(f"[preprocess] precompute legal angles -> {angles_path}")
        precompute_legal_angles(angles_path)

    # 固定参考相机配置
    isp_cfg = load_yaml(os.path.join(_project_root(), 'configs/isp_huawei_p60pro.yaml'))
    M_ccm = np.array(isp_cfg['forward_ccm'], dtype=np.float32)

    # 离线用固定基准（无扰动）——从配置取增益区间中点，避免硬编码失配；
    # 与在线 sample_wb_gains 的基准来源一致，保证离线/在线 WB 一致。
    wb_cfg = isp_cfg['forward_isp']['white_balance']
    g_r_fixed = float(np.mean(wb_cfg['r_gain_range']))
    g_b_fixed = float(np.mean(wb_cfg['b_gain_range']))
    vc_fixed = {
        'ccm_matrix': M_ccm,
        'wb_gains': np.array([g_r_fixed, wb_cfg['g_gain'], g_b_fixed], dtype=np.float32),
    }

    # 支持 png / jpg
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.png')) +
                      glob.glob(os.path.join(gt_dir, '*.jpg')) +
                      glob.glob(os.path.join(gt_dir, '*.jpeg')))

    for gt_path in gt_files:
        name = Path(gt_path).stem
        linear_raw_path = os.path.join(output_dir, 'linear_raw', f'{name}.npy')

        # 断点续跑：跳过已存在
        if os.path.exists(linear_raw_path):
            continue

        I_srgb = load_srgb(gt_path)
        I_linear = inverse_isp(I_srgb, vc_fixed, cfg)
        np.save(linear_raw_path, I_linear)

    # 生成 manifest
    generate_manifest(gt_dir, output_dir, cfg['preprocess_version'])
    print(f"[preprocess] done. manifest -> {os.path.join(output_dir, 'manifest.json')}")


def generate_manifest(gt_dir: str, output_dir: str, version: str):
    """生成 manifest.json。"""
    files = []
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.png')) +
                      glob.glob(os.path.join(gt_dir, '*.jpg')) +
                      glob.glob(os.path.join(gt_dir, '*.jpeg')))
    for gt_path in gt_files:
        name = Path(gt_path).stem
        # 相对路径以 manifest 所在目录（output_dir）为基准，与 resolve_path 一致；
        # 用正斜杠保证跨平台（Windows 写入时不做 os.path.join，避免反斜杠）。
        linear_raw_rel = f"linear_raw/{name}.npy"
        files.append({
            'name': name,
            'source_type': 'gt',
            'gt_path': os.path.abspath(gt_path),
            'flare3d_input_path': None,
            'flare3d_gt_path': None,
            'linear_raw': linear_raw_rel,
        })

    manifest = {
        'preprocess_version': version,
        'manifest_dir': '.',
        'files': files,
        'psf_bank_version': None,
        'psf_bank_path': None,
        'flare_bank_path': 'flare_templates/512/',
    }
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)


def generate_flare_templates(output_dir: str, num_templates: int = 20, size: int = 512):
    """从 FlareX 数据集转换，或生成简单光晕模板。"""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    for i in range(num_templates):
        template = np.zeros((size, size, 4), dtype=np.float32)

        # 简单方案：随机位置的高斯光斑 + 星芒
        cx = int(rng.integers(size // 4, 3 * size // 4))
        cy = int(rng.integers(size // 4, 3 * size // 4))

        # 主光斑
        y, x = np.ogrid[:size, :size]
        sigma_main = rng.uniform(5, 30)
        main_bloom = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma_main ** 2))

        # 星芒（3 条十字线 -> 6 个方向）
        for angle_deg in [0, 60, 120]:
            angle = np.deg2rad(angle_deg)
            dx, dy = np.cos(angle), np.sin(angle)
            line_dist = np.abs((y - cy) * dx - (x - cx) * dy)
            line_along = (y - cy) * dy + (x - cx) * dx
            streak = np.exp(-line_dist ** 2 / (2 * 3 ** 2)) * np.exp(-line_along ** 2 / (2 * (size // 3) ** 2))
            main_bloom = np.maximum(main_bloom, streak * 0.5)

        # 颜色：暖色调
        template[..., 0] = np.clip(main_bloom, 0, 1)  # R
        template[..., 1] = np.clip(main_bloom * 0.9, 0, 1)  # G
        template[..., 2] = np.clip(main_bloom * 0.7, 0, 1)  # B
        template[..., 3] = np.clip(main_bloom, 0, 1)  # Alpha

        np.save(os.path.join(output_dir, f'{i:04d}.npy'), template)


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='HYPIR 基线离线预处理')
    parser.add_argument('--gt_dir', required=True, help='原始 GT 输入目录')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--config', default=None,
                        help='主配置路径（默认 configs/degradation_baseline.yaml）')
    parser.add_argument('--num_templates', type=int, default=20,
                        help='生成 flare 模板数量')
    args = parser.parse_args()

    cfg_path = args.config or os.path.join(_project_root(), 'configs/degradation_baseline.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    run_preprocess(args.gt_dir, args.output_dir, cfg)

    # 若无 flare 模板，生成占位模板
    flare_dir = os.path.join(args.output_dir, 'flare_templates', '512')
    if not glob.glob(os.path.join(flare_dir, '*.npy')):
        print(f"[preprocess] generating {args.num_templates} flare templates -> {flare_dir}")
        generate_flare_templates(flare_dir, args.num_templates)
