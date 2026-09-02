# -*- coding: utf-8 -*-
"""基线离线预处理入口（含 zip 批次输出模式）。

职责：
- 生成 linear_raw/{name}.npy (固定参考相机逆 ISP)
- 默认按 zip_batch_size 分批聚合成 batches/batch_XXXXX.zip（每 zip 含 linear_raw/{name}.npy，
  批次按 manifest 打乱顺序切分；npy 先写本地 staging，zip 完成后即删，避免小文件直接入 OSS）
- 生成 preprocessed/manifest.json（zip 模式下每条记录含 'zip' 字段）
- 生成运动模糊合法角度表 JSON
- 生成 flare_templates/512/{id}.npy (当缺失时)

zip 模式设计（OSS 场景）：
  - output_dir 可以就是 OSS 挂载路径（只写入少量 zip 大文件，规避小文件写入慢/贵）；
  - staging_dir 必须在本地系统盘（默认系统临时目录），峰值占用约
    (zip_workers + 2) * batch_size * 3.2MB（float32 npy）；
  - 训练侧统一用 HYPIR.dataset.zip_store.ZipBatchStore 读取，与产物严格对应。

用法示例：
    # zip 批次模式（默认 zip_batch_size=1024，输出到 OSS 挂载目录）
    python -m HYPIR.dataset.preprocess --gt_dir <merged>/gt --output_dir /mnt/data/dataset \
        --zip-batch-size 1024 --workers 8

    # 传统松散模式（不打包，行为与旧版一致）
    python -m HYPIR.dataset.preprocess --gt_dir <gt> --output_dir <out> --zip-batch-size 0

    from HYPIR.dataset.preprocess import run_preprocess
    run_preprocess(gt_dir="preprocessed/gt", output_dir="preprocessed", cfg=cfg)  # 默认旧行为
"""

import glob
import json
import os
import shutil
import tempfile
import time
import zipfile
from collections import deque
from pathlib import Path

import numpy as np

from HYPIR.dataset.scene_degradation import (
    inverse_isp,
    load_srgb,
    load_yaml,
    precompute_legal_angles,
    _project_root,
)

# 每个 float32 512x512x3 npy 的近似大小（MB），用于 staging 峰值估算提示
_NPY_MB = 512 * 512 * 3 * 4 / 1e6


def _process_one(args):
    """worker：单张 GT 逆 ISP 并保存 linear_raw。args: (gt_path, linear_raw_path, vc_fixed, cfg)"""
    gt_path, linear_raw_path, vc_fixed, cfg = args
    try:
        I_srgb = load_srgb(gt_path)
        I_linear = inverse_isp(I_srgb, vc_fixed, cfg)
        np.save(linear_raw_path, I_linear)
        return (gt_path, None)
    except Exception as exc:
        return (gt_path, f"{type(exc).__name__}: {exc}")


def _make_pool(workers: int):
    """进程池（受限环境自动回退线程池）。"""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    try:
        return ProcessPoolExecutor(max_workers=workers)
    except (PermissionError, OSError, RuntimeError) as exc:
        print(f"⚠ 进程池不可用（{type(exc).__name__}: {exc}），回退线程池")
        return ThreadPoolExecutor(max_workers=workers)


def _run_legacy(gt_files, output_dir, vc_fixed, cfg, workers):
    """传统松散模式：linear_raw/*.npy 直接落盘（默认 workers=1 行为不变）。"""
    todo = []
    for gt_path in gt_files:
        name = Path(gt_path).stem
        linear_raw_path = os.path.join(output_dir, 'linear_raw', f'{name}.npy')
        if os.path.exists(linear_raw_path):
            continue
        todo.append((gt_path, linear_raw_path))

    failures = []
    if workers <= 1:
        for gt_path, linear_raw_path in todo:
            err = _process_one((gt_path, linear_raw_path, vc_fixed, cfg))[1]
            if err is not None:
                failures.append((gt_path, err))
                print(f"[preprocess] 处理 {Path(gt_path).name} 失败: {err}")
    else:
        jobs = [(gt_path, linear_raw_path, vc_fixed, cfg) for gt_path, linear_raw_path in todo]
        print(f"[preprocess] 并行预处理 {len(jobs)} 张（workers={workers}）...")
        with _make_pool(workers) as pool:
            for i, (gt_path, err) in enumerate(pool.map(_process_one, jobs), 1):
                if err is not None:
                    failures.append((gt_path, err))
                    print(f"[preprocess] 处理 {Path(gt_path).name} 失败: {err}")
                if i % 20000 == 0:
                    print(f"[preprocess]   {i}/{len(jobs)} ...")
    return failures


def _zip_dir(stage_dir, zip_path, compresslevel):
    """把 staging 目录下的 *.npy 打成 zip（arcname 前缀 linear_raw/）。返回 (zip_path, size)。"""
    tmp_path = zip_path + '.tmp'
    try:
        with zipfile.ZipFile(tmp_path, 'w') as zf:
            for name in sorted(os.listdir(stage_dir)):
                if not name.endswith('.npy'):
                    continue
                zf.write(os.path.join(stage_dir, name), arcname=f'linear_raw/{name}')
        os.replace(tmp_path, zip_path)
        return (zip_path, os.path.getsize(zip_path))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _run_zip_batches(gt_files, output_dir, vc_fixed, cfg, batch_size, workers,
                     staging_root, zip_compress, zip_workers):
    """zip 批次模式：分批本地聚合，每批一个 zip（npy 生产与 zip 打包流水线并行）。"""
    from concurrent.futures import ThreadPoolExecutor

    batches_dir = os.path.join(output_dir, 'batches')
    os.makedirs(batches_dir, exist_ok=True)
    os.makedirs(staging_root, exist_ok=True)

    # 断点续跑：已存在的 zip 记录其内容
    zip_map = {}  # stem -> 'batches/batch_XXXXX.zip'
    existing_zip_paths = set()
    for zp in sorted(glob.glob(os.path.join(batches_dir, 'batch_*.zip'))):
        try:
            with zipfile.ZipFile(zp) as zf:
                for n in zf.namelist():
                    if n.endswith('.npy'):
                        zip_map[Path(n).stem] = os.path.relpath(zp, output_dir).replace('\\', '/')
            existing_zip_paths.add(zp)
        except zipfile.BadZipFile:
            print(f"⚠ 跳过损坏 zip: {zp}")

    chunks = [gt_files[i:i + batch_size] for i in range(0, len(gt_files), batch_size)]
    print(f"[preprocess] zip 批次模式: {len(gt_files)} 张 -> {len(chunks)} 批"
          f"（batch_size={batch_size}, staging={staging_root}, 约需 "
          f"{(zip_workers + 2) * batch_size * _NPY_MB / 1024:.1f} GB staging 空间）")

    failures = []
    npy_pool = _make_pool(workers)
    zip_pool = ThreadPoolExecutor(max_workers=max(1, zip_workers))
    pending = deque()  # (future, stage_dir)
    t0 = time.time()
    try:
        for k, chunk in enumerate(chunks):
            batch_id = f'batch_{k:05d}'
            zip_path = os.path.join(batches_dir, batch_id + '.zip')
            stage_dir = os.path.join(staging_root, batch_id)
            zip_rel = f'batches/{batch_id}.zip'

            # 断点续跑：该批 zip 已存在
            if zip_path in existing_zip_paths:
                print(f"[preprocess] 跳过已存在批次 {batch_id}（{len(chunk)} 张）")
                continue

            # 限制 pending zips 数量以约束 staging 峰值
            while len(pending) >= max(1, zip_workers):
                fut, sd = pending.popleft()
                fut.result()
                shutil.rmtree(sd, ignore_errors=True)

            os.makedirs(stage_dir, exist_ok=True)
            # 1) 并行逆 ISP 写 staging
            jobs = [(gt_path, os.path.join(stage_dir, Path(gt_path).stem + '.npy'),
                     vc_fixed, cfg) for gt_path in chunk]
            failed_stems = set()
            for gt_path, err in npy_pool.map(_process_one, jobs):
                if err is not None:
                    failures.append((gt_path, err))
                    failed_stems.add(Path(gt_path).stem)

            # 2) 提交 zip 打包（与下一批 npy 生产并行）
            if len(chunk) == len(failed_stems):
                shutil.rmtree(stage_dir, ignore_errors=True)
                continue
            pending.append((zip_pool.submit(_zip_dir, stage_dir, zip_path, zip_compress),
                            stage_dir))
            for gt_path in chunk:
                stem = Path(gt_path).stem
                if stem not in failed_stems:
                    zip_map[stem] = zip_rel

            print(f"[preprocess] 批次 {batch_id}: {len(chunk) - len(failed_stems)}/{len(chunk)} 张已入队")

        # 收尾：等待剩余 zip
        while pending:
            fut, sd = pending.popleft()
            fut.result()
            shutil.rmtree(sd, ignore_errors=True)
        print(f"[preprocess] zip 批次完成: {len(zip_map)} 张 -> "
              f"{len(glob.glob(os.path.join(batches_dir, 'batch_*.zip')))} 个 zip"
              f"（耗时 {time.time() - t0:.0f}s）")
    finally:
        npy_pool.shutdown(wait=True)
        zip_pool.shutdown(wait=True)
    return failures, zip_map


def run_preprocess(gt_dir: str, output_dir: str, cfg: dict, workers: int = 1,
                   zip_batch_size: int = 0, staging_dir: str = None,
                   zip_compress: int = 1, zip_workers: int = 2):
    """基线离线预处理主入口。

    gt_dir: 原始 GT 输入目录（*.png / *.jpg）
    output_dir: 输出目录（linear_raw 或 batches/*.zip、manifest.json 将写于此）
    cfg: config dict (configs/degradation_baseline.yaml)
    workers: 并行进程数（1=串行，保持原行为；>1 并行加速，断点续跑/产物一致）
    zip_batch_size: >0 时启用 zip 批次模式（每批一个 zip，按 manifest 打乱顺序切分）；
                    0 = 传统松散模式（linear_raw/*.npy 直接落盘，兼容旧行为）
    staging_dir: zip 模式下 npy 暂存目录（必须在本地系统盘；默认系统临时目录）
    zip_compress: zip 压缩级别 0-9（0=不压缩，速度最快；默认 1）
    zip_workers: zip 打包线程数（默认 2；staging 峰值 ≈ (zip_workers+2)*batch*3.2MB）
    """
    os.makedirs(os.path.join(output_dir, 'linear_raw'), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # 全局产物：运动模糊合法角度表
    angles_path = os.path.join(output_dir, 'motion_blur_legal_angles.json')
    if not os.path.exists(angles_path):
        print(f"[preprocess] precompute legal angles -> {angles_path}")
        precompute_legal_angles(angles_path)

    # 固定参考相机配置
    isp_cfg = load_yaml(os.path.join(_project_root(), 'configs/isp_huawei_p60pro.yaml'))
    M_ccm = np.array(isp_cfg['forward_ccm'], dtype=np.float32)

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

    zip_map = None
    if zip_batch_size and zip_batch_size > 0:
        staging = staging_dir or os.path.join(tempfile.gettempdir(), 'hypir_preprocess_staging')
        failures, zip_map = _run_zip_batches(gt_files, output_dir, vc_fixed, cfg,
                                             batch_size=int(zip_batch_size), workers=workers,
                                             staging_root=staging, zip_compress=zip_compress,
                                             zip_workers=zip_workers)
    else:
        # 传统松散模式
        failures = _run_legacy(gt_files, output_dir, vc_fixed, cfg, workers)

    if failures:
        print(f"[preprocess] ⚠ {len(failures)} 张处理失败（均已跳过 manifest）：")
        for fp, err in failures[:10]:
            print(f"    {Path(fp).name}: {err}")

    # 生成 manifest
    generate_manifest(gt_dir, output_dir, cfg['preprocess_version'], zip_map=zip_map)
    print(f"[preprocess] done. manifest -> {os.path.join(output_dir, 'manifest.json')}")


def generate_manifest(gt_dir: str, output_dir: str, version: str, zip_map: dict = None):
    """生成 manifest.json。

    zip_map: None 时为传统松散模式（沿用旧字段）；
             否则 {stem: 'batches/batch_XXXXX.zip'}，每条记录追加 'zip' 字段，
             且仅收录 zip_map 中包含的成功样本。
    """
    files = []
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.png')) +
                      glob.glob(os.path.join(gt_dir, '*.jpg')) +
                      glob.glob(os.path.join(gt_dir, '*.jpeg')))
    for gt_path in gt_files:
        name = Path(gt_path).stem
        if zip_map is not None:
            zip_rel = zip_map.get(name)
            if zip_rel is None:
                continue  # 失败样本不入 manifest
        else:
            zip_rel = None
        # 相对路径以 manifest 所在目录（output_dir）为基准，与 resolve_path 一致；
        # 用正斜杠保证跨平台（Windows 写入时不做 os.path.join，避免反斜杠）。
        linear_raw_rel = f"linear_raw/{name}.npy"
        record = {
            'name': name,
            'source_type': 'gt',
            'gt_path': os.path.abspath(gt_path),
            'flare3d_input_path': None,
            'flare3d_gt_path': None,
            'linear_raw': linear_raw_rel,
        }
        if zip_rel is not None:
            record['zip'] = zip_rel
        files.append(record)

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

    parser = argparse.ArgumentParser(
        description='HYPIR 基线离线预处理（默认 zip 批次输出；--zip-batch-size 0 恢复松散模式）')
    parser.add_argument('--gt_dir', required=True, help='原始 GT 输入目录')
    parser.add_argument('--output_dir', required=True,
                        help='输出目录（zip 模式可直接指向 OSS 挂载路径；'
                             '传统模式为本地目录）')
    parser.add_argument('--config', default=None,
                        help='主配置路径（默认 configs/degradation_baseline.yaml）')
    parser.add_argument('--num_templates', type=int, default=20,
                        help='生成 flare 模板数量')
    parser.add_argument('--workers', type=int, default=1,
                        help='并行进程数（默认 1=串行；建议设备核心数-1，如 8）')
    parser.add_argument('--zip-batch-size', type=int, default=1024,
                        help='zip 批次大小：每批样本数生成一个 batch_XXXXX.zip（默认 1024；'
                             '0=传统松散模式不打包）')
    parser.add_argument('--staging-dir', default=None,
                        help='zip 模式 npy 暂存目录（必须在本地系统盘；默认系统临时目录）')
    parser.add_argument('--zip-compress', type=int, default=1, choices=range(0, 10),
                        help='zip 压缩级别 0-9（默认 1；0=不压缩最快）')
    parser.add_argument('--zip-workers', type=int, default=2,
                        help='zip 打包线程数（默认 2；staging 峰值≈(zip_workers+2)*batch*3.2MB）')
    args = parser.parse_args()

    cfg_path = args.config or os.path.join(_project_root(), 'configs/degradation_baseline.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    run_preprocess(args.gt_dir, args.output_dir, cfg, workers=args.workers,
                   zip_batch_size=args.zip_batch_size, staging_dir=args.staging_dir,
                   zip_compress=args.zip_compress, zip_workers=args.zip_workers)

    # 若无 flare 模板，生成占位模板
    flare_dir = os.path.join(args.output_dir, 'flare_templates', '512')
    if not glob.glob(os.path.join(flare_dir, '*.npy')):
        print(f"[preprocess] generating {args.num_templates} flare templates -> {flare_dir}")
        generate_flare_templates(flare_dir, args.num_templates)
