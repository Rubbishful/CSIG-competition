# -*- coding: utf-8 -*-
"""基线离线预处理入口（zip 批次输出 + zip 源读取）。

职责：
- 生成 linear_raw/{name}.npy (固定参考相机逆 ISP)
- gt_dir 支持两种源：普通目录，或未解压的 merged_512.zip（zip 源模式，直接读条目，
  避免在系统盘解压 62GB）
- 默认按 zip_batch_size 分批聚合成 batches/batch_XXXXX.zip（每 zip 含 linear_raw/{name}.npy，
  批次按 manifest 打乱顺序切分；npy 先写本地 staging，zip 完成后即删，避免小文件直接入 OSS）
- 生成 preprocessed/manifest.json（zip 批次模式下每条记录含 'zip' 字段；
  zip 源模式下另有 gt_zip/gt_arc 字段，gt_path=None）
- 生成运动模糊合法角度表 JSON
- 生成 flare_templates/512/{id}.npy (当缺失时)

zip 模式设计（OSS 场景）：
  - output_dir 可以就是 OSS 挂载路径（只写入少量 zip 大文件，规避小文件写入慢/贵）；
  - staging_dir 必须在本地系统盘（默认系统临时目录），峰值占用约
    (zip_workers + 2) * batch_size * 3.2MB（float32 npy）；
  - 训练侧统一用 HYPIR.dataset.zip_store.ZipBatchStore 读取，与产物严格对应。

用法示例：
    # zip 源 + zip 批次输出（实例上，gt 直接读 OSS 上的 merged_512.zip）
    python -m HYPIR.dataset.preprocess --gt_dir /mnt/data/merged_512.zip \
        --output_dir /mnt/data/dataset --zip-batch-size 512 --workers 8

    # 目录源 + zip 批次输出
    python -m HYPIR.dataset.preprocess --gt_dir <merged>/gt --output_dir /mnt/data/dataset \
        --zip-batch-size 512 --workers 8

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
    load_srgb_bytes,
    load_yaml,
    precompute_legal_angles,
    _project_root,
)

# 每个 float32 512x512x3 npy 的近似大小（MB），用于 staging 峰值估算提示
_NPY_MB = 512 * 512 * 3 * 4 / 1e6

_ZIP_PREFIX = 'zip://'


def _item_stem(item: str) -> str:
    """GT 条目名（普通路径或 zip://<zip>!/<entry>）提取 stem。"""
    return Path(item.split('!/')[-1]).stem


# ---------------------------------------------------------------------------- #
# GT 源读取（目录 / zip 通用）
# ---------------------------------------------------------------------------- #

_ZIP_CACHE = {}  # worker 进程内常驻 zip 句柄（ProcessPool 模式）


def _get_zip(zip_path: str, use_cache: bool):
    if not use_cache:
        return zipfile.ZipFile(zip_path)  # 线程池回退：每次独立打开，避免共享句柄竞态
    zf = _ZIP_CACHE.get(zip_path)
    if zf is None:
        zf = zipfile.ZipFile(zip_path)
        _ZIP_CACHE[zip_path] = zf
    return zf


def _load_gt(item: str, zip_cache: bool) -> np.ndarray:
    if item.startswith(_ZIP_PREFIX):
        zip_path, entry = item[len(_ZIP_PREFIX):].split('!/', 1)
        buf = _get_zip(zip_path, zip_cache).read(entry)
        return load_srgb_bytes(buf)
    return load_srgb(item)


def _process_one(args):
    """worker：单张 GT 逆 ISP 并保存 linear_raw。
    args: (item, linear_raw_path, vc_fixed, cfg, zip_cache)
    """
    item, linear_raw_path, vc_fixed, cfg, zip_cache = args
    try:
        I_srgb = _load_gt(item, zip_cache)
        I_linear = inverse_isp(I_srgb, vc_fixed, cfg)
        np.save(linear_raw_path, I_linear)
        return (item, None)
    except Exception as exc:
        return (item, f"{type(exc).__name__}: {exc}")


def _make_pool(workers: int):
    """进程池（受限环境自动回退线程池）。"""
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    try:
        return ProcessPoolExecutor(max_workers=workers)
    except (PermissionError, OSError, RuntimeError) as exc:
        print(f"⚠ 进程池不可用（{type(exc).__name__}: {exc}），回退线程池")
        return ThreadPoolExecutor(max_workers=workers)


def _zip_cache_ok(pool) -> bool:
    """进程池模式才能安全共享常驻 zip 句柄（每个 worker 进程一份）。"""
    return type(pool).__name__ == 'ProcessPoolExecutor'


# ---------------------------------------------------------------------------- #
# 传统松散模式 / zip 批次模式
# ---------------------------------------------------------------------------- #

def _run_legacy(items, output_dir, vc_fixed, cfg, workers):
    """传统松散模式：linear_raw/*.npy 直接落盘（默认 workers=1 行为不变）。"""
    todo = []
    for item in items:
        stem = _item_stem(item)
        linear_raw_path = os.path.join(output_dir, 'linear_raw', f'{stem}.npy')
        if os.path.exists(linear_raw_path):
            continue
        todo.append(item)

    failures = []
    if workers <= 1:
        for item in todo:
            linear_raw_path = os.path.join(output_dir, 'linear_raw',
                                           f'{_item_stem(item)}.npy')
            err = _process_one((item, linear_raw_path, vc_fixed, cfg, False))[1]
            if err is not None:
                failures.append((item, err))
                print(f"[preprocess] 处理 {_item_stem(item)} 失败: {err}")
    else:
        pool = _make_pool(workers)
        zip_cache = _zip_cache_ok(pool)
        jobs = [(item, os.path.join(output_dir, 'linear_raw', f'{_item_stem(item)}.npy'),
                 vc_fixed, cfg, zip_cache) for item in todo]
        print(f"[preprocess] 并行预处理 {len(jobs)} 张（workers={workers}）...")
        if pool is not None:
            try:
                with pool:
                    for i, (item, err) in enumerate(pool.map(_process_one, jobs), 1):
                        if err is not None:
                            failures.append((item, err))
                            print(f"[preprocess] 处理 {_item_stem(item)} 失败: {err}")
                        if i % 20000 == 0:
                            print(f"[preprocess]   {i}/{len(jobs)} ...")
            finally:
                pool.shutdown(wait=True)
    return failures


def _zip_dir(stage_dir, zip_path, compresslevel, work_dir):
    """把 staging 目录下的 *.npy 打成 zip（arcname 前缀 linear_raw/）。

    zip 必须在本地 work_dir（可 seek）构建，再 copyfile 到 zip_path：
    output_dir 可能是 OSS 挂载（fuse 不支持 zipfile 构建所需的 seek-back，
    直接写会抛 OSError: [Errno 22] Invalid argument）。返回 (zip_path, size)。
    """
    local_zip = os.path.join(work_dir, os.path.basename(zip_path))
    try:
        with zipfile.ZipFile(local_zip, 'w') as zf:
            for name in sorted(os.listdir(stage_dir)):
                if not name.endswith('.npy'):
                    continue
                zf.write(os.path.join(stage_dir, name), arcname=f'linear_raw/{name}')
        size = os.path.getsize(local_zip)
        shutil.copyfile(local_zip, zip_path)  # 跨设备复制（OSS 为 copyfile/写，非 rename）
        return (zip_path, size)
    finally:
        if os.path.exists(local_zip):
            os.remove(local_zip)


def _run_zip_batches(items, output_dir, vc_fixed, cfg, batch_size, workers,
                     staging_root, zip_compress, zip_workers):
    """zip 批次模式：分批本地聚合，每批一个 zip（npy 生产与 zip 打包流水线并行）。"""
    from concurrent.futures import ThreadPoolExecutor

    batches_dir = os.path.join(output_dir, 'batches')
    os.makedirs(batches_dir, exist_ok=True)
    os.makedirs(staging_root, exist_ok=True)

    # 断点续跑：用侧车索引记录 name->zip（避免打开 OSS 上的 zip 做中间目录 seek 读）
    index_path = os.path.join(batches_dir, '_zip_index.json')
    zip_map = {}  # stem -> 'batches/batch_XXXXX.zip'
    existing_zip_paths = set()
    if os.path.exists(index_path):
        try:
            with open(index_path) as f:
                zip_map = json.load(f)
        except Exception as exc:
            print(f"⚠ 读取 _zip_index.json 失败（{exc}），按空索引处理")
            zip_map = {}
    # 只按文件名跳过已生成批次，不读其内容
    existing_zip_paths = set(sorted(glob.glob(os.path.join(batches_dir, 'batch_*.zip'))))

    chunks = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    print(f"[preprocess] zip 批次模式: {len(items)} 张 -> {len(chunks)} 批"
          f"（batch_size={batch_size}, staging={staging_root}, 约需 "
          f"{(zip_workers + 2) * batch_size * _NPY_MB / 1024:.1f} GB staging 空间）")

    failures = []
    npy_pool = _make_pool(workers)
    zip_cache = _zip_cache_ok(npy_pool)
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
            jobs = [(item, os.path.join(stage_dir, _item_stem(item) + '.npy'),
                     vc_fixed, cfg, zip_cache) for item in chunk]
            failed_stems = set()
            for item, err in npy_pool.map(_process_one, jobs):
                if err is not None:
                    failures.append((item, err))
                    failed_stems.add(_item_stem(item))

            # 2) 提交 zip 打包（与下一批 npy 生产并行）
            if len(chunk) == len(failed_stems):
                shutil.rmtree(stage_dir, ignore_errors=True)
                continue
            pending.append((zip_pool.submit(_zip_dir, stage_dir, zip_path, zip_compress,
                                            staging_root),
                            stage_dir))
            for item in chunk:
                stem = _item_stem(item)
                if stem not in failed_stems:
                    zip_map[stem] = zip_rel

            # 增量持久化索引（便于 OSS 场景断点续跑时不读 zip 内容）
            with open(index_path, 'w') as f:
                json.dump(zip_map, f)

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
                   zip_compress: int = 1, zip_workers: int = 2,
                   flare_bank_path: str = 'flare_templates/512/'):
    """基线离线预处理主入口。

    gt_dir: 输入源：GT 目录（*.png / *.jpg），或未解压的 merged_512.zip（zip 源模式）
    output_dir: 输出目录（linear_raw 或 batches/*.zip、manifest.json 将写于此）
    cfg: config dict (configs/degradation_baseline.yaml)
    workers: 并行进程数（1=串行，保持原行为；>1 并行加速，断点续跑/产物一致）
    zip_batch_size: >0 时启用 zip 批次模式（每批一个 zip，按 manifest 打乱顺序切分）；
                    0 = 传统松散模式（linear_raw/*.npy 直接落盘，兼容旧行为）
    staging_dir: zip 模式下 npy 暂存目录（必须在本地系统盘；默认系统临时目录）
    zip_compress: zip 压缩级别 0-9（0=不压缩，速度最快；默认 1）
    zip_workers: zip 打包线程数（默认 2；staging 峰值 ≈ (zip_workers+2)*batch*3.2MB）
    flare_bank_path: 眩光模板库路径（写入 manifest；相对 output_dir 或绝对）
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

    # 解析 GT 源（目录 or zip）
    is_zip_source = gt_dir.lower().endswith('.zip')
    zip_source_abs = os.path.abspath(gt_dir) if is_zip_source else None
    if is_zip_source:
        if not os.path.exists(gt_dir):
            raise FileNotFoundError(f"GT zip 不存在: {gt_dir}")
        with zipfile.ZipFile(gt_dir) as zf:
            all_entries = zf.namelist()
            # merged_512.zip 即切片图集：任意图片条目都是 GT 图。
            # 注意：条目常为 'gt/00000000.png'（gt 在最前、无前导/），
            # 不能用 '/gt/' 这种要求嵌套路径的子串匹配。
            entries = [e for e in all_entries
                       if e.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not entries:
            raise RuntimeError(
                f"zip 内未找到图片条目（共 {len(all_entries)} 个条目）: {gt_dir}\n"
                f"  样例条目: {all_entries[:8]}")
        items = sorted(f'{_ZIP_PREFIX}{zip_source_abs}!/{e}' for e in entries)
        print(f"[preprocess] zip 源模式: {zip_source_abs}（{len(items)} 张 GT 图，"
              f"如 {entries[0]} ...）")
    else:
        items = sorted(glob.glob(os.path.join(gt_dir, '*.png')) +
                       glob.glob(os.path.join(gt_dir, '*.jpg')) +
                       glob.glob(os.path.join(gt_dir, '*.jpeg')))

    zip_map = None
    if zip_batch_size and zip_batch_size > 0:
        staging = staging_dir or os.path.join(tempfile.gettempdir(), 'hypir_preprocess_staging')
        failures, zip_map = _run_zip_batches(items, output_dir, vc_fixed, cfg,
                                             batch_size=int(zip_batch_size), workers=workers,
                                             staging_root=staging, zip_compress=zip_compress,
                                             zip_workers=zip_workers)
    else:
        # 传统松散模式
        failures = _run_legacy(items, output_dir, vc_fixed, cfg, workers)

    if failures:
        print(f"[preprocess] ⚠ {len(failures)} 张处理失败（均已跳过 manifest）：")
        for fp, err in failures[:10]:
            print(f"    {_item_stem(fp)}: {err}")

    # 生成 manifest
    generate_manifest(items, output_dir, cfg['preprocess_version'], zip_map=zip_map,
                      zip_source=zip_source_abs, flare_bank_path=flare_bank_path)
    print(f"[preprocess] done. manifest -> {os.path.join(output_dir, 'manifest.json')}")


def resolve_bank_dir(output_dir: str, flare_bank_path: str) -> str:
    """把 manifest 的 flare_bank_path 解析为实际模板目录（绝对路径直接用，相对则相对 output_dir）。"""
    return flare_bank_path if os.path.isabs(flare_bank_path) else os.path.join(output_dir, flare_bank_path)


def generate_manifest(items, output_dir: str, version: str, zip_map: dict = None,
                      zip_source: str = None, flare_bank_path: str = 'flare_templates/512/'):
    """生成 manifest.json。

    items: GT 条目列表（目录路径或 'zip://<zip>!/<entry>'；由 run_preprocess 传入）
    zip_map: None 时为传统松散模式（沿用旧字段）；
             否则 {stem: 'batches/batch_XXXXX.zip'}，每条记录追加 'zip' 字段，
             且仅收录 zip_map 中包含的成功样本。
    zip_source: 非 None 时表示 zip 源模式：记录 gt_zip+gt_arc（gt_path=None）
    flare_bank_path: 眩光模板库的路径（相对 output_dir 或绝对；训练侧按此解析）
    """
    files = []
    for item in items:
        name = _item_stem(item)
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
            'gt_path': None if zip_source is not None else os.path.abspath(item),
            'flare3d_input_path': None,
            'flare3d_gt_path': None,
            'linear_raw': linear_raw_rel,
        }
        if zip_source is not None:
            record['gt_zip'] = zip_source
            record['gt_arc'] = item.split('!/', 1)[1]
        if zip_rel is not None:
            record['zip'] = zip_rel
        files.append(record)

    manifest = {
        'preprocess_version': version,
        'manifest_dir': '.',
        'files': files,
        'psf_bank_version': None,
        'psf_bank_path': None,
        'flare_bank_path': flare_bank_path,
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
    parser.add_argument('--gt_dir', required=True,
                        help='GT 源：目录 或 未解压的 merged_512.zip（zip 源模式）')
    parser.add_argument('--output_dir', required=True,
                        help='输出目录（zip 模式可直接指向 OSS 挂载路径；'
                             '传统模式为本地目录）')
    parser.add_argument('--config', default=None,
                        help='主配置路径（默认 configs/degradation_baseline.yaml）')
    parser.add_argument('--num_templates', type=int, default=20,
                        help='生成 flare 模板数量')
    parser.add_argument('--workers', type=int, default=1,
                        help='并行进程数（默认 1=串行；建议设备核心数-1，如 8）')
    parser.add_argument('--zip-batch-size', type=int, default=512,
                        help='zip 批次大小：每批样本数生成一个 batch_XXXXX.zip（默认 512；'
                             '0=传统松散模式不打包）')
    parser.add_argument('--staging-dir', default=None,
                        help='zip 模式 npy 暂存目录（必须在本地系统盘；默认系统临时目录）')
    parser.add_argument('--zip-compress', type=int, default=1, choices=range(0, 10),
                        help='zip 压缩级别 0-9（默认 1；0=不压缩最快）')
    parser.add_argument('--zip-workers', type=int, default=2,
                        help='zip 打包线程数（默认 2；staging 峰值≈(zip_workers+2)*batch*3.2MB）')
    parser.add_argument('--flare-bank-path', default='flare_templates/512/',
                        help='眩光模板库路径，写入 manifest（相对 output_dir 或绝对；'
                             '默认 flare_templates/512/；训练侧据此解析）')
    args = parser.parse_args()

    cfg_path = args.config or os.path.join(_project_root(), 'configs/degradation_baseline.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    run_preprocess(args.gt_dir, args.output_dir, cfg, workers=args.workers,
                   zip_batch_size=args.zip_batch_size, staging_dir=args.staging_dir,
                   zip_compress=args.zip_compress, zip_workers=args.zip_workers,
                   flare_bank_path=args.flare_bank_path)

    # 若无 flare 模板，生成占位模板（路径跟随 flare_bank_path）
    flare_dir = resolve_bank_dir(args.output_dir, args.flare_bank_path)
    if not glob.glob(os.path.join(flare_dir, '*.npy')) and \
            not glob.glob(os.path.join(flare_dir, 'batch_*.zip')):
        print(f"[preprocess] generating {args.num_templates} flare templates -> {flare_dir}")
        generate_flare_templates(flare_dir, args.num_templates)
