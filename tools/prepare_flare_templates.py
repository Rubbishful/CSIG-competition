# -*- coding: utf-8 -*-
"""Flare2D 模板预处理：从 FlareX Flare2D 输入/GT 对提取光斑层并生成 512x512x4 模板。

背景（与数据集讨论结论一致）：
  - Flare2D 共 9,500 对 1440x1440（input 含光斑, gt 为干净场景），光斑层 = clip(input-gt, 0, 1)，
    非零像素占比实测 34%~47%（大面积 bloom + 星芒），亮区位置每张不同；
  - 因此不能随机裁剪（会切掉主体），也不能裁到亮区再放大（亮区太大）；采用
    【全图 LANCZOS 下采样 1440->512，零裁切、确定性】；
  - 输出 [512,512,4] float32：RGB = 光斑层，Alpha = 每像素亮度（max channel），
    与 FlareTemplateBank.sample() / apply_flare（screen 混合）直接兼容；
  - 存储按模板 zip 批次（batch_XXXXX.zip，内含 templates/{id:04d}.npy），
    模板库 FlareTemplateBank 已支持 zip + LRU 读取。

质量控制（v2）：
  - 过滤近空/低质模板（默认 peak>=0.05 且 coverage(>0.02)>=0.01），按原因计数；
  - 未凑满 N 个有效模板时继续抽样，直到达到 --num-templates 或用尽候选对；
  - 逐 zip 数量可能因过滤而不均（如 128/127），这不影响 FlareTemplateBank 均匀采样。

用法：
    python tools/prepare_flare_templates.py \
        --flare2d-root Data/FlareX/FlareX/Flare2D \
        --out-dir flare_templates/512 --num-templates 1024 --templates-per-zip 128 --workers 8
"""

import argparse
import csv
import glob
import os
import random
import shutil
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TARGET_SIZE = 512
SOURCE_SIZE = 1440

# 质量门槛默认值（峰值 & 覆盖率）
DEFAULT_MIN_PEAK = 0.05
DEFAULT_MIN_COVERAGE = 0.01


def load_pair(input_path: str, gt_path: str):
    """读取 input/gt 为 float32 [H,W,3] [0,1] RGB。

    用 Pillow 而非 cv2：FlareX PNG 带重复 eXIf 块，cv2.imread 会触发大量
    'libpng warning: eXIf: duplicate'（libpng 要求每图最多一个 eXIf），
    Pillow 解码时不打印该警告且像素一致（PIL 本即 RGB）。
    """
    def load(p: str) -> np.ndarray:
        im = Image.open(p)
        im.load()
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0

    inp = load(input_path)
    gt = load(gt_path)
    return inp, gt


def convert_template(input_path: str, gt_path: str, size: int = TARGET_SIZE,
                     min_peak: float = DEFAULT_MIN_PEAK,
                     min_coverage: float = DEFAULT_MIN_COVERAGE):
    """单对 -> 模板 [size,size,4] float32（全图下采样，零裁切 + 质量门槛）。

    Returns:
        (template, stats) 或 (None, stats)。stats 含 peak/coverage/cx/cy/reason
        （reason in {'empty','low_peak','low_coverage','ok'}）。
    """
    inp, gt = load_pair(input_path, gt_path)
    flare = np.clip(inp - gt, 0.0, 1.0)
    alpha = flare.max(axis=2)  # 每像素亮度
    peak = float(alpha.max())
    cov = float((alpha > 0.02).mean())
    idx = int(alpha.argmax())
    cy, cx = idx // flare.shape[1], idx % flare.shape[1]

    # 质量门槛：过滤空/近空/极淡光斑，避免把无效模板入库
    if peak < min_peak or cov < min_coverage:
        reason = 'empty' if peak < 0.01 else ('low_peak' if peak < min_peak else 'low_coverage')
        return None, {"peak": peak, "coverage": cov, "cx": -1, "cy": -1, "reason": reason}

    # 全图 LANCZOS 下采样（与源同纵横比，1440->512）
    h, w = flare.shape[:2]
    nw = int(round(w * size / max(w, h)))
    nh = int(round(h * size / max(w, h)))
    rgb = cv2.resize(flare, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    al = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    # 居中补零到精确 size x size（消除取整误差，不改变像素内容）；
    # LANCZOS 振铃可能使少量像素略超 [0,1]，裁剪保证模板契约
    canvas = np.zeros((size, size, 4), dtype=np.float32)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw, :3] = rgb
    canvas[y0:y0 + nh, x0:x0 + nw, 3] = al
    canvas = np.clip(canvas, 0.0, 1.0)

    stats = {"peak": peak, "coverage": cov, "cx": cx, "cy": cy, "reason": "ok"}
    return canvas, stats


def prepare_bank(flare2d_root, out_dir, num_templates, templates_per_zip, seed,
                 size, zip_compress, workers, force, min_peak, min_coverage):
    os.makedirs(out_dir, exist_ok=True)
    if not force and glob_zips(out_dir):
        raise FileExistsError(f"{out_dir} 已存在模板 zip，请更换 --out-dir 或 --force 重跑")

    input_dir = os.path.join(flare2d_root, "input")
    gt_dir = os.path.join(flare2d_root, "gt")
    inputs = sorted(Path(input_dir).glob("*.png")) + sorted(Path(input_dir).glob("*.jpg"))
    pairs = []
    for p in inputs:
        g = Path(gt_dir) / (p.stem + p.suffix)
        if g.exists():
            pairs.append((str(p), str(g)))
    print(f"[flare2d] 发现 {len(pairs)} 对 input/gt")

    # 确定性打乱后，按需抽样直到凑满 N 个【有效】模板
    rng = random.Random(seed)
    pool_pairs = list(pairs)
    rng.shuffle(pool_pairs)

    valid = []            # (template ndarray, stats)
    reason_counts = Counter()
    processed = 0
    cand_chunk = max(templates_per_zip, workers * 2)
    target = min(num_templates, len(pairs))
    t0 = time.time()

    while len(valid) < target and processed < len(pool_pairs):
        chunk = pool_pairs[processed:processed + cand_chunk]
        processed += len(chunk)

        def work(pair):
            t, s = convert_template(pair[0], pair[1], size, min_peak, min_coverage)
            s["src"] = os.path.basename(pair[0])
            return t, s

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for tmpl, stats in pool.map(work, chunk):
                if tmpl is None:
                    reason_counts[stats["reason"]] += 1
                    continue
                valid.append((tmpl, stats))
                if len(valid) >= target:
                    break  # 达到目标即停，避免单批越界补多

        if len(valid) >= target:
            break

        if processed % (cand_chunk * 5) == 0:
            print(f"[flare2d] 已处理 {processed}/{len(pool_pairs)} 候选，有效 {len(valid)} ...")

    missing = num_templates - len(valid)
    print(f"[flare2d] 有效模板 {len(valid)}/{num_templates}（共处理 {processed} 候选"
          f"{'，候选已用尽' if processed >= len(pool_pairs) else ''}）")
    if reason_counts:
        print(f"[flare2d] 过滤统计: " + ", ".join(
            f"{k}={reason_counts[k]}" for k in ["empty", "low_peak", "low_coverage"] if reason_counts[k]))
    if missing > 0:
        print(f"[flare2d] ⚠ 未凑满 {num_templates}（缺 {missing}，可能门槛过严或候选不足）")
    if not valid:
        print("[flare2d] ⚠ 未生成任何有效模板，请检查 Flare2D 数据/门槛")
        return 0, 0

    # 逐 zip 打包（stage -> zip -> 删，与 preprocess 一致的小文件聚合策略）
    staging_root = os.path.join(tempfile_gettempdir(), "hypir_flare_staging")
    os.makedirs(staging_root, exist_ok=True)
    n_zips = 0
    stats_rows = []
    for k in range(0, len(valid), templates_per_zip):
        batch_id = f"batch_{n_zips:05d}"
        stage_dir = os.path.join(staging_root, batch_id)
        zip_path = os.path.join(out_dir, batch_id + ".zip")
        os.makedirs(stage_dir, exist_ok=True)

        batch = valid[k:k + templates_per_zip]
        for i, (tmpl, stats) in enumerate(batch):
            tid = k + i
            np.save(os.path.join(stage_dir, f"{tid:04d}.npy"), tmpl)
            stats_rows.append({"id": tid, "src": stats["src"], "peak": stats["peak"],
                               "coverage": stats["coverage"], "cx": stats["cx"],
                               "cy": stats["cy"]})

        tmp_zip = zip_path + ".tmp"
        with zipfile.ZipFile(tmp_zip, "w") as zf:
            for n in sorted(os.listdir(stage_dir)):
                if n.endswith(".npy"):
                    zf.write(os.path.join(stage_dir, n), arcname=f"templates/{n}")
        os.replace(tmp_zip, zip_path)
        shutil.rmtree(stage_dir, ignore_errors=True)
        n_zips += 1
        print(f"[flare2d] 批次 {batch_id}: {len(batch)} 个模板已打包 -> {os.path.basename(zip_path)}")

    # 统计汇总
    peaks = [r["peak"] for r in stats_rows]
    covs = [r["coverage"] for r in stats_rows]
    cxs = [r["cx"] for r in stats_rows]
    cys = [r["cy"] for r in stats_rows]
    print("\n=== Flare2D 模板统计 ===")
    print(f"  模板数: {len(valid)}（{n_zips} 个 zip）")
    print(f"  峰值: min {min(peaks):.3f} / mean {float(np.mean(peaks)):.3f} / max {max(peaks):.3f}")
    print(f"  覆盖占比(>0.02): min {min(covs):.3f} / mean {float(np.mean(covs)):.3f} / max {max(covs):.3f}")
    print(f"  亮区质心: x mean {float(np.mean(cxs)):.0f} / y mean {float(np.mean(cys)):.0f}"
          f"（源 {SOURCE_SIZE}x{SOURCE_SIZE} 坐标）")
    print(f"  耗时: {time.time() - t0:.0f}s")
    info_path = os.path.join(out_dir, "flare_templates_info.csv")
    with open(info_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "src", "peak", "coverage", "cx", "cy"])
        w.writeheader()
        w.writerows(stats_rows)
    print(f"  信息表: {info_path}")
    return len(valid), n_zips


def glob_zips(out_dir):
    return sorted(glob.glob(os.path.join(out_dir, "batch_*.zip")))


def tempfile_gettempdir():
    import tempfile
    return tempfile.gettempdir()


def main():
    parser = argparse.ArgumentParser(
        description="Flare2D -> 512x512x4 模板（zip 批次，零裁切全图下采样，含质量门槛）")
    parser.add_argument("--flare2d-root", default="Data/FlareX/FlareX/Flare2D",
                        help="Flare2D 根目录（含 input/ 与 gt/）")
    parser.add_argument("--out-dir", default="flare_templates/512",
                        help="模板库输出目录（生成 batch_XXXXX.zip）")
    parser.add_argument("--num-templates", type=int, default=512,
                        help="期望的有效模板数（默认 512；全量可设 9500）")
    parser.add_argument("--templates-per-zip", type=int, default=128,
                        help="每个 zip 的模板数（默认 128）")
    parser.add_argument("--min-peak", type=float, default=DEFAULT_MIN_PEAK,
                        help="峰值下限：光斑层最大亮度不低于该值（默认 0.05）")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                        help="覆盖率下限：alpha>0.02 像素占比不低于该值（默认 0.01）")
    parser.add_argument("--seed", type=int, default=42, help="抽样随机种子")
    parser.add_argument("--size", type=int, default=TARGET_SIZE, help="模板边长（默认 512）")
    parser.add_argument("--zip-compress", type=int, default=1, choices=range(0, 10),
                        help="zip 压缩级别（默认 1）")
    parser.add_argument("--workers", type=int, default=8, help="转换线程数")
    parser.add_argument("--force", action="store_true", help="输出目录已存在时清空重跑")
    args = parser.parse_args()

    if args.force and glob_zips(args.out_dir):
        shutil.rmtree(args.out_dir)

    prepare_bank(args.flare2d_root, args.out_dir, args.num_templates,
                 args.templates_per_zip, args.seed, args.size, args.zip_compress,
                 args.workers, args.force, args.min_peak, args.min_coverage)
    print(f"[flare2d] 完成 -> {args.out_dir}（FlareTemplateBank 自动识别 zip 批次）")


if __name__ == "__main__":
    main()
