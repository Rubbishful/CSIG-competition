#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 Data/ 下指定图像数据集的图片数量，并粗略检查分辨率等级。
主要用于判断是否有较多高分辨率图像可以切分为 512x512 块以增加训练数据量。

数据集布局（相对于 DATA_ROOT，默认是脚本所在目录下的 Data/）：
- DIV2K_train_HR/             图片直接位于该目录下
- Flickr2K/Flickr2K_HR/       图片直接位于该目录下
- LSDIR/                      含较多子目录，递归扫描

用法：
python3 check_images.py
python3 check_images.py --data-root /path/to/Data
python3 check_images.py --show 10
python3 check_images.py --workers 16
"""

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image
except ImportError:
    sys.exit("需要 Pillow 库，请先安装：pip install pillow")


DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data")

# (数据集名称, 相对 DATA_ROOT 的路径列表)，均递归扫描
DATASETS = [
    ("DIV2K_train_HR", ["DIV2K_train_HR"]),
    ("Flickr2K_HR", ["Flickr2K", "Flickr2K_HR"]),
    ("LSDIR", ["LSDIR"]),
]

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".gif", ".tif", ".tiff", ".webp"
}

CROP_SIZE = 512


def make_short_side_bins(crop_size):
    """根据目标切块大小生成短边统计区间。"""
    c = int(crop_size)
    return [
        (0, c, f"<{c}"),
        (c, int(c * 1.5), f"{c}-{int(c * 1.5) - 1}"),
        (int(c * 1.5), c * 2, f"{int(c * 1.5)}-{c * 2 - 1}"),
        (c * 2, c * 3, f"{c * 2}-{c * 3 - 1}"),
        (c * 3, c * 4, f"{c * 3}-{c * 4 - 1}"),
        (c * 4, c * 6, f"{c * 4}-{c * 6 - 1}"),
        (c * 6, c * 8, f"{c * 6}-{c * 8 - 1}"),
        (c * 8, float("inf"), f">={c * 8}"),
    ]


SHORT_SIDE_BINS = make_short_side_bins(CROP_SIZE)

# 单张图可切出多少个 512x512 块的粗略等级
PATCH_BINS = [
    (0, 1, "0"),
    (1, 2, "1"),
    (2, 4, "2-3"),
    (4, 8, "4-7"),
    (8, 16, "8-15"),
    (16, 32, "16-31"),
    (32, float("inf"), ">=32"),
]


def collect_images(root):
    """递归收集目录下所有图片文件路径。返回 (文件列表, 跳过文件数)。"""
    files, skipped = [], 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                files.append(os.path.join(dirpath, name))
            else:
                skipped += 1
    return files, skipped


def probe(path):
    """
    读取单张图片的尺寸。
    为了保证大数据库扫描速度，这里默认只读取图片头信息。
    """
    try:
        with Image.open(path) as img:
            return ("ok", img.size)
    except Exception as exc:
        return (f"error: {type(exc).__name__}", None)


def get_bin_label(value, bins):
    """根据数值返回对应区间名称。"""
    for low, high, label in bins:
        if low <= value < high:
            return label
    return bins[-1][2]


def analyze_dataset(data_root, rel_paths, workers, show):
    name = rel_paths[0]
    root = os.path.join(data_root, *rel_paths)

    if not os.path.isdir(root):
        print(f"[{name}] 目录不存在，跳过: {root}")
        return None

    files, skipped = collect_images(root)
    if not files:
        print(f"[{name}] 未找到图片文件: {root}")
        return None

    total_ok = 0
    bad_count = 0

    small_count = 0
    croppable_count = 0
    total_patches = 0

    short_counter = Counter()
    patch_counter = Counter()

    small_examples = []
    bad_examples = []

    processed = 0
    chunk_size = max(1000, workers * 200)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(files), chunk_size):
            chunk = files[start:start + chunk_size]
            futures = {pool.submit(probe, f): f for f in chunk}

            for future in as_completed(futures):
                f = futures[future]
                status, size = future.result()
                processed += 1

                if status != "ok":
                    bad_count += 1
                    if len(bad_examples) < show:
                        bad_examples.append(f)
                    continue

                total_ok += 1
                w, h = size
                short = min(w, h)

                # 短边等级统计
                short_label = get_bin_label(short, SHORT_SIDE_BINS)
                short_counter[short_label] += 1

                # 短边不足 512
                if short < CROP_SIZE:
                    small_count += 1
                    if len(small_examples) < show:
                        small_examples.append((f, w, h))

                # 估算不重叠可切出的 512x512 块数
                patches = (w // CROP_SIZE) * (h // CROP_SIZE)
                total_patches += patches

                if patches > 0:
                    croppable_count += 1

                patch_label = get_bin_label(patches, PATCH_BINS)
                patch_counter[patch_label] += 1

                if processed % 10000 == 0:
                    print(f"  [{name}] 已检查 {processed}/{len(files)} ...")

    print(f"\n=== {name} ===")
    print(f"目录: {root}")
    print(
        f"图片总数: {len(files)} "
        f"(跳过非图片文件 {skipped} 个, 成功读取 {total_ok} 张, 损坏/无法读取 {bad_count} 张)"
    )

    print("短边等级分布:")
    for _, _, label in SHORT_SIDE_BINS:
        print(f"    {label:>12}: {short_counter.get(label, 0):>8} 张")

    print(f"短边不足 {CROP_SIZE} 的图片: {small_count} 张")

    print(f"可切出至少 1 个 {CROP_SIZE}x{CROP_SIZE} 块的图片: {croppable_count} 张")
    if croppable_count > 0:
        avg_patches = total_patches / croppable_count
        print(
            f"预计可非重叠切出 {CROP_SIZE}x{CROP_SIZE} 块总数: {total_patches} 块 "
            f"(可切图片平均 {avg_patches:.2f} 块/张)"
        )
    else:
        print(f"预计可非重叠切出 {CROP_SIZE}x{CROP_SIZE} 块总数: 0 块")

    print("单图可切块数等级分布:")
    for _, _, label in PATCH_BINS:
        print(f"    {label:>12}: {patch_counter.get(label, 0):>8} 张")

    if small_examples:
        print(f"短边不足 {CROP_SIZE} 示例 (最多 {show} 个):")
        for f, w, h in small_examples:
            print(f"    {w}x{h}\t{f}")

    if bad_examples:
        print(f"损坏/无法读取示例 (最多 {show} 个):")
        for f in bad_examples:
            print(f"    {f}")

    return {
        "name": name,
        "total": len(files),
        "ok": total_ok,
        "bad": bad_count,
        "small": small_count,
        "croppable": croppable_count,
        "patches": total_patches,
        "short_counter": short_counter,
        "patch_counter": patch_counter,
    }


def main():
    parser = argparse.ArgumentParser(
        description="统计图像数量、粗略分辨率等级，并估算可切分 512x512 块的数量"
    )
    parser.add_argument(
        "--data-root",
        default=DEFAULT_ROOT,
        help="数据集根目录（默认: 脚本所在目录下的 Data/）"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="并发读取线程数（默认 16）"
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="最多打印多少个短边不足/损坏示例（默认 5，设为 0 可关闭）"
    )
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    print(f"数据集根目录: {data_root}")
    print(f"目标切块大小: {CROP_SIZE}x{CROP_SIZE}")

    results = []
    for name, rel_paths in DATASETS:
        res = analyze_dataset(data_root, rel_paths, args.workers, args.show)
        if res:
            results.append(res)

    if not results:
        print("没有可统计的数据集。")
        return

    print("\n" + "=" * 80)
    print("汇总:")

    global_short_counter = Counter()
    global_patch_counter = Counter()

    total_all = 0
    ok_all = 0
    bad_all = 0
    small_all = 0
    croppable_all = 0
    patches_all = 0

    for r in results:
        print(
            f"  {r['name']:<16} "
            f"总数 {r['total']:>7} | "
            f"成功 {r['ok']:>7} | "
            f"短边<{CROP_SIZE} {r['small']:>7} | "
            f"可切图 {r['croppable']:>7} | "
            f"可切块 {r['patches']:>8}"
        )

        total_all += r["total"]
        ok_all += r["ok"]
        bad_all += r["bad"]
        small_all += r["small"]
        croppable_all += r["croppable"]
        patches_all += r["patches"]

        global_short_counter.update(r["short_counter"])
        global_patch_counter.update(r["patch_counter"])

    print(
        f"  {'合计':<16} "
        f"总数 {total_all:>7} | "
        f"成功 {ok_all:>7} | "
        f"短边<{CROP_SIZE} {small_all:>7} | "
        f"可切图 {croppable_all:>7} | "
        f"可切块 {patches_all:>8}"
    )

    print("\n全局短边等级分布:")
    for _, _, label in SHORT_SIDE_BINS:
        print(f"    {label:>12}: {global_short_counter.get(label, 0):>8} 张")

    print("\n全局单图可切块数等级分布:")
    for _, _, label in PATCH_BINS:
        print(f"    {label:>12}: {global_patch_counter.get(label, 0):>8} 张")

    if ok_all > 0:
        print(
            f"\n成功读取图片中，短边不足 {CROP_SIZE} 的比例: "
            f"{small_all / ok_all * 100:.2f}%"
        )

    if patches_all > ok_all:
        print(
            f"提示：成功读取的 {ok_all} 张图，预计可非重叠切出 "
            f"{patches_all} 个 {CROP_SIZE}x{CROP_SIZE} 块，"
            f"约为原图数量的 {patches_all / max(ok_all, 1):.2f} 倍。"
        )

    if small_all == 0:
        print(f"✓ 所有成功读取图片的短边均 >= {CROP_SIZE}，都可以尝试切 {CROP_SIZE}x{CROP_SIZE}。")
    else:
        print(f"✗ 有 {small_all} 张成功读取图片短边不足 {CROP_SIZE}，不能直接切 {CROP_SIZE}x{CROP_SIZE}。")

    if bad_all > 0:
        print(f"⚠ 存在 {bad_all} 张损坏/无法读取图片，建议后续训练前剔除。")


if __name__ == "__main__":
    main()
