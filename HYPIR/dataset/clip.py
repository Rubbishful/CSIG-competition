# -*- coding: utf-8 -*-
"""数据集切分与聚合工具（替代原 clip.py 的中心裁剪 + 缩放逻辑）。

职责：
- 递归扫描多个数据集目录（默认与 check_images.py 的 DATASETS 完全一致）
- 每张图切分为非重叠 TILE_SIZE x TILE_SIZE 网格块（短边 < TILE_SIZE 的图跳过并计数）
- 全部块全局打乱（固定 --seed 可复现），按乱序全局编号命名 00000001.png ...
- 输出 tiles.csv 溯源（id, dataset, 源路径, 源尺寸, 块坐标）
- 校验：产出块数 == Σ (w//T) * (h//T)（与 check_images.py 的估算口径一致）

依赖前置条件：所有源图短边 >= TILE_SIZE（由 check_images.py 验证），
因此不再需要小图缩放逻辑；防御性保留跳过+计数分支以防新增数据。

用法示例：
    # 全量切分（默认输出 Data/merged_512/）
    python HYPIR/dataset/clip.py --data-root Data --out-dir Data/merged_512 --workers 16

    # 小批量冒烟测试：每个数据集只取前 10 张
    python HYPIR/dataset/clip.py --data-root Data --out-dir smoke/merged --limit 10 --workers 4

    # 指定数据集（默认 3 个全部）
    python HYPIR/dataset/clip.py --datasets DIV2K_train_HR LSDIR --seed 123
"""

import argparse
import csv
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps

TILE_SIZE = 512
TARGET_SIZE = TILE_SIZE  # 兼容原命名

# 与 check_images.py 保持一致（覆盖原 clip.py 的 5 种 + tif/gif）
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

# (数据集名称, 相对 DATA_ROOT 的路径列表)，与 check_images.py 的 DATASETS 一致
DATASETS = [
    ("DIV2K_train_HR", ["DIV2K_train_HR"]),
    ("Flickr2K_HR", ["Flickr2K", "Flickr2K_HR"]),
    ("LSDIR", ["LSDIR"]),
]


# ---------------------------------------------------------------------------- #
# 阶段 1：收集与探查
# ---------------------------------------------------------------------------- #

def collect_images(root: Path):
    """递归收集图片文件。返回 (文件列表, 跳过文件数)。"""
    files, skipped = [], 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if Path(name).suffix.lower() in VALID_EXTENSIONS:
                files.append(Path(dirpath) / name)
            else:
                skipped += 1
    return files, skipped


def probe(path: Path):
    """只读图片头获得尺寸；应用 EXIF 方向（与切分阶段 exif_transpose 后的坐标一致）。"""
    try:
        with Image.open(path) as img:
            w, h = img.size
            # EXIF Orientation 5-8 表示旋转/镜像，交换宽高
            orient = img.getexif().get(0x0112, 1)
            if orient in (5, 6, 7, 8):
                w, h = h, w
            return ("ok", w, h)
    except Exception as exc:
        return (f"error: {type(exc).__name__}: {exc}", 0, 0)


def plan_tiles(records, tile_size: int):
    """为每张图生成非重叠网格块坐标。返回 (块记录列表, 短边不足跳过数)。"""
    tiles = []
    skipped_small = 0
    for dataset, src_path, w, h in records:
        if w < tile_size or h < tile_size:
            skipped_small += 1
            continue
        nx, ny = w // tile_size, h // tile_size
        for ty in range(ny):
            for tx in range(nx):
                tiles.append((dataset, src_path, w, h, tx * tile_size, ty * tile_size))
    return tiles, skipped_small


# ---------------------------------------------------------------------------- #
# 阶段 2：切块保存（进程池）
# ---------------------------------------------------------------------------- #

def crop_and_save(job):
    """job: (tile_size, out_dir_str, src_path_str, [(idx, x0, y0), ...]) -> (src_path, error|None)"""
    tile_size, out_dir_str, src_path_str, items = job
    try:
        with Image.open(src_path_str) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            for idx, x0, y0 in items:
                tile = img.crop((x0, y0, x0 + tile_size, y0 + tile_size))
                tile.save(Path(out_dir_str) / f"{idx:08d}.png", format="PNG")
        return (src_path_str, None)
    except Exception as exc:
        return (src_path_str, f"{type(exc).__name__}: {exc}")


def _make_pool(workers: int, backend: str):
    """创建并行池。backend=process 时若进程池初始化失败（如受限环境禁止命名管道），
    自动回退线程池并提示；PIL 解码/编码会释放 GIL，线程池仍有明显加速。"""
    if backend == "thread":
        return ThreadPoolExecutor(max_workers=workers)
    try:
        return ProcessPoolExecutor(max_workers=workers)
    except (PermissionError, OSError, RuntimeError) as exc:
        print(f"⚠ 进程池不可用（{type(exc).__name__}: {exc}），回退线程池")
        return ThreadPoolExecutor(max_workers=workers)


# ---------------------------------------------------------------------------- #
# 主流程
# ---------------------------------------------------------------------------- #

def run_split(data_root: Path, dataset_names, out_dir: Path, tile_size: int = TILE_SIZE,
              workers: int = 16, seed: int = 42, limit: int = 0, force: bool = False,
              backend: str = "process"):
    """切分 + 打乱 + 聚合主入口。返回 (tile_count, expected_count, stats)。"""
    # 防误覆盖：输出目录非空时拒绝，除非 --force
    gt_dir = out_dir / "gt"
    if gt_dir.exists() and any(gt_dir.iterdir()) and not force:
        raise FileExistsError(
            f"输出目录 {gt_dir} 已存在且非空，请更换 --out-dir 或使用 --force 清空重跑")

    if gt_dir.exists() and force:
        import shutil
        shutil.rmtree(gt_dir)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 收集 ----
    all_records = []   # (dataset, src_path, w, h)
    skipped_by_ext = {}
    errors_probe = []
    stats = {}

    for name, rel_paths in dataset_names:
        root = data_root.joinpath(*rel_paths)
        if not root.is_dir():
            print(f"[{name}] 目录不存在，跳过: {root}")
            continue
        files, skipped = collect_images(root)
        skipped_by_ext[name] = skipped

        if limit > 0:
            files = files[:limit]

        # 探查尺寸（线程池，只读头部）
        records = []
        bad = 0
        with ThreadPoolExecutor(max_workers=max(4, workers)) as pool:
            futures = {pool.submit(probe, f): f for f in files}
            for fut in as_completed(futures):
                status, w, h = fut.result()
                if status != "ok":
                    bad += 1
                    if len(errors_probe) < 10:
                        errors_probe.append((str(futures[fut]), status))
                    continue
                records.append((name, str(futures[fut]), w, h))

        all_records.extend(records)
        stats[name] = {"files": len(files), "ok": len(records), "bad": bad,
                       "skipped_ext": skipped}
        print(f"[{name}] 收集 {len(files)} 张，可切 {len(records)} 张，"
              f"异常 {bad} 张" + (f"（limit={limit}）" if limit else ""))

    if not all_records:
        raise RuntimeError("未收集到任何图片，请检查 --data-root/--datasets")

    # ---- 2. 规划块 ----
    tiles, skipped_small = plan_tiles(all_records, tile_size)
    print(f"已规划非重叠 {tile_size}x{tile_size} 块: {len(tiles)} 个 "
          f"（短边不足跳过 {skipped_small} 张）")

    # ---- 3. 全局打乱 + 编号 ----
    rng = random.Random(seed)
    ordered = list(tiles)
    rng.shuffle(ordered)
    print(f"全局打乱完成（seed={seed}），开始切块保存 ...")

    # ---- 4. 按源图分组，进程池切块保存 ----
    # 每个 job: 加载一张源图，切出其全部块
    by_src = {}
    for idx, (dataset, src_path, w, h, x0, y0) in enumerate(ordered):
        by_src.setdefault(src_path, []).append((idx, x0, y0))
    jobs = [(tile_size, str(gt_dir), src, items) for src, items in by_src.items()]

    failures = []
    done = 0
    with _make_pool(workers, backend) as pool:
        for src, err in pool.map(crop_and_save, jobs):
            done += 1
            if err is not None:
                failures.append((src, err))
            if done % 5000 == 0:
                print(f"  已处理源图 {done}/{len(jobs)} ...")

    tile_count = len(ordered)
    expected_count = sum((w // tile_size) * (h // tile_size) for _, _, w, h in all_records)

    # ---- 5. 校验 ----
    actual_count = len(list(gt_dir.glob("*.png")))
    if actual_count != tile_count:
        print(f"⚠ 磁盘块数 {actual_count} != 计划块数 {tile_count}")
    if tile_count != expected_count:
        print(f"⚠ 计划块数 {tile_count} != 期望块数 {expected_count}（请检查图片尺寸口径）")
    if failures:
        print(f"⚠ {len(failures)} 张源图切块失败:")
        for src, err in failures[:10]:
            print(f"    {src}: {err}")

    # ---- 6. 溯源表 ----
    csv_path = out_dir / "tiles.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "dataset", "src_path", "src_w", "src_h", "tile_x", "tile_y"])
        for idx, (dataset, src_path, w, h, x0, y0) in enumerate(ordered):
            writer.writerow([idx, dataset, src_path, w, h, x0, y0])

    # ---- 7. 汇总 ----
    print("\n=== 切分汇总 ===")
    for name, s in stats.items():
        print(f"  {name:<16} 收集 {s['files']:>6} | 可切 {s['ok']:>6} | 异常 {s['bad']:>4} "
              f"| 跳过非图 {s['skipped_ext']:>6}")
    print(f"  切出 {tile_size}x{tile_size} 块总数: {tile_count}（期望 {expected_count}）"
          f" → {'✓ 一致' if tile_count == expected_count else '✗ 不一致'}")
    print(f"  输出目录: {gt_dir}")
    print(f"  溯源表:   {csv_path}")
    if failures:
        print(f"✗ 失败源图 {len(failures)} 张，请处理后重跑")
    return tile_count, expected_count, stats


def main():
    parser = argparse.ArgumentParser(
        description="递归扫描数据集 -> 非重叠网格切块 -> 全局打乱 -> 编号聚合（替代原 clip.py）")
    parser.add_argument("--data-root", default="Data", help="数据集根目录（默认 Data）")
    parser.add_argument("--datasets", nargs="+", default=[d[0] for d in DATASETS],
                        help=f"数据集列表（默认: {' '.join(d[0] for d in DATASETS)}）")
    parser.add_argument("--out-dir", default="merged_512", help="输出目录（含 gt/ 与 tiles.csv）")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE, help="切块边长（默认 512）")
    parser.add_argument("--workers", type=int, default=16, help="切块进程数（默认 16）")
    parser.add_argument("--backend", choices=["process", "thread"], default="process",
                        help="并行后端（默认 process；受限环境自动回退 thread）")
    parser.add_argument("--seed", type=int, default=42, help="打乱随机种子（默认 42）")
    parser.add_argument("--limit", type=int, default=0,
                        help="每个数据集只取前 N 张（冒烟测试用，默认 0=全部）")
    parser.add_argument("--force", action="store_true", help="输出目录已存在时清空重跑")
    args = parser.parse_args()

    data_root = Path(os.path.abspath(args.data_root))
    out_dir = Path(os.path.abspath(args.out_dir))
    print(f"数据集根目录: {data_root}")
    print(f"输出目录:     {out_dir}")

    dataset_names = []
    for name in args.datasets:
        for dname, rel_paths in DATASETS:
            if name == dname:
                dataset_names.append((dname, rel_paths))
                break
        else:
            print(f"⚠ 未知数据集 {name}，跳过（可选: {' '.join(d[0] for d in DATASETS)}）")

    run_split(data_root, dataset_names, out_dir, tile_size=args.tile_size,
              workers=args.workers, seed=args.seed, limit=args.limit, force=args.force,
              backend=args.backend)


if __name__ == "__main__":
    main()
