"""evaluate_tool CLI:批量分析与评估入口。

用法:
    python -m evaluate_tool.analyze paired --gt-dir Data/competition/evaluate \
        --pred-dir Data/competition/Result/2026-08-18-1 \
        --lq-dir Data/competition/evaluate --out-dir evaluate_output/paired_xxx
    python -m evaluate_tool.analyze nr --dir Data/competition/test \
        --dir2 Data/competition/Result/2026-08-18-1 --metric niqe,maniqa,musiq
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from evaluate_tool import degradation, metrics, report
from evaluate_tool.nr_metrics import NRMetricSet, heuristic_fingerprint

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_OUT_ROOT = Path("evaluate_output")


def load_image(path: Path) -> np.ndarray:
    """读取图片为 uint8 RGB numpy (H,W,3)。"""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def collect_images(directory: Path) -> list[Path]:
    """递归收集目录下所有图片并排序。"""
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    return sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )


def extract_case_number(stem: str) -> Optional[int]:
    """从文件名提取第一个数字作为配对键(case1_gt → 1, case-1 → 1, case100 → 100)。"""
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else None


def make_case_map(paths: list[Path], include: str | None = None) -> dict[int, Path]:
    """按数字序号建索引;提取不到数字的按排序位置补 0..n-1 并告警。

    include: 可选文件名包含子串过滤(如 "_gt"),用于区分同目录下的 GT/LQ。
    """
    if include is not None:
        paths = [p for p in paths if include in p.stem]
    mapping: dict[int, Path] = {}
    unnamed = []
    for p in paths:
        n = extract_case_number(p.stem)
        if n is None:
            unnamed.append(p)
        elif n in mapping:
            raise ValueError(f"序号冲突: {mapping[n]} 与 {p} 都映射到 case {n}")
        else:
            mapping[n] = p
    if unnamed:
        print(f"[警告] {len(unnamed)} 个文件无数字序号,按文件名排序兜底配对:")
        for p in unnamed:
            print(f"  - {p}")
        for i, p in enumerate(sorted(unnamed, key=lambda x: str(x))):
            while i in mapping:
                i += 1
            mapping[i] = p
    return mapping


def _outdir_for(sub: str, out_dir: Optional[str]) -> Path:
    if out_dir:
        d = Path(out_dir)
    else:
        d = DEFAULT_OUT_ROOT / f"{sub}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# paired(成对,有 GT)
# ---------------------------------------------------------------------------
def cmd_paired(args: argparse.Namespace) -> int:
    gt_map = make_case_map(collect_images(Path(args.gt_dir)), include=args.gt_suffix)
    pred_map = make_case_map(collect_images(Path(args.pred_dir)), include=args.pred_suffix)
    lq_map = (
        make_case_map(collect_images(Path(args.lq_dir)), include=args.lq_suffix)
        if args.lq_dir
        else {}
    )

    cases = sorted(set(gt_map) & set(pred_map))
    if not cases:
        print("错误: GT 与 pred 没有可配对的序号(检查文件名数字)。")
        return 2
    if args.lq_dir:
        lq_cases = sorted(set(cases) & set(lq_map))
        if len(lq_cases) != len(cases):
            print(f"[警告] LQ 目录仅配对到 {len(lq_cases)}/{len(cases)} 个 case,"
                  f"缺失: {sorted(set(cases) - set(lq_map))}")
        cases = lq_cases

    if args.limit:
        cases = cases[: args.limit]

    print(f"配对 {len(cases)} 个 case,开始评估 ...")
    rows = []
    for i, case in enumerate(cases, 1):
        gt = load_image(gt_map[case])
        pred = load_image(pred_map[case])
        if gt.shape != pred.shape:
            print(f"[跳过] case{case}: 尺寸不一致 GT{gt.shape} vs pred{pred.shape}")
            continue

        row = {
            "case": case,
            "gt_file": str(gt_map[case]),
            "pred_file": str(pred_map[case]),
        }
        row["psnr"] = metrics.psnr(gt, pred)
        row["ssim"] = metrics.ssim(gt, pred)
        row["lpips"] = metrics.lpips_score(gt, pred, max_side=args.lpips_max_side)

        # pred 相对 GT 的残留退化构成
        fp = degradation.residual_fingerprint(pred, gt)
        row["pred_rms"] = fp.rms
        row["pred_hf_ratio"] = fp.hf_ratio
        row["pred_edge_res"] = fp.edge_res
        row["pred_flat_res"] = fp.flat_res
        row["pred_edge_flat_ratio"] = fp.edge_flat_ratio
        row["pred_cast_norm"] = fp.cast_norm
        p_profile = degradation.degradation_profile(fp)
        row["pred_dominant"] = p_profile["label"]          # 完整标签,如 "重度·模糊+偏色"
        row["pred_severity"] = p_profile["severity"]
        row["pred_components"] = "+".join(p_profile["components"]) or "无"
        row["pred_structure"] = p_profile["dominant"] or "无"
        row.update({f"pred_ch_{k}": v for k, v in zip("rgb", fp.channel_means)})

        # LQ 相对 GT 的原始退化构成(提供 --lq-dir 时)
        if args.lq_dir and case in lq_map:
            lq = load_image(lq_map[case])
            if lq.shape == gt.shape:
                fp_lq = degradation.residual_fingerprint(lq, gt)
                row["lq_rms"] = fp_lq.rms
                row["lq_hf_ratio"] = fp_lq.hf_ratio
                row["lq_edge_flat_ratio"] = fp_lq.edge_flat_ratio
                row["lq_cast_norm"] = fp_lq.cast_norm
                lq_profile = degradation.degradation_profile(fp_lq)
                row["lq_dominant"] = lq_profile["label"]
                row["lq_severity"] = lq_profile["severity"]
                row["lq_components"] = "+".join(lq_profile["components"]) or "无"
                row["lq_structure"] = lq_profile["dominant"] or "无"
                row["delta_rms"] = round(fp.rms - fp_lq.rms, 4)
                row["delta_cast_norm"] = round(fp.cast_norm - fp_lq.cast_norm, 4)
        rows.append(row)
        print(f"  [{i}/{len(cases)}] case{case}: "
              f"PSNR={row['psnr']:.2f} SSIM={row['ssim']:.4f} "
              f"LPIPS={row['lpips']:.4f} 残留={row['pred_dominant']}")

    if not rows:
        print("错误: 没有成功评估任何 case。")
        return 2

    out_dir = _outdir_for("paired", args.out_dir)
    csv_path = report.save_csv(rows, out_dir / "paired_report.csv")
    numeric_cols = [
        "psnr", "ssim", "lpips",
        "pred_rms", "pred_hf_ratio", "pred_edge_res", "pred_flat_res",
        "pred_edge_flat_ratio", "pred_cast_norm",
        "lq_rms", "lq_hf_ratio", "lq_edge_flat_ratio", "lq_cast_norm",
        "delta_rms", "delta_cast_norm",
    ]
    summary = report.summarize(rows, numeric_cols)
    summary["n_paired"] = len(rows)
    summary["gt_dir"] = str(args.gt_dir)
    summary["pred_dir"] = str(args.pred_dir)
    summary["lq_dir"] = str(args.lq_dir) if args.lq_dir else None
    report.save_json(summary, out_dir / "paired_summary.json")

    print("\n=== 逐 case 明细 ===")
    report.print_table(rows)
    print("\n=== 汇总 ===")
    print(f"配对数量: {len(rows)}")
    for col in ["psnr", "ssim", "lpips"]:
        s = summary[col]
        print(f"  {col}: mean={s['mean']}  std={s['std']}  "
              f"min={s['min']}  max={s['max']}")
    print(f"\n结果已保存: {csv_path}")
    print(f"            {out_dir / 'paired_summary.json'}")
    return 0


# ---------------------------------------------------------------------------
# nr(无参考,无 GT)
# ---------------------------------------------------------------------------
def cmd_nr(args: argparse.Namespace) -> int:
    paths = collect_images(Path(args.dir))
    if args.limit:
        paths = paths[: args.limit]
    metrics_cfg = args.metric.split(",") if args.metric else ["niqe"]
    models = NRMetricSet(models=metrics_cfg, device=args.device)

    ref_map = make_case_map(collect_images(Path(args.dir2))) if args.dir2 else {}
    print(f"评估 {len(paths)} 张图(无参考),指标: {metrics_cfg} ...")
    rows = []
    for i, p in enumerate(paths, 1):
        img = load_image(p)
        row = {"file": str(p)}
        hf = heuristic_fingerprint(img)
        row.update(hf)
        for name in models.available():
            row[name] = models.score(name, img)
        if args.dir2:
            n = extract_case_number(p.stem)
            if n is not None and n in ref_map:
                ref = load_image(ref_map[n])
                for name in models.available():
                    row[f"delta_{name}"] = round(models.score(name, ref) - row[name], 4)
        rows.append(row)
        print(f"  [{i}/{len(paths)}] {p.name}: " +
              " ".join(f"{k}={row[k]:.3f}" for k in metrics_cfg))

    out_dir = _outdir_for("nr", args.out_dir)
    csv_path = report.save_csv(rows, out_dir / "nr_report.csv")
    numeric_cols = list(metrics_cfg) + [f"delta_{k}" for k in metrics_cfg]
    summary = report.summarize(rows, numeric_cols)
    summary["n_images"] = len(rows)
    summary["dir"] = str(args.dir)
    summary["dir2"] = str(args.dir2) if args.dir2 else None
    summary["metrics"] = metrics_cfg
    report.save_json(summary, out_dir / "nr_summary.json")

    print("\n=== 逐图明细 ===")
    report.print_table(rows)
    print("\n=== 汇总 ===")
    print(f"图片数量: {len(rows)}")
    for col in numeric_cols:
        if col in summary:
            s = summary[col]
            print(f"  {col}: mean={s['mean']}  std={s['std']}  "
                  f"min={s['min']}  max={s['max']}")
    print(f"\n结果已保存: {csv_path}")
    print(f"            {out_dir / 'nr_summary.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate_tool",
        description="HYPIR 赛题批量分析与评估工具链(paired 成对 / nr 无参考)。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_paired = sub.add_parser("paired", help="成对评估: PSNR/SSIM/LPIPS + 残差指纹")
    p_paired.add_argument("--gt-dir", required=True, help="GT 图目录(如 Data/competition/evaluate)")
    p_paired.add_argument("--pred-dir", required=True, help="恢复结果目录(如 Data/competition/Result/<run>)")
    p_paired.add_argument("--lq-dir", default=None, help="可选: LQ 图目录,提供原始退化构成与恢复前后 delta")
    p_paired.add_argument("--gt-suffix", default="_gt",
                          help="GT 文件名包含子串(默认 '_gt',区分同目录 GT/LQ)")
    p_paired.add_argument("--lq-suffix", default="_lq",
                          help="LQ 文件名包含子串(默认 '_lq')")
    p_paired.add_argument("--pred-suffix", default=None,
                          help="pred 文件名包含子串(默认不过滤)")
    p_paired.add_argument("--lpips-max-side", type=int, default=1024,
                          help="LPIPS 计算前最长边缩放上限(默认 1024,控制 4K 图开销)")
    p_paired.add_argument("--limit", type=int, default=None, help="只评估前 N 个 case(调试用)")
    p_paired.add_argument("--out-dir", default=None, help="输出目录(默认 evaluate_output/paired_<时间戳>)")
    p_paired.set_defaults(func=cmd_paired)

    p_nr = sub.add_parser("nr", help="无参考评估: NIQE/MANIQA/MUSIQ + 单图退化指纹")
    p_nr.add_argument("--dir", required=True, help="待评估图像目录(如测试集或结果目录)")
    p_nr.add_argument("--dir2", default=None, help="可选: 对比目录(按序号配对,输出指标相对提升 delta)")
    p_nr.add_argument("--metric", default="niqe,maniqa,musiq",
                      help="逗号分隔指标: niqe / maniqa / musiq(默认 niqe,maniqa,musiq全部启用;"
                           "maniqa/musiq 需要本地权重 HYPIR_model/MANIQA.pt、MUSIQ.pth)")
    p_nr.add_argument("--device", default=None, help="cuda/cpu(默认自动)")
    p_nr.add_argument("--limit", type=int, default=None, help="只评估前 N 张(调试用)")
    p_nr.add_argument("--out-dir", default=None, help="输出目录(默认 evaluate_output/nr_<时间戳>)")
    p_nr.set_defaults(func=cmd_nr)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - CLI 顶层统一报错
        print(f"错误: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
