# -*- coding: utf-8 -*-
"""小批量冒烟测试：OSS 压缩包解压 -> 选取少量数据 -> 预处理(zip批次) -> 训练读取验证 -> 可选短训。

用于：
  1. 小批量测试完整预处理-训练读取链路（zip 批次产物格式、manifest、zip 读取 API、退化合成契约）；
  2. 选择合适 zip batch 大小（--batch-trials 定时对比，或人工多次运行观察 zip 大小/时间/内存）。

实例上用法（在仓库根目录运行）：
    python tools/smoke_pipeline.py \
        --zip /mnt/data/merged_512.zip \
        --num-samples 64 --batch-size 64 --workers 8

    # 对比多个 batch 大小的预处理耗时/产物大小
    python tools/smoke_pipeline.py --num-samples 128 --batch-trials 32,64,128,256

    # 跑 4 步真实训练（场景线：SceneDegradationDataset 跑 zip 产物；首次会下载 SD2 base 权重约 5GB）
    python tools/smoke_pipeline.py --num-samples 64 --batch-size 64 --train-steps 4 \
        --train-backend scene

说明：
  - 只解压选中样本到本地 work-dir（不解压整个 zip），系统盘占用 ≈ 少量 PNG + staging；
  - zi 预处理 zip 批次模式产物位于 <work-dir>/preprocessed/（batches/*.zip + manifest.json），
    可直接同步到 OSS 作为训练数据；
  - 训练读取验证使用 ZipBatchStore + SceneDegradation（含 zero-degradation 往返检查）。
"""

import argparse
import io
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def log(msg):
    print(f"[smoke] {msg}", flush=True)


def run_subprocess(cmd, cwd, desc, env=None):
    """流式运行子进程：实时打印 stdout（train.py 卡住时能立即看到卡在哪一步）。"""
    log(f"运行: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=env)
    out_lines = []
    for line in proc.stdout:
        print(f"    | {line.rstrip()}", flush=True)
        out_lines.append(line)
    proc.wait()
    dt = time.time() - t0
    out_text = ''.join(out_lines)
    if proc.returncode != 0:
        raise RuntimeError(f"{desc} 失败（exit={proc.returncode}），完整输出见上方")
    log(f"{desc} 完成，耗时 {dt:.1f}s")
    return dt, out_text


def check_train_prereq(env):
    """预检训练前提：SD2 base 模型本地是否可用，避免 train.py 联网下载卡住。"""
    import glob
    # ① 仓库内本地 Diffusers 模型目录（优先）
    local_model = os.path.join(REPO_ROOT, "HYPIR_model", "sd2-1-base")
    if os.path.isfile(os.path.join(local_model, "model_index.json")):
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        log(f"提示: 使用本地 SD2 base（{local_model}），已设离线模式，无需下载")
        return True
    # ② 判别器 backbone（open_clip convnext_xxlarge）本地文件检查
    conx = os.path.join(REPO_ROOT, "HYPIR_model", "convnext_xxlarge", "open_clip_pytorch_model.bin")
    if os.path.isfile(conx):
        log(f"提示: 判别器 convnext_xxlarge 本地权重就绪（{conx}）")
    else:
        log("⚠ 提示: 未找到判别器 backbone 本地权重（HYPIR_model/convnext_xxlarge/"
            "open_clip_pytorch_model.bin），训练时会在 init_discriminator 联网下载约数 GB；"
            "建议手动下载 open_clip_pytorch_model.bin（源 laion/CLIP-convnext_xxlarge-"
            "laion2B-s34B-b82K-augreg-soup）放到该位置。")
    # ③ HuggingFace 缓存
    hub_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    base_pattern = os.path.join(hub_dir, "models--stabilityai--stable-diffusion-2-1-base", "snapshots", "*")
    if bool(glob.glob(base_pattern)):
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        log("提示: SD2 base 已在 HF 缓存，使用离线模式加载（HF_HUB_OFFLINE=1）")
        return True
    log("⚠ 提示: 未找到本地 SD2 base（HYPIR_model/sd2-1-base 或 HF 缓存），"
        "train.py 将联网下载约 5GB；若网络不可达/超时会导致卡住。")
    return False


# ---------------------------------------------------------------------------- #
# 1. 选取样本
# ---------------------------------------------------------------------------- #

def select_and_extract(zip_path, work_dir, num_samples, seed):
    """从 zip 中随机选取 N 张 GT PNG 解压到本地。返回选中的文件名列表。"""
    gt_dir = os.path.join(work_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)

    log(f"扫描压缩包: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        entries = zf.namelist()
        # merged_512.zip 即切片图集：任意图片条目视为 GT 图（条目常为 'gt/<id>.png'，
        # 不能用要求嵌套路径的 '/gt/' 子串匹配）
        png_entries = [e for e in entries if e.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not png_entries:
            raise RuntimeError(
                f"压缩包中未找到图片条目（共 {len(entries)} 个条目），示例: {entries[:5]}")
        rng = random.Random(seed)
        chosen = rng.sample(png_entries, min(num_samples, len(png_entries)))

        names = []
        for e in chosen:
            base = os.path.basename(e)
            with zf.open(e) as src, open(os.path.join(gt_dir, base), "wb") as dst:
                shutil.copyfileobj(src, dst)
            names.append(base)

        # 可选：还原 tiles.csv 并过滤到选中样本
        csv_entries = [e for e in entries if e.endswith("tiles.csv")]
        if csv_entries:
            with zf.open(csv_entries[0]) as src, open(os.path.join(work_dir, "tiles.csv"), "wb") as dst:
                shutil.copyfileobj(src, dst)

    small_csv = os.path.join(work_dir, "tiles_small.csv")
    if os.path.exists(os.path.join(work_dir, "tiles.csv")):
        import csv
        chosen_set = {n[:-4] for n in names}  # stem
        with open(os.path.join(work_dir, "tiles.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with open(small_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "dataset", "src_path", "src_w", "src_h", "tile_x", "tile_y"])
            for r in rows:
                stem = os.path.splitext(os.path.basename(r["src_path"]))[0]
                if stem in chosen_set:
                    w.writerow([r["id"], r["dataset"], r["src_path"], r["src_w"],
                                r["src_h"], r["tile_x"], r["tile_y"]])
        log(f"溯源子集: {small_csv}")

    log(f"已解压 {len(names)} 张到 {gt_dir}（来源: {zip_path}）")
    return names


# ---------------------------------------------------------------------------- #
# 2. 预处理（zip 批次模式）
# ---------------------------------------------------------------------------- #

def run_preprocessing(work_dir, batch_size, workers, zip_compress=1):
    gt_dir = os.path.join(work_dir, "gt")
    out_dir = os.path.join(work_dir, "preprocessed")
    cmd = [
        sys.executable, "-m", "HYPIR.dataset.preprocess",
        "--gt_dir", gt_dir,
        "--output_dir", out_dir,
        "--workers", str(workers),
        "--zip-batch-size", str(batch_size),
        "--zip-compress", str(zip_compress),
    ]
    dt, _ = run_subprocess(cmd, cwd=REPO_ROOT, desc="预处理(zip 批次)")
    totals = {}
    zips = sorted(glob_zips(out_dir))
    for zp in zips:
        totals[zp] = os.path.getsize(zp)
    log(f"预处理完成: {len(zips)} 个 zip, 共 {sum(totals.values()) / 1e6:.1f} MB, 耗时 {dt:.1f}s")
    return out_dir, dt


def glob_zips(out_dir):
    import glob
    return sorted(glob.glob(os.path.join(out_dir, "batches", "batch_*.zip")))


# ---------------------------------------------------------------------------- #
# 3. 训练读取验证（zip 读取 + SceneDegradation 合成契约）
# ---------------------------------------------------------------------------- #

def validate_zip_flow(work_dir, expected_names):
    from HYPIR.dataset.zip_store import ZipBatchStore
    import yaml
    from HYPIR.dataset.scene_degradation import SceneDegradation, load_yaml, forward_isp

    manifest_path = os.path.join(work_dir, "preprocessed", "manifest.json")
    store = ZipBatchStore(manifest_path)
    assert len(store) == len(expected_names), (len(store), len(expected_names))
    log(f"manifest 样本数: {len(store)} ✓")

    zips = store.batch_zip_paths()
    assert zips, "无批次 zip"
    # 每个 zip 的 npy 数量校验
    for zp in zips:
        with zipfile.ZipFile(zp) as zf:
            n = sum(1 for x in zf.namelist() if x.endswith(".npy"))
        names_in = store.batch_names(zp)
        assert n == len(names_in), f"{os.path.basename(zp)}: zip 内 {n} != manifest {len(names_in)}"
    log(f"批次 zip 对账: {len(zips)} 个 ✓（细粒度: "
        f"{', '.join(os.path.basename(z) + ':' + str(len(store.batch_names(z))) for z in zips)}）")

    # 单样本 zip 读取 + 完整退化合成契约
    cfg = load_yaml(os.path.join(REPO_ROOT, "configs", "degradation_baseline.yaml"))
    dataset = SceneDegradation(cfg, manifest_path, base_seed=42,
                               linear_raw_loader=store.load_linear_raw)
    rng = np.random.default_rng(0)
    for name in random.Random(0).sample(store.names, min(4, len(store.names))):
        res = dataset(name, rng, source="gt")
        lq, gt = res["lq"], res["gt"]
        assert lq.shape == gt.shape == (512, 512, 3), lq.shape
        assert lq.dtype == gt.dtype == np.float32
        assert 0.0 <= lq.min() and lq.max() <= 1.0
        assert float(gt.max()) > 0.01
        log(f"  {name}: lq/gt {lq.shape} float32 [0,1] 非全黑 ✓ "
            f"stages={','.join(res['sample_params']['stages_run'])}")

    # zero-degradation 往返（npy 与 GT 一致性）
    import copy
    base_cfg = load_yaml(os.path.join(REPO_ROOT, "configs", "degradation_baseline.yaml"))
    cfg0 = copy.deepcopy(base_cfg)
    for k in ["motion_blur", "zoom", "jpeg", "flare"]:
        cfg0[k]["prob"] = 0.0
    cfg0["sensor"]["chroma_noise"]["enable"] = False
    cfg0["sensor"]["dark_region_noise_boost"]["enable"] = False
    for k in ["vignetting", "sharpening", "color_cast"]:
        cfg0["forward_isp"][k]["prob"] = 0.0
    cfg0["forward_isp"]["white_balance"]["perturbation_prob"] = 0.0
    cfg0["forward_isp"]["color_correction"]["perturbation_prob"] = 0.0
    from HYPIR.dataset.scene_degradation import load_srgb
    isp_cfg = load_yaml(os.path.join(REPO_ROOT, "configs", "isp_huawei_p60pro.yaml"))
    M = np.array(isp_cfg["forward_ccm"], dtype=np.float32)
    wb = isp_cfg["forward_isp"]["white_balance"]
    vc_fixed = {"ccm_matrix": M, "wb_gains": np.array(
        [float(np.mean(wb["r_gain_range"])), wb["g_gain"],
         float(np.mean(wb["b_gain_range"]))], dtype=np.float32)}
    name = store.names[0]
    I = store.load_linear_raw(name)
    rec = store.record(name)
    gt = load_srgb(rec["gt_path"])
    rec_lin = forward_isp(I, dict(vc_fixed), np.random.default_rng(42), cfg0)
    rmse = float(np.sqrt(np.mean((gt - rec_lin) ** 2)))
    psnr = 10 * np.log10(1.0 / (rmse ** 2 + 1e-12))
    assert psnr >= 50.0, psnr
    log(f"zip 读取 -> 零退化往返: PSNR {psnr:.2f} dB（>=50 ✓）")
    return True


# ---------------------------------------------------------------------------- #
# 4. 可选真实短训（SD2 + RealESRGANDataset）
# ---------------------------------------------------------------------------- #

def run_short_training(work_dir, num_steps, backend="scene"):
    import yaml
    if backend == "scene":
        # 场景退化合成线：SceneDegradationDataset -> SD2Trainer（zip 批次产物入口）
        with open(os.path.join(REPO_ROOT, "configs", "sd2_scene_train.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg["output_dir"] = os.path.join(work_dir, "train_scene_out")
        cfg["logging_dir"] = "logs"
        cfg["max_train_steps"] = int(num_steps)
        cfg["resume_from_checkpoint"] = None
        cfg["data_config"]["train"]["batch_size"] = 2
        cfg["data_config"]["train"]["dataloader_num_workers"] = 2
        cfg["data_config"]["train"]["dataset"]["params"]["manifest_path"] = \
            os.path.join(work_dir, "preprocessed", "manifest.json")
        cfg["data_config"]["train"]["dataset"]["params"]["cfg_path"] = \
            os.path.join(REPO_ROOT, "configs", "degradation_baseline.yaml")
        # 本地 SD2 base + 官方预训练权重（绝对路径，避免 cwd/联网问题）
        cfg["base_model_path"] = os.path.join(REPO_ROOT, "HYPIR_model", "sd2-1-base")
        cfg["g_pretrain_path"] = os.path.join(REPO_ROOT, "HYPIR_model", "HYPIR_sd2.pth")
        cfg["d_pretrain_path"] = os.path.join(REPO_ROOT, "HYPIR_model", "HYPIR_sd2_D.safetensors")
        log(f"场景线训练配置生成（manifest={cfg['data_config']['train']['dataset']['params']['manifest_path']}，"
            f"base_model={cfg['base_model_path']}，g_pretrain={cfg['g_pretrain_path']}）")
    else:
        with open(os.path.join(REPO_ROOT, "configs", "sd2_train.yaml"), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        gt_dir = os.path.join(work_dir, "gt")
        file_list = os.path.join(work_dir, "file_list.txt")
        with open(file_list, "w", encoding="utf-8") as f:
            for n in sorted(os.listdir(gt_dir)):
                if n.endswith(".png"):
                    f.write(os.path.abspath(os.path.join(gt_dir, n)) + "\n")

        cfg["output_dir"] = os.path.join(work_dir, "train_out")
        cfg["logging_dir"] = "logs"
        cfg["max_train_steps"] = int(num_steps)
        cfg["resume_from_checkpoint"] = None
        cfg["data_config"]["train"]["batch_size"] = 2
        cfg["data_config"]["train"]["dataloader_num_workers"] = 2
        cfg["data_config"]["train"]["dataset"]["params"]["file_meta"] = {
            "file_list": file_list,
            "image_path_prefix": "",
            "image_path_key": "path",
            "prompt_key": "prompt",
        }
        cfg["data_config"]["train"]["dataset"]["params"]["crop_type"] = "none"

    train_cfg = os.path.join(work_dir, "train_config.yaml")
    with open(train_cfg, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None)

    log(f"生成训练配置: {train_cfg}（backend={backend}, max_train_steps={num_steps}）")
    # 预检模型缓存/网络，并传环境变量加载
    env = dict(os.environ)
    check_train_prereq(env)
    dt, _ = run_subprocess(
        [sys.executable, "train.py", "--config", train_cfg],
        cwd=REPO_ROOT, desc="短训（train.py）", env=env)
    log(f"短训完成: {dt:.1f}s，产物在 {cfg['output_dir']}")
    log("提示: 训练日志中的 'VRAM peak' 可用于评估 batch/zip 大小对显存的影响")
    return True


# ---------------------------------------------------------------------------- #
# 5. batch 大小对比试验
# ---------------------------------------------------------------------------- #

def run_batch_trials(work_dir, trials, workers, zip_compress=1):
    results = []
    for b in trials:
        trial_dir = os.path.join(work_dir, "trials", f"batch_{b}")
        os.makedirs(trial_dir, exist_ok=True)
        cmd = [
            sys.executable, "-m", "HYPIR.dataset.preprocess",
            "--gt_dir", os.path.join(work_dir, "gt"),
            "--output_dir", os.path.join(trial_dir, "preprocessed"),
            "--workers", str(workers),
            "--zip-batch-size", str(b),
            "--zip-compress", str(zip_compress),
        ]
        t0 = time.time()
        _, _ = run_subprocess(cmd, cwd=REPO_ROOT, desc=f"batch={b} 预处理")
        dt = time.time() - t0
        zips = glob_zips(os.path.join(trial_dir, "preprocessed"))
        size = sum(os.path.getsize(z) for z in zips)
        results.append((b, len(zips), size / 1e6, dt))
    log("\n=== batch 大小对比 ===")
    log(f"{'batch':>8} {'zip数':>6} {'总大小MB':>12} {'耗时s':>10} {'staging峰值GB(约)':>18}")
    for b, nz, size, dt in results:
        staging_gb = (2 + 2) * b * 3.2 / 1024  # zip_workers=2 时约 4 批
        log(f"{b:>8} {nz:>6} {size:>12.1f} {dt:>10.1f} {staging_gb:>18.1f}")
    log("说明: staging 峰值 ≈ (zip_workers+2)*batch*3.2MB；zip 数少则 OSS 小文件少、上传更省")
    return results


# ---------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="OSS 数据集小批量冒烟：解压采样 -> 预处理(zip批次) -> 训练读取验证 -> 可选短训")
    parser.add_argument("--zip", default="/mnt/data/merged_512.zip",
                        help="merged_512 压缩包路径（默认 /mnt/data/merged_512.zip）")
    parser.add_argument("--work-dir", default="/tmp/hypir_smoke",
                        help="本地工作目录（系统盘; 默认 /tmp/hypir_smoke）")
    parser.add_argument("--num-samples", type=int, default=64, help="采样张数（默认 64）")
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="预处理 zip 批次大小（每 zip 样本数; 默认 64）")
    parser.add_argument("--batch-trials", default=None,
                        help="对比多个 batch 大小, 逗号分隔, 如 32,64,128,256")
    parser.add_argument("--workers", type=int, default=8, help="预处理并行进程数")
    parser.add_argument("--zip-compress", type=int, default=1, choices=range(0, 10),
                        help="zip 压缩级别（默认 1）")
    parser.add_argument("--train-steps", type=int, default=0,
                        help=">0 时运行 train.py 短训 N 步（首次需下载 SD2 base 权重约 5GB）")
    parser.add_argument("--train-backend", choices=["scene", "realesrgan"], default="scene",
                        help="短训数据入口：scene=SceneDegradationDataset(zip 批次, 默认)；"
                             "realesrgan=RealESRGANDataset(解压 GT 在线退化)")
    parser.add_argument("--clean", action="store_true",
                        help="验证通过后清理 work-dir 下的 gt 与 trials")
    args = parser.parse_args()

    if not os.path.exists(args.zip):
        raise SystemExit(f"找不到压缩包: {args.zip}")
    os.makedirs(args.work_dir, exist_ok=True)

    log(f"zip={args.zip}  work-dir={args.work_dir}  samples={args.num_samples}")
    names = select_and_extract(args.zip, args.work_dir, args.num_samples, args.seed)

    if args.batch_trials:
        trials = [int(x) for x in args.batch_trials.split(",") if x.strip()]
        run_batch_trials(args.work_dir, trials, args.workers, args.zip_compress)

    # 正式冒烟：选定 batch 跑预处理 + 训练读取验证
    out_dir, dt = run_preprocessing(args.work_dir, args.batch_size, args.workers,
                                    args.zip_compress)
    validate_zip_flow(args.work_dir, names)
    log(f"预处理 zip 批次模式验证通过: {out_dir}")

    if args.train_steps > 0:
        run_short_training(args.work_dir, args.train_steps, args.train_backend)
    else:
        log("跳过真实训练（加 --train-steps 4 --train-backend scene 可跑 4 步场景线短训）")

    if args.clean:
        shutil.rmtree(os.path.join(args.work_dir, "gt"), ignore_errors=True)
        shutil.rmtree(os.path.join(args.work_dir, "trials"), ignore_errors=True)
        log(f"已清理 gt/trials（保留 preprocessed/ 供检查）")

    log("全部完成 ✓")
    log(f"关键产物: {os.path.join(args.work_dir, 'preprocessed')}")
    log("下一步建议: 将 preprocessed/ 同步到 OSS（batches/*.zip + manifest.json），"
        "后续训练用 ZipBatchStore 读取")


if __name__ == "__main__":
    main()
