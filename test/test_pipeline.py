# -*- coding: utf-8 -*-
"""HYPIR 基线数据合成管线 —— 主链路冒烟测试。

运行方式（在仓库根目录）：
    python test/test_pipeline.py

覆盖：
    - smoothstep 正/逆往返
    - 逆 WB 数学不变量 (f(t,g)=t/g, f(1,g)=1)
    - Bresenham 运动模糊核
    - 约束拓扑序列预生成与保序选择
    - 传感器噪声 (归一化 beta 域)
    - double_jpeg
    - 离线预处理 (run_preprocess -> manifest -> 路径解析)
    - SceneDegradation 全链路 (lq/gt shape, 值域, 约束顺序, debug, 确定性)
"""

import os
import sys
import json
import shutil
import tempfile

import numpy as np
import cv2

# 项目根 = test/ 的上一级目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml
from HYPIR.dataset.scene_degradation import (
    smoothstep_forward,
    smoothstep_inverse,
    highlight_preserving_inverse,
    generate_motion_kernel,
    apply_sensor_noise,
    apply_dark_region_noise_boost,
    double_jpeg,
    validate_constraints,
    pregenerate_sequences,
    SceneDegradation,
    DEFAULT_STAGES,
    _select_sequence,
    resolve_path,
)
from HYPIR.dataset.preprocess import run_preprocess, generate_flare_templates

PASS = []


def ok(name, cond, extra=""):
    PASS.append(bool(cond))
    print(("[PASS] " if cond else "[FAIL] ") + name, extra)


def main():
    with open(os.path.join(ROOT, "configs", "degradation_baseline.yaml")) as f:
        cfg = yaml.safe_load(f)

    # 1. smoothstep roundtrip
    x = np.linspace(0, 1, 1000, dtype=np.float32)
    y = smoothstep_forward(x)
    xr = smoothstep_inverse(y)
    ok("smoothstep roundtrip",
       float(np.max(np.abs(x - xr))) < 1e-4,
       f"err={float(np.max(np.abs(x - xr))):.2e}")

    # 2. invWB invariants
    for g in [1.0, 2.35, 2.6, 0.9]:
        xs = np.array([0.0, 0.3, 0.9, 1.0], dtype=np.float32)
        r = highlight_preserving_inverse(xs, g)
        ok(f"invWB g={g} f(t,g)=t/g", abs(float(r[2]) - 0.9 / g) < 1e-4,
           f"f(0.9,g)={float(r[2]):.5f}")
        ok(f"invWB g={g} f(1,g)=1", abs(float(r[3]) - 1.0) < 1e-4, "")

    # 3. motion kernel
    k = generate_motion_kernel(20, 37)
    ok("motion kernel sum=1", abs(float(k.sum()) - 1.0) < 1e-6,
       f"sum={float(k.sum()):.4f}")
    ok("motion kernel nonzero", k.max() > 0, "")

    # 3b. 几何回归（P0-2）：线段质心必须落在核中心（避免卷积平移）
    def _centroid(kk):
        h, w = kk.shape
        yy, xx = np.mgrid[0:h, 0:w]
        s = kk.sum()
        return (yy * kk).sum() / s, (xx * kk).sum() / s, (h - 1) / 2.0, (w - 1) / 2.0

    max_off = 0.0
    for th in [0, 10, 30, 44, 45, 60, 90]:
        kk = generate_motion_kernel(20, float(th))
        cy, cx, ey, ex = _centroid(kk)
        max_off = max(max_off, abs(cy - ey), abs(cx - ex))
    ok("motion kernel anchor aligned (P0-2)", max_off <= 1.0, f"maxoff={max_off:.2f}")

    validate_constraints(cfg["scheduler"]["constraints"], DEFAULT_STAGES)
    ok("pregenerate seq",
       len(pregenerate_sequences(DEFAULT_STAGES, cfg["scheduler"]["constraints"],
                                 50, np.random.default_rng(0))) >= 1, "")
    seqs = pregenerate_sequences(DEFAULT_STAGES, cfg["scheduler"]["constraints"],
                                 50, np.random.default_rng(0))
    sel = _select_sequence(seqs, ["sensor_noise", "forward_isp", "zoom_jpeg"],
                           np.random.default_rng(5))
    ok("select preserves order", sel == ["sensor_noise", "forward_isp", "zoom_jpeg"],
       str(sel))

    # 4. sensor noise
    rng = np.random.default_rng(7)
    I = np.full((32, 32, 3), 0.5, dtype=np.float32)
    b2 = apply_dark_region_noise_boost(I, 1e-5, rng, cfg)
    noisy = apply_sensor_noise(I, 1e-5, b2[..., 0], rng)
    ok("sensor noise shape/range",
       noisy.shape == (32, 32, 3) and noisy.min() >= 0 and noisy.max() <= 1,
       f"min={noisy.min():.3f} max={noisy.max():.3f}")

    # 5. double_jpeg
    jg = double_jpeg(I, rng, cfg)
    ok("double_jpeg shape/range",
       jg.shape == (32, 32, 3) and jg.min() >= 0 and jg.max() <= 1, "")

    # 6. full pipeline on synthetic GT
    tmp = tempfile.mkdtemp(prefix="hypir_test_")
    try:
        gt_dir = os.path.join(tmp, "gt")
        os.makedirs(gt_dir)
        out_dir = os.path.join(tmp, "preprocessed")
        os.makedirs(out_dir, exist_ok=True)

        yy, xx = np.mgrid[0:512, 0:512]
        base = np.stack([xx / 512.0, yy / 512.0, np.full((512, 512), 0.2)],
                        axis=-1).astype(np.float32)
        gauss = np.exp(-((xx - 300) ** 2 + (yy - 300) ** 2) / (2 * 40.0 ** 2))
        base += gauss[..., None] * 0.5
        base = np.clip(base, 0, 1)
        gt8 = (base * 255).astype(np.uint8)
        gt_path = os.path.join(gt_dir, "00001.png")
        cv2.imwrite(gt_path, cv2.cvtColor(gt8, cv2.COLOR_RGB2BGR))

        run_preprocess(gt_dir, out_dir, cfg)
        mpath = os.path.join(out_dir, "manifest.json")
        ok("manifest exists", os.path.exists(mpath))
        with open(mpath) as f:
            man = json.load(f)
        ok("manifest linear_raw present",
           man["files"][0]["linear_raw"] is not None, man["files"][0]["linear_raw"])
        mdir = os.path.dirname(os.path.abspath(mpath))
        lr = resolve_path(mdir, man["files"][0], "linear_raw")
        ok("resolve linear_raw exists", lr is not None and os.path.exists(lr), str(lr))

        flare_dir = os.path.join(out_dir, "flare_templates", "512")
        generate_flare_templates(flare_dir, num_templates=4, size=512)
        ok("flare templates generated", len(os.listdir(flare_dir)) == 4, "")

        sd = SceneDegradation(cfg, mpath, base_seed=42)
        res = sd("00001", source="gt", debug=True)
        lq, gt = res["lq"], res["gt"]
        ok("SceneDegrad lq/gt shape",
           lq.shape == (512, 512, 3) and gt.shape == (512, 512, 3), "")
        ok("SceneDegrad lq/gt range",
           lq.min() >= 0 and lq.max() <= 1 and gt.min() >= 0 and gt.max() <= 1, "")
        ok("lq != gt", not np.array_equal(lq, gt), "")
        ok("sample_params present",
           "stages_run" in res["sample_params"] and "vc" in res["sample_params"], "")
        ok("debug populated",
           res["debug"] is not None and len(res["debug"]) > 0,
           str(list(res["debug"].keys())))

        order = res["sample_params"]["stages_run"]

        def si(o, s):
            return o.index(s) if s in o else -1

        for c in cfg["scheduler"]["constraints"]:
            ia, ib = si(order, c["from"]), si(order, c["before"])
            ok(f"order {c['from']} < {c['before']}",
               ia == -1 or ib == -1 or ia < ib, str(order))

        # 7. determinism with same rng
        sd2 = SceneDegradation(cfg, mpath, base_seed=42)
        a = sd2("00001", np.random.default_rng(123))
        b = sd2("00001", np.random.default_rng(123))
        ok("deterministic same rng",
           np.array_equal(np.round(a["lq"], 6), np.round(b["lq"], 6)), "")
    finally:
        shutil.rmtree(tmp)

    print("\nSUMMARY:", sum(PASS), "/", len(PASS), "passed")
    print("ALL_PASS" if all(PASS) else "SOME_FAIL")
    return 0 if all(PASS) else 1


if __name__ == "__main__":
    sys.exit(main())
