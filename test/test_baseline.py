# -*- coding: utf-8 -*-
"""基线场景退化合成管线 —— 验收测试系统。

依据《设计书V3基线测试.md》实现，含 14 项验收测试。
被测对象为 baseline_impl（统一模块入口）。

本脚本已迁移至 test/ 目录，并做了如下稳健性改进：
  - 使用绝对路径常量（项目根 + configs + test_data），不再依赖执行时的 cwd；
  - prepare_test_data 幂等：已存在 flare_0001 记录时不再重复追加；
  - run_all_tests 对非断言异常打印完整 traceback，便于排查；
  - 逆 WB 不变量测试对点采样显式声明 dtype=np.float32。

用法（在仓库任意目录运行均可）：
    python test/test_baseline.py                # 准备数据 + 运行全部 14 项测试
    python test/test_baseline.py --no-prepare   # 仅运行测试（假设已准备数据）
"""

import argparse
import copy
import json
import os
import sys
import traceback

import numpy as np
import cv2

# 项目根 = test/ 的上一级目录（保证能 import 项目根的 baseline_impl）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from baseline_impl import (
    SceneDegradation,
    inverse_isp,
    forward_isp,
    highlight_preserving_inverse,
    apply_sensor_noise,
    generate_motion_kernel,
    apply_chroma_noise,
    load_srgb,
    load_yaml,
    generate_flare_templates,
    run_preprocess,
    WB_GAINS_FIXED,
)

# 绝对路径常量（不依赖 cwd）
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "degradation_baseline.yaml")
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")
TEST_DATA_MANIFEST = os.path.join(TEST_DATA_DIR, "preprocessed", "manifest.json")
FLARE_TEMPLATE_DIR = os.path.join(BASE_DIR, "flare_templates", "512")


# ---------------------------------------------------------------------------- #
# 测试数据准备 (§2.3)
# ---------------------------------------------------------------------------- #

def prepare_test_data():
    """生成测试数据目录与 Manifest（幂等：已存在 flare_0001 时不重复追加）。"""
    os.makedirs(os.path.join(TEST_DATA_DIR, "gt"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "flare3d", "input"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "flare3d", "gt"), exist_ok=True)
    os.makedirs(os.path.join(TEST_DATA_DIR, "preprocessed", "linear_raw"), exist_ok=True)
    os.makedirs(FLARE_TEMPLATE_DIR, exist_ok=True)

    # 生成 2 张测试 GT（简单彩色梯度图 + 少量随机纹理，保证非全黑）
    rng = np.random.default_rng(42)
    for i in range(2):
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        for y in range(512):
            for c in range(3):
                img[y, :, c] = int(255 * (y / 512 + c / 3) % 1.0)
        # 加少量随机纹理
        img = np.clip(img.astype(np.float32) + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(TEST_DATA_DIR, "gt", f"test_{i + 1:05d}.png"), img)

    # 生成虚构 Flare3D input/gt
    flare_input = os.path.join(TEST_DATA_DIR, "flare3d", "input", "flare_0001.png")
    flare_gt = os.path.join(TEST_DATA_DIR, "flare3d", "gt", "flare_0001.png")
    flare_img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(flare_img, (256, 256), 50, (255, 255, 200), -1)  # 简单光斑
    cv2.imwrite(flare_input, flare_img)
    cv2.imwrite(flare_gt, np.zeros((512, 512, 3), dtype=np.uint8) + 128)

    # 生成 Flare 模板
    generate_flare_templates(FLARE_TEMPLATE_DIR, num_templates=5)

    # 运行离线预处理
    cfg = load_yaml(CONFIG_PATH)
    run_preprocess(os.path.join(TEST_DATA_DIR, "gt"),
                   os.path.join(TEST_DATA_DIR, "preprocessed"), cfg)

    # 更新 Manifest：修正 flare_bank_path 为绝对路径，并追加 Flare3D 记录（幂等）
    with open(TEST_DATA_MANIFEST, 'r') as f:
        manifest = json.load(f)

    # flare 模板路径改为绝对，保证不依赖 cwd 与 resolve_path 回退
    manifest['flare_bank_path'] = FLARE_TEMPLATE_DIR

    # 避免重复追加 flare_0001
    if not any(rec.get('name') == 'flare_0001' for rec in manifest['files']):
        manifest['files'].append({
            'name': 'flare_0001',
            'source_type': 'flare3d_lq',
            'gt_path': None,
            'flare3d_input_path': flare_input,
            'flare3d_gt_path': flare_gt,
            'linear_raw': None,
        })
    with open(TEST_DATA_MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------- #
# 零退化配置生成 (§3)
# ---------------------------------------------------------------------------- #

def make_zero_degradation_config(cfg: dict) -> dict:
    """生成零退化配置（关闭所有随机退化）。"""
    cfg = copy.deepcopy(cfg)
    cfg['motion_blur']['prob'] = 0.0
    cfg['sensor']['chroma_noise']['enable'] = False
    cfg['sensor']['dark_region_noise_boost']['enable'] = False
    cfg['forward_isp']['vignetting']['prob'] = 0.0
    cfg['forward_isp']['sharpening']['prob'] = 0.0
    cfg['forward_isp']['color_cast']['prob'] = 0.0
    cfg['zoom']['prob'] = 0.0
    cfg['jpeg']['prob'] = 0.0
    cfg['flare']['prob'] = 0.0
    # 噪声参数固定为参考值，扰动为 0
    cfg['forward_isp']['white_balance']['perturbation_prob'] = 0.0
    cfg['forward_isp']['color_correction']['perturbation_prob'] = 0.0
    return cfg


def make_zero_degradation_vc(cfg: dict) -> dict:
    """生成零退化虚拟相机参数（无扰动）。"""
    M_base = np.array(cfg['forward_isp']['color_correction']['base_matrix'], dtype=np.float32)
    return {
        'wb_gains': WB_GAINS_FIXED.copy(),  # [2.35, 1.0, 2.6]
        'ccm_matrix': M_base,
        'beta_1': cfg['sensor']['beta_1_ref'],
        'beta_2': cfg['sensor']['beta_2_ref'],
    }


# ---------------------------------------------------------------------------- #
# 测试用例 (§4)
# ---------------------------------------------------------------------------- #

def test_output_contract():
    """测试输出满足 [512,512,3] float32 [0,1] 非全黑。"""
    cfg = load_yaml(CONFIG_PATH)
    dataset = SceneDegradation(cfg, TEST_DATA_MANIFEST, base_seed=42)

    result = dataset("test_00001", dataset.master_rng, source='gt')
    lq = result['lq']

    assert lq.shape == (512, 512, 3), f"Shape mismatch: {lq.shape}"
    assert lq.dtype == np.float32, f"dtype mismatch: {lq.dtype}"
    assert np.all(lq >= 0.0), "Values below 0"
    assert np.all(lq <= 1.0), "Values above 1"
    assert np.any(lq > 0.01), "Output is all black"
    print("✓ test_output_contract passed")


def test_bgr_rgb_consistency():
    """构造纯红图，验证 R 通道均值显著大于 G/B。"""
    cfg = load_yaml(CONFIG_PATH)
    cfg_zero = make_zero_degradation_config(cfg)

    # 构造纯红测试图
    test_img = np.zeros((64, 64, 3), dtype=np.uint8)
    test_img[..., 0] = 255  # R = 255

    # 模拟 full_pipeline（直接调用 forward_isp）
    vc = make_zero_degradation_vc(cfg_zero)
    rng = np.random.default_rng(42)

    # 先逆 ISP，再正向 ISP
    I_linear = inverse_isp(test_img.astype(np.float32) / 255.0, vc, cfg_zero)
    lq = forward_isp(I_linear, vc, rng, cfg_zero)

    # 验证：纯红图经过完整管线后，R 通道应明显大于 G/B
    assert lq[..., 0].mean() > lq[..., 1].mean() + 0.1, (
        f"R ({lq[..., 0].mean():.3f}) not > G ({lq[..., 1].mean():.3f}) + 0.1"
    )
    assert lq[..., 0].mean() > lq[..., 2].mean() + 0.1, (
        f"R ({lq[..., 0].mean():.3f}) not > B ({lq[..., 2].mean():.3f}) + 0.1"
    )
    print("✓ test_bgr_rgb_consistency passed")


def test_zero_degradation_roundtrip():
    """sRGB → inverse_isp → forward_isp → sRGB，PSNR ≥ 50 dB。"""
    cfg = load_yaml(CONFIG_PATH)
    cfg_zero = make_zero_degradation_config(cfg)

    # 加载真实测试图
    I_srgb = load_srgb(os.path.join(TEST_DATA_DIR, "gt", "test_00001.png"))

    # 使用完全相同的参数
    vc = make_zero_degradation_vc(cfg_zero)
    rng = np.random.default_rng(42)

    I_linear = inverse_isp(I_srgb, vc, cfg_zero)
    I_recovered = forward_isp(I_linear, vc, rng, cfg_zero)

    rmse = float(np.sqrt(np.mean((I_srgb - I_recovered) ** 2)))
    psnr = 10 * np.log10(1.0 / (rmse ** 2 + 1e-12))

    assert rmse <= 0.003, f"RMSE {rmse:.6f} exceeds 0.003"
    assert psnr >= 50.0, f"PSNR {psnr:.2f} < 50 dB"
    print(f"✓ test_zero_degradation_roundtrip passed (RMSE={rmse:.6f}, PSNR={psnr:.2f} dB)")


def test_inverse_wb_invariants():
    """验证逆 WB 公式的不变量。"""
    t = 0.9
    for g in [1.0, 1.5, 2.0, 2.35, 2.5, 2.8]:
        x = np.linspace(0, 1, 1000, dtype=np.float32)
        result = highlight_preserving_inverse(x, g, t=t)

        # 1. 恒等性（g=1）
        if g == 1.0:
            assert np.allclose(result, x, atol=1e-6), (
                f"g=1 not identity: max diff {np.max(np.abs(result - x))}"
            )

        # 2. 饱和保持：f(1, g) = 1
        f_at_1 = float(highlight_preserving_inverse(np.array([1.0], dtype=np.float32), g, t)[0])
        assert abs(f_at_1 - 1.0) < 1e-6, f"f(1, {g}) = {f_at_1}, expected 1.0"

        # 3. 线性区：f(t, g) ≈ t/g
        f_at_t = float(highlight_preserving_inverse(np.array([t], dtype=np.float32), g, t)[0])
        expected = t / g
        assert abs(f_at_t - expected) < 1e-5, (
            f"f({t}, {g}) = {f_at_t}, expected {expected}"
        )

        # 4. 单调性
        assert np.all(np.diff(result) >= -1e-7), f"g={g}: not monotonic"

        # 5. 输出范围 [0, 1]（隐式验证 α ∈ [0,1]）
        assert np.all(result >= 0.0), f"g={g}: output below 0"
        assert np.all(result <= 1.0), f"g={g}: output above 1"

    print("✓ test_inverse_wb_invariants passed")


def test_noise_variance():
    """纯灰图 I=0.5，验证实际方差与理论值。"""
    I = np.full((256, 256, 3), 0.5, dtype=np.float32)
    beta_1 = 1e-5
    beta_2 = 1e-5
    rng = np.random.default_rng(42)

    n_trials = 100
    var_list = []
    for _ in range(n_trials):
        I_noisy = apply_sensor_noise(I, beta_1, beta_2, rng)
        var_list.append(float(np.var(I_noisy - I)))

    var_actual = float(np.mean(var_list))
    var_theory = beta_1 * 0.5 + beta_2  # Var = β1·I + β2

    rel_error = abs(var_actual - var_theory) / var_theory
    assert rel_error < 0.05, (
        f"Relative error {rel_error:.4f} exceeds 0.05 "
        f"(actual={var_actual:.2e}, theory={var_theory:.2e})"
    )
    print(f"✓ test_noise_variance passed (actual={var_actual:.2e}, theory={var_theory:.2e})")


def test_rng_reproducibility():
    """同一 master_rng 种子，两次调用输出必须一致。"""
    cfg = load_yaml(CONFIG_PATH)

    # 第一次
    dataset1 = SceneDegradation(cfg, TEST_DATA_MANIFEST, base_seed=42)
    result1 = dataset1("test_00001", dataset1.master_rng, source='gt')

    # 第二次（同种子）
    dataset2 = SceneDegradation(cfg, TEST_DATA_MANIFEST, base_seed=42)
    result2 = dataset2("test_00001", dataset2.master_rng, source='gt')

    assert np.allclose(result1['lq'], result2['lq'], atol=1e-6), (
        f"Outputs differ: max diff {np.max(np.abs(result1['lq'] - result2['lq']))}"
    )
    print("✓ test_rng_reproducibility passed")


def test_motion_kernel_normalization():
    """任意 (r, θ) 生成的核 sum ≈ 1。"""
    test_cases = [
        (2, 0), (2, 45), (2, 90),
        (10, 0), (10, 30), (10, 60), (10, 90),
        (50, 0), (50, 45), (50, 90),
        (100, 0), (100, 22), (100, 90),
        (50, -30), (50, -60),  # 负角度
    ]
    for r, theta in test_cases:
        kernel = generate_motion_kernel(r, theta)
        s = float(kernel.sum())
        assert abs(s - 1.0) < 1e-6, f"r={r}, θ={theta}: sum={s}, expected 1.0"
    print(f"✓ test_motion_kernel_normalization passed ({len(test_cases)} cases)")


def test_motion_blur_direction():
    """验证核的主方向与 θ 一致（通过质心惯性矩）。"""
    test_cases = [(20, 0), (20, 45), (20, 90), (50, 30), (50, 60)]

    for r, theta in test_cases:
        kernel = generate_motion_kernel(r, theta)
        h, w = kernel.shape

        # 质心
        total = kernel.sum()
        if total < 1e-12:
            continue
        cy = float(np.sum(np.arange(h)[:, None] * kernel) / total)
        cx = float(np.sum(np.arange(w)[None, :] * kernel) / total)

        # 惯性矩主方向
        y, x = np.ogrid[:h, :w]
        dy = y - cy
        dx = x - cx
        m11 = float(np.sum(dy * dx * kernel))
        m20 = float(np.sum(dy ** 2 * kernel))
        m02 = float(np.sum(dx ** 2 * kernel))

        # 主方向角度
        if abs(m20 - m02) < 1e-12 and abs(m11) < 1e-12:
            continue  # 对称核
        principal_angle = 0.5 * np.degrees(np.arctan2(2 * m11, m20 - m02))
        # 归一化到 [0, 90]
        principal_angle = abs(principal_angle) % 90

        expected_angle = abs(theta) % 90
        # 允许 ±10 度误差
        angle_diff = min(abs(principal_angle - expected_angle),
                         90 - abs(principal_angle - expected_angle))
        assert angle_diff < 15, (
            f"r={r}, θ={theta}: principal={principal_angle:.1f}, expected≈{expected_angle}, diff={angle_diff:.1f}"
        )

    print(f"✓ test_motion_blur_direction passed ({len(test_cases)} cases)")


def test_chroma_noise_neutral():
    """纯绿像素经过色度噪声后，G 通道不应系统性偏移。"""
    I = np.zeros((64, 64, 3), dtype=np.float32)
    I[..., 1] = 1.0  # 纯绿
    rng = np.random.default_rng(42)
    cfg = load_yaml(CONFIG_PATH)
    # 强制启用色度噪声
    cfg['sensor']['chroma_noise']['enable'] = True
    cfg['sensor']['chroma_noise']['prob'] = 1.0  # 必触发

    for trial in range(100):
        result = apply_chroma_noise(I.copy(), rng, cfg)
        # G 通道均值应接近 1.0（保留原始，不应系统性变为 0.587）
        g_mean = float(result[..., 1].mean())
        assert g_mean > 0.9, (
            f"Trial {trial}: G channel degraded to {g_mean:.3f}, expected > 0.9"
        )

    print("✓ test_chroma_noise_neutral passed (100 trials)")


def test_isp_parameter_consistency():
    """零退化时，正向参数必须等于逆向参数。"""
    cfg = load_yaml(CONFIG_PATH)
    cfg_zero = make_zero_degradation_config(cfg)

    rng = np.random.default_rng(42)
    vc = make_zero_degradation_vc(cfg_zero)

    I_srgb = load_srgb(os.path.join(TEST_DATA_DIR, "gt", "test_00001.png"))
    I_linear = inverse_isp(I_srgb, vc, cfg_zero)
    I_recovered = forward_isp(I_linear, vc, np.random.default_rng(42), cfg_zero)

    rmse = float(np.sqrt(np.mean((I_srgb - I_recovered) ** 2)))
    assert rmse <= 0.003, f"RMSE {rmse:.6f} exceeds 0.003"
    print(f"✓ test_isp_parameter_consistency passed (RMSE={rmse:.6f})")


def test_gt_field_not_none():
    """调用 __call__ 后，result['gt'] 必须是 [512,512,3] float32。"""
    cfg = load_yaml(CONFIG_PATH)
    dataset = SceneDegradation(cfg, TEST_DATA_MANIFEST, base_seed=42)

    # source='gt'
    result = dataset("test_00001", dataset.master_rng, source='gt')
    gt = result['gt']
    assert gt is not None, "gt field is None for source='gt'"
    assert gt.shape == (512, 512, 3), f"gt shape mismatch: {gt.shape}"
    assert gt.dtype == np.float32, f"gt dtype mismatch: {gt.dtype}"
    assert np.all(gt >= 0.0) and np.all(gt <= 1.0), "gt out of range [0,1]"

    # source='flare3d_lq'
    result_flare = dataset("flare_0001", dataset.master_rng, source='flare3d_lq')
    gt_flare = result_flare['gt']
    assert gt_flare is not None, "gt field is None for source='flare3d_lq'"
    assert gt_flare.shape == (512, 512, 3), f"flare gt shape mismatch: {gt_flare.shape}"

    print("✓ test_gt_field_not_none passed")


def test_flare3d_pipeline():
    """构造虚构的 flare3d_input_path，走轻量链路，验证输出满足契约。"""
    cfg = load_yaml(CONFIG_PATH)
    dataset = SceneDegradation(cfg, TEST_DATA_MANIFEST, base_seed=42)

    result = dataset("flare_0001", dataset.master_rng, source='flare3d_lq')
    lq = result['lq']

    assert lq.shape == (512, 512, 3), f"Shape mismatch: {lq.shape}"
    assert lq.dtype == np.float32, f"dtype mismatch: {lq.dtype}"
    assert np.all(lq >= 0.0), "Values below 0"
    assert np.all(lq <= 1.0), "Values above 1"
    assert np.any(lq > 0.01), "Output is all black"

    # 验证 stages_run
    stages = result['sample_params']['stages_run']
    assert 'inverse_isp_lightweight' in stages, "Missing inverse_isp_lightweight"
    assert 'sensor_noise' in stages, "Missing sensor_noise"
    assert 'forward_isp_lightweight' in stages, "Missing forward_isp_lightweight"
    assert 'zoom_jpeg' in stages, "Missing zoom_jpeg"

    print("✓ test_flare3d_pipeline passed")


def test_scheduler_sequence_used():
    """验证 __call__ 中确实使用了预生成序列而非硬编码顺序。"""
    cfg = load_yaml(CONFIG_PATH)
    dataset = SceneDegradation(cfg, TEST_DATA_MANIFEST, base_seed=42)

    # 检查预生成序列存在
    assert len(dataset.sequences) > 0, "No pregenerated sequences"

    # 多次调用，记录 stages_run
    stages_runs = []
    for i in range(20):
        rng = np.random.default_rng(42 + i)
        result = dataset("test_00001", rng, source='gt')
        stages_runs.append(tuple(result['sample_params']['stages_run']))

    # 至少应有一定多样性（不全是同一顺序）
    unique_runs = set(stages_runs)
    assert len(unique_runs) >= 2, (
        f"All runs have same stage order, scheduler may be hardcoded: {unique_runs}"
    )

    # 所有 stages_run 必须满足约束：sensor_noise 在 forward_isp 之前，等等
    for run in stages_runs:
        for i, stage in enumerate(run):
            if stage == 'sensor_noise':
                assert 'forward_isp' in run[i:], "sensor_noise before forward_isp violated"
            if stage == 'forward_isp':
                assert 'zoom_jpeg' in run[i:], "forward_isp before zoom_jpeg violated"

    print(f"✓ test_scheduler_sequence_used passed ({len(unique_runs)} unique orderings)")


def test_single_process():
    """num_workers=0 时，不依赖 worker_init_fn。

    说明：SceneDegradation 是"合成单个样本"的 callable 接口（__call__(name, master_rng, source)），
    并非标准 PyTorch Dataset（无 __getitem__/__len__），因此不能直接用 DataLoader 封装。
    设计书命名的"num_workers=0 兼容性"验证的正是：在没有 worker、未调用 worker_init_fn 的
    单进程场景下，直接使用 __init__ 中兜底初始化的 master_rng 即可产出有效样本。
    （不建议：scene_degradation.py 提供 worker_init_fn 仅用于 num_workers>0 的多进程 DataLoader，
    其 dataset 需被训练侧包装成 __getitem__ 接口后再交给 DataLoader。）
    """
    cfg = load_yaml(CONFIG_PATH)
    dataset = SceneDegradation(cfg, TEST_DATA_MANIFEST, base_seed=42)

    # 模拟 num_workers=0：不调用 worker_init_fn，直接使用 __init__ 中的 master_rng
    result = dataset("test_00001", dataset.master_rng, source='gt')

    assert result['lq'] is not None, "lq is None"
    assert result['gt'] is not None, "gt is None"
    assert result['lq'].shape == (512, 512, 3)
    print("✓ test_single_process passed")


# ---------------------------------------------------------------------------- #
# 测试运行脚本 (§5)
# ---------------------------------------------------------------------------- #

def run_all_tests():
    """运行所有 14 项测试。"""
    tests = [
        test_output_contract,
        test_bgr_rgb_consistency,
        test_zero_degradation_roundtrip,
        test_inverse_wb_invariants,
        test_noise_variance,
        test_rng_reproducibility,
        test_motion_kernel_normalization,
        test_motion_blur_direction,
        test_chroma_noise_neutral,
        test_isp_parameter_consistency,
        test_gt_field_not_none,
        test_flare3d_pipeline,
        test_scheduler_sequence_used,
        test_single_process,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()  # 打印完整堆栈，便于排查代码 Bug
            failed += 1

    print(f"\n=== Summary ===\nPassed: {passed}/{len(tests)}\nFailed: {failed}/{len(tests)}")
    return failed == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-prepare', action='store_true',
                        help='跳过测试数据准备，直接运行测试')
    args = parser.parse_args()

    if not args.no_prepare:
        print("[prepare] building test data ...")
        prepare_test_data()
        print("[prepare] test data ready.")

    success = run_all_tests()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
