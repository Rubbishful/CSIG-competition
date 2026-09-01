# -*- coding: utf-8 -*-
"""基线场景退化合成管线 —— 统一模块入口。

《设计书V3基线测试.md》约定基线实现以本模块为统一命名空间；
本文档只负责把实际实现（HYPIR/dataset/scene_degradation.py 与
HYPIR/dataset/preprocess.py）中的函数/常量集中导出，供测试系统与使用方
统一 `import baseline_impl`。参见《设计书V3基线实现.md》。
"""

from HYPIR.dataset.scene_degradation import (  # noqa: F401
    # 常数
    WB_GAINS_FIXED,
    DEFAULT_STAGES,
    # 基础工具
    load_srgb,
    load_yaml,
    deep_merge,
    _project_root,
    resolve_path,
    # 数据契约
    PreprocessManager,
    # Tone Mapping
    smoothstep_forward,
    smoothstep_inverse,
    # 逆 ISP
    apply_inverse_ccm,
    highlight_preserving_inverse,
    inverse_isp,
    # 虚拟相机
    sample_wb_gains,
    sample_ccm_matrix,
    sample_noise_params,
    sample_virtual_camera,
    # 传感器噪声
    apply_sensor_noise,
    apply_chroma_noise,
    apply_dark_region_noise_boost,
    apply_noise_chain,
    # 正向 ISP
    apply_vignetting,
    apply_sharpening,
    apply_color_cast,
    forward_isp,
    # 变焦 + JPEG
    double_jpeg,
    zoom_and_jpeg,
    # 运动模糊
    generate_motion_kernel,
    sample_motion_blur_params,
    precompute_legal_angles,
    apply_motion_blur,
    # 眩光
    FlareTemplateBank,
    should_apply_flare,
    apply_flare,
    # 调度器
    validate_constraints,
    random_topological_sort,
    pregenerate_sequences,
    _select_sequence,
    # 主入口
    serialize_vc,
    SceneDegradation,
    worker_init_fn,
)

from HYPIR.dataset.preprocess import (  # noqa: F401
    run_preprocess,
    generate_manifest,
    generate_flare_templates,
)

__all__ = [
    # 常数
    "WB_GAINS_FIXED",
    "DEFAULT_STAGES",
    # 基础工具
    "load_srgb",
    "load_yaml",
    "deep_merge",
    "_project_root",
    "resolve_path",
    # 数据契约
    "PreprocessManager",
    # 预处理入口
    "run_preprocess",
    "generate_manifest",
    "generate_flare_templates",
    # Tone Mapping / 逆 ISP / 正向 ISP
    "smoothstep_forward",
    "smoothstep_inverse",
    "apply_inverse_ccm",
    "highlight_preserving_inverse",
    "inverse_isp",
    "apply_vignetting",
    "apply_sharpening",
    "apply_color_cast",
    "forward_isp",
    # 虚拟相机
    "sample_wb_gains",
    "sample_ccm_matrix",
    "sample_noise_params",
    "sample_virtual_camera",
    # 噪声
    "apply_sensor_noise",
    "apply_chroma_noise",
    "apply_dark_region_noise_boost",
    "apply_noise_chain",
    # 变焦 + JPEG
    "double_jpeg",
    "zoom_and_jpeg",
    # 运动模糊
    "generate_motion_kernel",
    "sample_motion_blur_params",
    "precompute_legal_angles",
    "apply_motion_blur",
    # 眩光
    "FlareTemplateBank",
    "should_apply_flare",
    "apply_flare",
    # 调度器
    "validate_constraints",
    "random_topological_sort",
    "pregenerate_sequences",
    "_select_sequence",
    # 主入口
    "serialize_vc",
    "SceneDegradation",
    "worker_init_fn",
]
