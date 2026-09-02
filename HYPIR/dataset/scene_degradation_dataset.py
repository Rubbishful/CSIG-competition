# -*- coding: utf-8 -*-
"""SceneDegradation 数据集的 torch 包装层（zip 批次产物 -> SD2 训练 batch）。

设计（与数据集讨论结论一致）：
  - zip_store.ZipBatchStore 是存储层（linear_raw/GT 按名随机读，zip 句柄 LRU）；
  - 本类在存储层之上：__getitem__(i) 确定性地为样本 i 生成退化 lq / gt；
  - __getitem__ 输出与 RealESRGANDataset 相同契约：CHW float32 [0,1] tensor（hq/gt 域），
    hence DataLoader 默认 collate 即可，无需自定义 collate；
  - SceneBatchTransform 把 collate 结果改名为 SD2Trainer.prepare_batch_inputs 期望的
    {GT, LQ, txt}（与 RealESRGANBatchTransform 的输出契约一致，但不做退化——退化已由
    SceneDegradation 完成）。

训练侧接入（configs/sd2_scene_train.yaml）：
  data_config.train.dataset.target = HYPIR.dataset.scene_degradation_dataset.SceneDegradationDataset
  data_config.train.batch_transform.target = ...SceneBatchTransform
  train.py（SD2Trainer）无需改动；退化决策链为每样本确定性 RNG（(base_seed, idx) 派生）。
"""

import numpy as np
import torch
from torch.utils import data

from HYPIR.dataset.batch_transform import BatchTransform
from HYPIR.dataset.scene_degradation import SceneDegradation, load_yaml, _project_root
from HYPIR.dataset.zip_store import ZipBatchStore

import os


class SceneDegradationDataset(data.Dataset):
    """按样本索引的退化合成数据集（lq/gt 由 SceneDegradation 现算）。

    cfg / cfg_path: 退化基线配置（configs/degradation_baseline.yaml）；cfg 优先
    manifest_path: 预处理产物 manifest.json（zip 批次或传统模式均可）
    zip_manifest: 为 True（默认）用 ZipBatchStore 读 linear_raw/GT；False 退回松散模式
    return_gt: 尽力返回原始 GT sRGB（zip 源下读 gt_zip/gt_arc；不可用时高保真重建）
    """

    def __init__(self, manifest_path: str, cfg: dict = None,
                 cfg_path: str = os.path.join('configs', 'degradation_baseline.yaml'),
                 zip_manifest: bool = True, return_gt: bool = True,
                 base_seed: int = 0, zip_cache_size: int = 4):
        if cfg is None:
            cfg = load_yaml(os.path.join(_project_root(), cfg_path))
        self.cfg = cfg
        self.return_gt = return_gt

        self.store = ZipBatchStore(manifest_path, zip_cache_size=zip_cache_size) \
            if zip_manifest else None
        self.scene = SceneDegradation(
            cfg, manifest_path, base_seed=base_seed,
            linear_raw_loader=(self.store.load_linear_raw if self.store is not None else None),
        )
        self._base_seed = int(base_seed)
        self.names = [rec['name'] for rec in self.scene.manifest.manifest['files']]

    def set_epoch(self, epoch: int):
        """调整样本级种子流（每 epoch 退化链变化；DataLoader shuffle 之外再增多样性）。"""
        self._base_seed = int(epoch) * 1000003

    def _sample_rng(self, idx: int) -> np.random.Generator:
        seed = (self._base_seed * 1000003 + idx) % (2 ** 32)
        return np.random.default_rng(seed)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        rng = self._sample_rng(idx)
        res = self.scene(name, rng, source='gt')
        lq, gt = res['lq'], res['gt']

        # 若 zip 源提供原始 GT PNG，用真 GT 代替重建 GT（保真度更高）
        if self.return_gt and self.store is not None:
            gt_orig = self.store.load_gt_png(name)
            if gt_orig is not None:
                gt = np.clip(gt_orig, 0.0, 1.0).astype(np.float32)

        # 与 RealESRGANDataset 契约一致：CHW, [0,1], float32 tensor
        lq_t = torch.from_numpy(np.ascontiguousarray(lq)).permute(2, 0, 1).contiguous().float()
        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1).contiguous().float()
        return {'lq': lq_t, 'gt': gt_t, 'name': name, 'txt': ''}


def collate_samples(samples) -> dict:
    """手动 collate（不依赖 DataLoader 默认 collate 时使用）：返回 batch 张量。"""
    return {
        'lq': torch.stack([s['lq'] for s in samples]),
        'gt': torch.stack([s['gt'] for s in samples]),
        'name': [s['name'] for s in samples],
        'txt': [s['txt'] for s in samples],
    }


class SceneBatchTransform(BatchTransform):
    """把 SceneDegradationDataset 的 collate 结果改名/整理为 SD2Trainer 输入契约。

    输入（collate 后）: {'lq': [B,3,H,W], 'gt': [B,3,H,W], 'name': [...], 'txt': [...]}
    输出: {'GT': [B,3,H,W], 'LQ': [B,3,H,W], 'txt': [...], 'name': [...]}（GT/LQ 均为 [0,1]）
    """

    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, batch: dict) -> dict:
        return {
            'GT': batch['gt'],
            'LQ': batch['lq'],
            'txt': batch['txt'],
            'name': batch['name'],
        }
