# -*- coding: utf-8 -*-
"""预处理 zip 批次产物的统一读取 API（训练侧与 preprocess zip 模式严格对应）。

产物布局（preprocess.py zip 批次模式）：
    <output_dir>/
    ├── manifest.json            # 每条记录含 'zip': 'batches/batch_XXXXX.zip'
    │                            # zip 源模式下另有 gt_zip/gt_arc（原始 GT PNG 条目）
    └── batches/
        ├── batch_00000.zip      # 内含 linear_raw/{name}.npy * batch_size
        └── batch_00001.zip ...

设计要点（OSS 场景）：
    - 一次只打开一个 zip、按需读取单个条目；内部维护最近打开的 zip 句柄 LRU；
    - 训练 DataLoader 可按样本名随机访问（zip=数据块语义），内存峰值 = 单批样本数；
    - 与 manifest 的 name 索引一致，可直接配合 SceneDegradation 使用。

用法示例：
    from HYPIR.dataset.zip_store import ZipBatchStore
    store = ZipBatchStore("preprocessed/manifest.json")
    names = store.batch_names("batch_00000.zip")     # 该批全部样本名
    I = store.load_linear_raw(names[0])              # [512,512,3] float32
    gt = store.load_gt_png(names[0])                 # 原始 GT sRGB（zip 源模式下可用）
"""

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np


class ZipBatchStore:
    """按 manifest 索引的 zip 批次读取器（含 zip 句柄 LRU）。"""

    def __init__(self, manifest_path: str, zip_cache_size: int = 4):
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest_dir = os.path.dirname(self.manifest_path)
        self.zip_cache_size = max(1, zip_cache_size)
        self._zip_cache = {}  # zip_path -> ZipFile（部分打开）
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)
        self._index = {rec['name']: rec for rec in self.manifest['files']}

        # 预计算: name -> (zip_abs, arcname)；name -> GT 源描述
        self._linear_map = {}
        self._gt_map = {}  # name -> ('zip', zip_abs, arc) | ('file', path) | None
        for rec in self.manifest['files']:
            name = rec['name']
            if rec.get('zip'):
                self._linear_map[name] = (
                    os.path.join(self.manifest_dir, rec['zip']),
                    f"linear_raw/{name}.npy",
                )
            if rec.get('gt_zip') and rec.get('gt_arc'):
                self._gt_map[name] = ('zip', rec['gt_zip'], rec['gt_arc'])
            elif rec.get('gt_path'):
                self._gt_map[name] = ('file', rec['gt_path'])

    # ------------------------------------------------------------------ #
    # 元信息
    # ------------------------------------------------------------------ #

    @property
    def names(self):
        return [rec['name'] for rec in self.manifest['files']]

    def __len__(self):
        return len(self.manifest['files'])

    def record(self, name: str) -> dict:
        return self._index[name]

    def batch_zip_paths(self) -> list:
        """按名字典序返回全部批次 zip 的绝对路径（batch_00000, batch_00001, ...）。"""
        zips = sorted({os.path.join(self.manifest_dir, rec['zip'])
                       for rec in self.manifest['files'] if rec.get('zip')})
        return zips

    def batch_names(self, batch_zip: str) -> list:
        """返回某个批次 zip（绝对或相对 manifest 的路径）包含的样本名（按 manifest 顺序）。"""
        abs_zip = batch_zip if os.path.isabs(batch_zip) else os.path.join(self.manifest_dir, batch_zip)
        return [rec['name'] for rec in self.manifest['files']
                if rec.get('zip') and os.path.join(self.manifest_dir, rec['zip']) == abs_zip]

    # ------------------------------------------------------------------ #
    # 底层读取（LRU 句柄）
    # ------------------------------------------------------------------ #

    def _read(self, zip_path: str, arcname: str) -> bytes:
        zf = self._zip_cache.pop(zip_path, None)
        if zf is None:
            zf = zipfile.ZipFile(zip_path)
        try:
            return zf.read(arcname)
        finally:
            self._zip_cache[zip_path] = zf
            while len(self._zip_cache) > self.zip_cache_size:
                oldest = next(iter(self._zip_cache))
                self._zip_cache.pop(oldest).close()

    # ------------------------------------------------------------------ #
    # 进程安全（DataLoader worker = spawn 时需可 pickle）
    # ------------------------------------------------------------------ #

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_zip_cache'] = {}
        for zf in self._zip_cache.values():
            try:
                zf.close()
            except Exception:
                pass
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ------------------------------------------------------------------ #
    # 数据读取
    # ------------------------------------------------------------------ #

    def load_linear_raw(self, name: str) -> np.ndarray:
        """从批次 zip 中读取单张 linear_raw（[512,512,3] float32 [0,1]）。"""
        if name not in self._linear_map:
            raise KeyError(f"记录 {name} 无 'zip' 字段（非 zip 批次产物）")
        zip_path, arcname = self._linear_map[name]
        data = self._read(zip_path, arcname)
        return np.load(io.BytesIO(data))

    def load_gt_png(self, name: str) -> np.ndarray:
        """读取原始 GT sRGB（[512,512,3] float32 [0,1]）。

        优先 zip 源 (gt_zip/gt_arc)，其次传统 gt_path；均无时返回 None
        （由场景层用 forward_isp 高保真重建代替）。
        """
        src = self._gt_map.get(name)
        if src is None:
            return None
        if src[0] == 'zip':
            buf = self._read(src[1], src[2])
        else:
            with open(src[1], 'rb') as f:
                buf = f.read()
        from HYPIR.dataset.scene_degradation import load_srgb_bytes
        return load_srgb_bytes(buf)

    def load_batch(self, batch_zip: str):
        """读取整个批次：返回 (names, np.ndarray [N,512,512,3])。"""
        names = self.batch_names(batch_zip)
        if not names:
            raise ValueError(f"批次文件中没有 manifest 记录: {batch_zip}")
        imgs = [self.load_linear_raw(n) for n in names]
        return names, np.stack(imgs)
