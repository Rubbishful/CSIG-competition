# -*- coding: utf-8 -*-
"""预处理 zip 批次产物的统一读取 API（训练侧与 preprocess.zipp 模式严格对应）。

产物布局（preprocess.py zip 批次模式）：
    <output_dir>/
    ├── manifest.json            # 每条记录含 'zip': 'batches/batch_XXXXX.zip'
    └── batches/
        ├── batch_00000.zip      # 内含 linear_raw/{name}.npy * batch_size
        └── batch_00001.zip ...

设计要点（OSS 场景）：
    - 一次只打开一个 zip、按需读取单个 npy（ZipFile 读中央目录，随机访问单文件）；
    - 训练 DataLoader 可按 batch_zip 顺序/乱序消费，内存峰值 = 单 zip 内样本数；
    - 与 manifest 的 name 索引一致，可直接配合 SceneDegradation 使用。

用法示例：
    from HYPIR.dataset.zip_store import ZipBatchStore
    store = ZipBatchStore("preprocessed/manifest.json")
    names = store.batch_names("batch_00000.zip")     # 该批全部样本名
    I = store.load_linear_raw(names[0])              # [512,512,3] float32
"""

import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np


class ZipBatchStore:
    """按 manifest 索引的 zip 批次读取器。"""

    def __init__(self, manifest_path: str):
        self.manifest_path = os.path.abspath(manifest_path)
        self.manifest_dir = os.path.dirname(self.manifest_path)
        with open(self.manifest_path, 'r') as f:
            self.manifest = json.load(f)
        self._index = {rec['name']: rec for rec in self.manifest['files']}

    # ------------------------------------------------------------------ #
    # manifest 元信息
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
        zips = sorted({rec['zip'] for rec in self.manifest['files'] if rec.get('zip')})
        return [os.path.join(self.manifest_dir, z) for z in zips]

    def batch_names(self, batch_zip: str) -> list:
        """返回某个批次 zip（绝对或相对 manifest 的路径）包含的样本名（按 manifest 顺序）。"""
        abs_zip = batch_zip if os.path.isabs(batch_zip) else os.path.join(self.manifest_dir, batch_zip)
        return [rec['name'] for rec in self.manifest['files']
                if rec.get('zip') and os.path.join(self.manifest_dir, rec['zip']) == abs_zip]

    # ------------------------------------------------------------------ #
    # 数据读取
    # ------------------------------------------------------------------ #

    def _zip_and_arcname(self, name: str):
        rec = self._index[name]
        zip_rel = rec.get('zip')
        if not zip_rel:
            raise KeyError(f"记录 {name} 无 'zip' 字段（非 zip 批次产物）")
        return os.path.join(self.manifest_dir, zip_rel), f"linear_raw/{name}.npy"

    def load_linear_raw(self, name: str) -> np.ndarray:
        """从批次 zip 中读取单张 linear_raw（[512,512,3] float32 [0,1]）。"""
        zip_path, arcname = self._zip_and_arcname(name)
        with zipfile.ZipFile(zip_path) as zf:
            data = zf.read(arcname)
        return np.load(io.BytesIO(data))

    def load_batch(self, batch_zip: str):
        """读取整个批次：返回 (names, np.ndarray [N,512,512,3])。"""
        names = self.batch_names(batch_zip)
        if not names:
            raise ValueError(f"批次文件中没有 manifest 记录: {batch_zip}")
        imgs = [self.load_linear_raw(n) for n in names]
        return names, np.stack(imgs)
