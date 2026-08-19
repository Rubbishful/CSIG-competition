"""评估结果输出:CSV 明细 + JSON 汇总 + 控制台表格。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_csv(rows: list[dict], path: Path) -> Path:
    """把每张图的指标行存为 CSV(自动排序列、保留浮点精度)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_table(rows: list[dict], float_fmt: str = "{:.4f}") -> None:
    """控制台打印表格;数值列四舍五入到 4 位小数。"""
    if not rows:
        print("(无数据)")
        return
    df = pd.DataFrame(rows)
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda v: float_fmt.format(v) if pd.notna(v) else v)
    print(df.to_string(index=False))


def summarize(rows: list[dict], numeric_cols: list[str]) -> dict:
    """对数值列输出 mean / std / min / max 汇总。"""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    summary = {"count": len(rows)}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        summary[col] = {
            "mean": None if s.isna().all() else round(float(s.mean()), 4),
            "std": None if s.isna().all() else round(float(s.std(ddof=0)), 4),
            "min": None if s.isna().all() else round(float(s.min()), 4),
            "max": None if s.isna().all() else round(float(s.max()), 4),
        }
    return summary
