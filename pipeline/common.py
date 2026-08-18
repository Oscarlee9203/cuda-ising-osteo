"""共用工具:讀設定、路徑、存取中繼檔。"""
from __future__ import annotations
import os
import json
import yaml
import numpy as np


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def workdir(cfg: dict) -> str:
    d = cfg.get("paths", {}).get("workdir", "outputs")
    os.makedirs(d, exist_ok=True)
    return d


def save_npz(path: str, **arrays) -> None:
    np.savez_compressed(path, **arrays)


def load_npz(path: str):
    return np.load(path, allow_pickle=True)


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
