"""YAML config loading with dotted-key command line overrides.

    cfg = load_config("configs/default.yaml", overrides=["reward.success=20"])
    cfg.reward.success  -> 20
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access (cfg.robot.dh.d)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    def to_dict(self):
        return _plain(self)


def _wrap(obj):
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def deep_merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_override(cfg: dict, expr: str) -> None:
    """Apply 'a.b.c=value'. The value is parsed as YAML (numbers, lists, bools...)."""
    if "=" not in expr:
        raise ValueError(f"override must look like key=value, got {expr!r}")
    key, raw = expr.split("=", 1)
    value = yaml.safe_load(raw)
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"unknown config section '{p}' in override {expr!r}")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"unknown config key '{key}' (typo?)")
    node[parts[-1]] = value


def load_config(path=None, overrides=None, extra_files=None) -> Config:
    """Load default.yaml, merge optional extra YAML files, then dotted overrides."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    files = []
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        files.append(path)
    files += list(extra_files or [])
    for p in files:
        with open(p, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for expr in overrides or []:
        apply_override(cfg, expr)
    return _wrap(cfg)


def save_config(cfg, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_plain(cfg), f, allow_unicode=True, sort_keys=False)
