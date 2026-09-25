"""Pipeline configuration loader. One YAML file -> reproducible runs (config hash is stored with results)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "pipeline.yaml"


class Config:
    """Read-only-ish nested config: cfg.yolo_seg.conf, cfg['yolo_seg']['conf'] or cfg.get('yolo_seg.conf')."""

    def __init__(self, data: Dict[str, Any], source: str = ""):
        self._data = data
        self.source = source

    def __getattr__(self, name: str) -> Any:
        try:
            v = self._data[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Config(v) if isinstance(v, dict) else v

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    @property
    def hash(self) -> str:
        blob = json.dumps(self._data, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def override(self, **dotted_values: Any) -> "Config":
        """Return a copy with dotted overrides, e.g. cfg.override(**{"yolo_seg.conf": 0.4})."""
        data = copy.deepcopy(self._data)
        for dotted, value in dotted_values.items():
            cur = data
            parts = dotted.split(".")
            for p in parts[:-1]:
                cur = cur.setdefault(p, {})
            cur[parts[-1]] = value
        return Config(data, self.source)

    def path(self, dotted: str) -> Path:
        """Resolve a config path relative to the project root."""
        p = Path(self.get(dotted))
        return p if p.is_absolute() else ROOT_DIR / p


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("MARINEGUARD_CONFIG") or DEFAULT_CONFIG_PATH)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data, source=str(path))
