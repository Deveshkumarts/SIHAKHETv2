"""Shared plumbing for every pipeline stage: timing, structured logging, model versions."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger("marineguard")


def file_version(path: str | Path | None, label: str) -> str:
    """Stable model version string: '<label>@<sha256[:10]>' or '<label>@unavailable'."""
    if not path:
        return f"{label}@unavailable"
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return f"{label}@unavailable"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{label}@{h.hexdigest()[:10]}"


@dataclass
class StageRecord:
    stage: str
    seconds: float
    ok: bool
    summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class RunLog:
    """Collects a StageRecord per stage and mirrors it to a JSONL file + the python logger."""

    def __init__(self, run_id: Optional[str] = None, log_dir: str | Path | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.records: List[StageRecord] = []
        self._file: Optional[Path] = None
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._file = Path(log_dir) / f"{self.run_id}.jsonl"

    @contextmanager
    def stage(self, name: str) -> Iterator[Dict[str, Any]]:
        """Usage:  with runlog.stage('median') as s:  ...; s['pixels'] = n   (summary is logged)."""
        summary: Dict[str, Any] = {}
        t0 = time.perf_counter()
        ok, err = True, None
        try:
            yield summary
        except Exception as e:                        # log then re-raise: failures are never swallowed
            ok, err = False, f"{type(e).__name__}: {e}"
            raise
        finally:
            rec = StageRecord(name, round(time.perf_counter() - t0, 5), ok, _jsonable(summary), err)
            self.records.append(rec)
            log.info("[%s] %-18s %8.1f ms ok=%s %s", self.run_id, name, rec.seconds * 1000, ok, rec.summary)
            if self._file:
                with open(self._file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"run_id": self.run_id, **asdict(rec)}) + "\n")

    def record(self, name: str, seconds: float, summary: Optional[Dict[str, Any]] = None, ok: bool = True) -> None:
        """Log a stage whose time was accumulated elsewhere (e.g. summed over per-frame calls)."""
        rec = StageRecord(name, round(float(seconds), 5), ok, _jsonable(summary or {}), None)
        self.records.append(rec)
        log.info("[%s] %-18s %8.1f ms ok=%s %s", self.run_id, name, rec.seconds * 1000, ok, rec.summary)
        if self._file:
            with open(self._file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"run_id": self.run_id, **asdict(rec)}) + "\n")

    def timings(self) -> Dict[str, float]:
        return {r.stage: r.seconds for r in self.records}

    def total_seconds(self) -> float:
        return round(sum(r.seconds for r in self.records), 5)

    def to_list(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.records]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "shape", None) == ():   # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
