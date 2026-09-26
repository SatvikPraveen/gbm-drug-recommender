"""Provenance for a pipeline run: code version, environment, configuration."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

from .. import __version__
from ..config import PROJECT_ROOT, config_summary

_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "rdkit",
    "torch",
    "torch-geometric",
    "umap-learn",
    "statsmodels",
)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except Exception:
        return None


def run_metadata(extra: dict | None = None) -> dict:
    versions = {}
    for pkg in _PACKAGES:
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = None
    meta = {
        "package_version": __version__,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependencies": versions,
        "config": config_summary(),
    }
    if extra:
        meta.update(extra)
    return meta


def write_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default) + "\n")
    return path


def _default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)
