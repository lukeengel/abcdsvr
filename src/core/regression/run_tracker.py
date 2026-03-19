"""Run metadata tracking for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class RunMetadata:
    """Metadata for a single regression run."""
    run_id: str
    run_name: str
    seed: int
    timestamp: str
    git_commit: str
    config_hash: str
    description: str = ""
    changes_from_last_run: str = ""
    pearson_r: float | None = None
    pearson_p_emp: float | None = None
    n_samples: int | None = None
    model_name: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunMetadata":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def _git_commit(repo_root: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=repo_root, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _config_hash(configs: dict) -> str:
    try:
        serialized = json.dumps(configs, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:12]
    except Exception:
        return "unknown"


def _serialize_configs(env) -> dict:
    out = {}
    for attr in ("data", "regression", "harmonize", "run"):
        cfg = getattr(env.configs, attr, None)
        if cfg is not None:
            out[attr] = cfg
    return out


def save_run_metadata(
    env,
    results_dir: str | Path,
    description: str = "",
    changes_from_last_run: str = "",
    metrics: dict | None = None,
) -> RunMetadata:
    """Save run_metadata.yaml and config_snapshot.yaml to results_dir."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = env.configs.run
    all_configs = _serialize_configs(env)

    meta = RunMetadata(
        run_id=run_cfg.get("run_id", "unknown"),
        run_name=run_cfg.get("run_name", "unknown"),
        seed=run_cfg.get("seed", 42),
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        git_commit=_git_commit(),
        config_hash=_config_hash(all_configs),
        description=description or run_cfg.get("description", ""),
        changes_from_last_run=changes_from_last_run,
    )

    if metrics:
        meta.pearson_r = metrics.get("pearson_r")
        meta.pearson_p_emp = metrics.get("pearson_p_emp")
        meta.n_samples = metrics.get("n_samples")
        meta.model_name = metrics.get("model_name")

    meta_path = results_dir / "run_metadata.yaml"
    with open(meta_path, "w") as f:
        yaml.dump(meta.to_dict(), f, default_flow_style=False, sort_keys=True)

    snapshot_path = results_dir / "config_snapshot.yaml"
    with open(snapshot_path, "w") as f:
        yaml.dump(all_configs, f, default_flow_style=False, sort_keys=True)

    logger.info(f"Run metadata saved: {meta_path}")
    return meta


def load_run_metadata(results_dir: str | Path) -> RunMetadata | None:
    meta_path = Path(results_dir) / "run_metadata.yaml"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        d = yaml.safe_load(f)
    return RunMetadata.from_dict(d)


def list_runs(
    outputs_root: str | Path,
    run_name: str | None = None,
) -> list[RunMetadata]:
    outputs_root = Path(outputs_root)
    metas: list[RunMetadata] = []
    for meta_path in sorted(outputs_root.rglob("run_metadata.yaml")):
        meta = load_run_metadata(meta_path.parent)
        if meta is None:
            continue
        if run_name is not None and meta.run_name != run_name:
            continue
        metas.append(meta)
    metas.sort(key=lambda m: m.timestamp)
    return metas
