"""Configuration loading utilities for ABCD Release 6."""

from __future__ import annotations

import yaml
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4


def initialize_notebook(
    run_name: str | None = None, *, regenerate_run_id: bool = False
):
    """Initialize project environment."""

    repo_root = _set_dir()
    loaded_configs = _load_configs()

    run_cfg = dict(loaded_configs.run)

    if run_name is not None:
        run_cfg["run_name"] = run_name
    run_cfg.setdefault("run_name", "regression")

    raw_run_id = run_cfg.get("run_id")
    if not isinstance(raw_run_id, str):
        raw_run_id = str(raw_run_id) if raw_run_id is not None else ""
    if regenerate_run_id or not raw_run_id or not raw_run_id.startswith("run-"):
        raw_run_id = f"run-{uuid4().hex[:10]}"
    run_cfg["run_id"] = raw_run_id

    _persist_run_config(repo_root, run_cfg)

    loaded_configs.run = run_cfg

    # Regression-only: no research_group, maximize subject count
    data_config = loaded_configs.data
    data_config["columns"]["derived"] = ["sex_mapped"]
    data_config.setdefault("splits", {})["stratify"] = None

    env = SimpleNamespace(
        repo_root=repo_root,
        configs=loaded_configs,
    )

    output_dir = _create_output_folder(env)
    print(f"Initialized notebook for run '{run_cfg['run_name']}'")
    print(f"Saved output summary to {output_dir}")
    return env


def _set_dir():
    """Find project root by looking for configs/ directory."""
    cwd = Path.cwd()
    current = cwd
    max_levels = 5
    for _ in range(max_levels):
        if (current / "configs").exists():
            return current
        if current.parent == current:
            break
        current = current.parent

    raise FileNotFoundError(
        f"Could not find 'configs' directory. Started from: {cwd}\n"
        f"Make sure you're running from within the project directory."
    )


def _load_configs():
    """Load configuration from YAML files."""
    repo = _set_dir()
    config_dir = repo / "configs"

    configs = {}
    for file in config_dir.glob("*.yaml"):
        with open(file, encoding="utf-8") as fh:
            configs[file.stem] = yaml.safe_load(fh)

    return SimpleNamespace(**configs)


def _create_output_folder(env):
    """Create the run output directory."""
    run_cfg = env.configs.run
    run_name = run_cfg.get("run_name", "regression")
    run_id = str(run_cfg.get("run_id", "run-unknown"))
    output_dir = env.repo_root / "outputs" / run_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_figures_dir(env, notebook: str) -> Path:
    """Return and create figures directory for a specific notebook's run."""
    run_cfg = env.configs.run
    fig_dir = (
        env.repo_root / "outputs" / run_cfg["run_name"]
        / run_cfg["run_id"] / "figures" / notebook
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def _persist_run_config(repo_root: Path, run_cfg: dict) -> None:
    """Persist run configuration back to configs/run.yaml for reuse."""
    config_path = repo_root / "configs" / "run.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(run_cfg, fh, sort_keys=False)
