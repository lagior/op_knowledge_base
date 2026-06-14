"""Load and provide access to application configuration."""

import os
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def _resolve_env_vars(value):
    """Replace ${ENV_VAR} patterns with environment variable values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.environ.get(env_var, value)
    if isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load config from YAML file and resolve environment variables."""
    with open(path) as f:
        config = yaml.safe_load(f)
    return _resolve_env_vars(config)
