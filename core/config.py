"""
core/config.py
YAML configuration loader with environment variable overrides.
"""
import os
import yaml
from pathlib import Path
from typing import Any

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"
_settings: dict[str, Any] = {}


def _load_config() -> dict[str, Any]:
    """Load YAML config and merge environment variable overrides."""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found at {_CONFIG_PATH}. "
            "Copy config/settings.yaml.example → config/settings.yaml"
        )
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Allow env var overrides: ALPHAGRID_ALPACA_API_KEY → cfg["alpaca"]["api_key"]
    _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: dict, prefix: str = "AG") -> None:
    """Recursively apply environment variable overrides to config dict."""
    for key, val in cfg.items():
        env_key = f"{prefix}_{key.upper()}"
        if isinstance(val, dict):
            _apply_env_overrides(val, env_key)
        else:
            env_val = os.getenv(env_key)
            if env_val is not None:
                # Attempt type coercion
                if isinstance(val, bool):
                    cfg[key] = env_val.lower() in ("true", "1", "yes")
                elif isinstance(val, int):
                    cfg[key] = int(env_val)
                elif isinstance(val, float):
                    cfg[key] = float(env_val)
                else:
                    cfg[key] = env_val


def get_settings() -> dict[str, Any]:
    """Return the global settings dict, loading if necessary."""
    global _settings
    if not _settings:
        _settings = _load_config()
    return _settings


# Module-level convenience
settings = get_settings()
