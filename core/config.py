"""
core/config.py
YAML configuration loader with environment variable overrides.
"""
import os
import re
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

    # Allow env var overrides: AG_ALPACA_API_KEY → cfg["alpaca"]["api_key"]
    _apply_env_overrides(cfg)
    _apply_common_env_aliases(cfg)
    _normalize_legacy_keys(cfg)
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


def _apply_common_env_aliases(cfg: dict[str, Any]) -> None:
    """Support the .env names used elsewhere in the project."""
    env_map = {
        ("alpaca", "api_key"): "ALPACA_API_KEY",
        ("alpaca", "secret_key"): "ALPACA_SECRET_KEY",
        ("alpaca", "base_url"): "ALPACA_BASE_URL",
        ("news_api", "api_key"): "NEWS_API_KEY",
        ("reddit", "client_id"): "REDDIT_CLIENT_ID",
        ("reddit", "client_secret"): "REDDIT_CLIENT_SECRET",
    }
    for path, env_name in env_map.items():
        env_val = os.getenv(env_name)
        if env_val is None:
            continue
        current = cfg
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = env_val


def _normalize_legacy_keys(cfg: dict[str, Any]) -> None:
    """Backfill aliases so older modules and current YAML stay compatible."""
    has_symbols_section = "symbols" in cfg
    universe = cfg.get("universe", {})
    symbols = cfg.setdefault("symbols", {})
    if not has_symbols_section:
        try:
            from core.ticker_universe import US_SYMBOLS, FOREX_SYMBOLS
            symbols.setdefault("us_equities", list(US_SYMBOLS))
            symbols.setdefault("forex", list(FOREX_SYMBOLS))
        except Exception:
            symbols.setdefault("us_equities", universe.get("us_equities", []))
            symbols.setdefault("forex", universe.get("forex", []))
    else:
        symbols.setdefault("us_equities", universe.get("us_equities", []))
        symbols.setdefault("forex", universe.get("forex", []))
    symbols["us_equities"] = _normalize_symbol_list(symbols.get("us_equities", []))
    symbols["forex"] = _normalize_symbol_list(symbols.get("forex", []))

    risk = cfg.setdefault("risk", {})
    stop_loss = risk.setdefault("stop_loss", {})
    take_profit = risk.setdefault("take_profit", {})

    if "trailing_activation" not in stop_loss and "trailing_act" in stop_loss:
        stop_loss["trailing_activation"] = stop_loss["trailing_act"]
    if "risk_reward_ratio" not in take_profit and "rr_ratio" in take_profit:
        take_profit["risk_reward_ratio"] = take_profit["rr_ratio"]

    database = cfg.setdefault("database", {})
    if "url" not in database:
        sqlite_path = database.get("sqlite_path")
        if sqlite_path:
            database["url"] = f"sqlite:///{sqlite_path}"


def _normalize_symbol_list(values: list[Any]) -> list[str]:
    """
    Expand YAML rows like "- AAPL - MSFT - NVDA" into individual symbols.
    """
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        parts = re.split(r"\s+-\s+", value.strip())
        for part in parts:
            symbol = part.strip()
            if symbol:
                result.append(symbol)
    return result


def get_settings() -> dict[str, Any]:
    """Return the global settings dict, loading if necessary."""
    global _settings
    if not _settings:
        _settings = _load_config()
    return _settings


# Module-level convenience
settings = get_settings()
