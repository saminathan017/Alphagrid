"""
core/logger.py
Centralized logging for the AlphaGrid system.
Uses loguru for structured, colorized, rotating logs.
"""
import sys
from pathlib import Path
from loguru import logger
from .config import settings


def setup_logger() -> None:
    """Initialize the global logger with file + console sinks."""
    logger.remove()  # Remove default handler

    log_cfg = settings.get("logging", {})
    log_level = log_cfg.get("level", "INFO")
    log_file  = log_cfg.get("file", "logs/alphagrid.log")
    rotation  = log_cfg.get("rotation", "1 day")
    retention = log_cfg.get("retention", "30 days")
    fmt       = log_cfg.get(
        "format",
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console sink
    logger.add(
        sys.stderr,
        level=log_level,
        format=fmt,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File sink (rotating)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level=log_level,
        format=fmt,
        rotation=rotation,
        retention=retention,
        compression="zip",
        backtrace=True,
        diagnose=False,  # Don't expose sensitive data in log files
        enqueue=True,    # Thread-safe async logging
    )

    logger.info("AlphaGrid logger initialized.")


def get_logger(name: str):
    """Return a bound logger with the module name context."""
    return logger.bind(name=name)
