"""AEGIS Logger - Structured logging with colored console output."""

import logging
import sys
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def get_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (usually module name).
        level: Log level string. Defaults to settings.LOG_LEVEL.
        log_file: Optional file path to also log to.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if level is None:
        try:
            from app.config import settings
            level = getattr(settings, "LOG_LEVEL", "INFO")
        except Exception:
            level = "INFO"

    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        fmt = ColoredFormatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            import os
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_fmt = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)

    return logger
