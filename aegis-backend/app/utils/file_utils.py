"""File utility functions for AEGIS."""

import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> Path:
    """Save data as JSON file."""
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str, ensure_ascii=False)
    return filepath


def load_json(filepath: Union[str, Path]) -> Any:
    """Load JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]) -> Path:
    """Save data as pickle file."""
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    with open(filepath, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return filepath


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load pickle file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)


def get_file_hash(filepath: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute file hash for integrity checking."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    hasher = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def cleanup_old_files(
    directory: Union[str, Path],
    max_files: int = 100,
    pattern: str = "*",
) -> int:
    """Remove oldest files exceeding max count.

    Returns number of files removed.
    """
    directory = Path(directory)
    if not directory.exists():
        return 0

    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    removed = 0
    while len(files) > max_files:
        files[0].unlink()
        files.pop(0)
        removed += 1
    return removed


def list_files_by_extension(
    directory: Union[str, Path],
    extension: str = ".pkl",
    recursive: bool = True,
) -> List[Path]:
    """List files with given extension."""
    directory = Path(directory)
    if not directory.exists():
        return []
    if recursive:
        return sorted(directory.rglob(f"*{extension}"))
    return sorted(directory.glob(f"*{extension}"))
