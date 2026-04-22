"""
FileHandler – manages file uploads, dataset loading, and file lifecycle.

Supports CSV validation, file metadata extraction, and size limits.
"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import pandas as pd

    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False
    pd = None  # type: ignore[assignment]

try:
    from fastapi import UploadFile

    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False

# Default configuration
DEFAULT_UPLOAD_DIR = "uploads"
DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS: Dict[str, set] = {
    "csv": {".csv"},
    "excel": {".xlsx", ".xls"},
    "json": {".json"},
    "all": {".csv", ".xlsx", ".xls", ".json", ".parquet", ".tsv"},
}


class FileHandler:
    """Handles file uploads and dataset loading for the AEGIS platform.

    Parameters
    ----------
    upload_dir:
        Directory for uploaded files.  Created if it doesn't exist.
    max_file_size:
        Maximum allowed file size in bytes (default 50 MB).
    allowed_extensions:
        Set of allowed file extensions (e.g. ``{'.csv', '.json'}``).
        If None, allows all known types.
    """

    def __init__(
        self,
        upload_dir: Optional[str] = None,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        allowed_extensions: Optional[set] = None,
    ) -> None:
        self.upload_dir = Path(upload_dir or DEFAULT_UPLOAD_DIR)
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or ALLOWED_EXTENSIONS["all"]

        # Ensure upload directory exists
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "FileHandler initialised (upload_dir=%s, max_size=%d MB)",
            self.upload_dir,
            self.max_file_size // (1024 * 1024),
        )

    # ------------------------------------------------------------------
    # Upload handling
    # ------------------------------------------------------------------
    async def save_upload(
        self,
        file: Any,
        upload_dir: Optional[str] = None,
    ) -> str:
        """Save an uploaded file to disk.

        Parameters
        ----------
        file:
            A FastAPI :class:`UploadFile` or any object with
            ``filename`` and ``read()`` / ``file`` attributes.
        upload_dir:
            Override destination directory.

        Returns
        -------
        Absolute path to the saved file.
        """
        dest_dir = Path(upload_dir) if upload_dir else self.upload_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename
        if hasattr(file, "filename"):
            filename = file.filename
        elif hasattr(file, "name"):
            filename = Path(file.name).name
        else:
            raise ValueError("File object has no filename attribute")

        # Sanitise filename
        safe_name = self._sanitise_filename(filename)
        dest_path = dest_dir / safe_name

        # Handle duplicate filenames
        if dest_path.exists():
            stem = dest_path.stem
            suffix = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        # Write file
        if HAS_FASTAPI and isinstance(file, UploadFile):
            # FastAPI UploadFile – read chunks
            total_size = 0
            with open(dest_path, "wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    total_size += len(chunk)
                    if total_size > self.max_file_size:
                        # Clean up partial file
                        os.remove(dest_path)
                        raise ValueError(
                            f"File exceeds maximum size of "
                            f"{self.max_file_size // (1024 * 1024)} MB "
                            f"(got {total_size // (1024 * 1024)} MB)"
                        )
                    f.write(chunk)
            await file.close()
        elif hasattr(file, "file"):
            # File-like object with .file attribute (e.g. SpooledTemporaryFile)
            src = file.file if hasattr(file.file, "read") else file
            shutil.copyfileobj(src, dest_path)
        else:
            raise ValueError("Unsupported file object type")

        abs_path = str(dest_path.resolve())
        logger.info("File saved: %s", abs_path)
        return abs_path

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def load_csv(
        self,
        filepath: str,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> Any:
        """Load a CSV file into a pandas DataFrame.

        Parameters
        ----------
        filepath:
            Path to the CSV file.
        encoding:
            File encoding.
        **kwargs:
            Additional arguments passed to :func:`pandas.read_csv`.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        RuntimeError if pandas is not available.
        """
        if not HAS_PANDAS:
            raise RuntimeError("pandas is required to load CSV files but is not installed")

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if path.suffix.lower() == ".tsv":
            df = pd.read_csv(filepath, sep="\t", encoding=encoding, **kwargs)
        else:
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)

        logger.info("Loaded CSV: %s (%d rows, %d cols)", filepath, len(df), len(df.columns))
        return df

    def load_dataframe(self, filepath: str, **kwargs: Any) -> Any:
        """Load a file into a pandas DataFrame based on its extension.

        Supports CSV, TSV, JSON, Excel, and Parquet.
        """
        if not HAS_PANDAS:
            raise RuntimeError("pandas is required but not installed")

        path = Path(filepath)
        ext = path.suffix.lower()

        loaders = {
            ".csv": lambda: pd.read_csv(filepath, **kwargs),
            ".tsv": lambda: pd.read_csv(filepath, sep="\t", **kwargs),
            ".json": lambda: pd.read_json(filepath, **kwargs),
            ".parquet": lambda: pd.read_parquet(filepath, **kwargs),
            ".xlsx": lambda: pd.read_excel(filepath, **kwargs),
            ".xls": lambda: pd.read_excel(filepath, **kwargs),
        }

        loader = loaders.get(ext)
        if loader is None:
            raise ValueError(f"Unsupported file format: {ext}")

        df = loader()
        logger.info("Loaded %s: %s (%d rows, %d cols)", ext, filepath, len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_file(
        self,
        filepath: str,
        allowed_extensions: Optional[set] = None,
    ) -> bool:
        """Validate that a file exists, is within size limits, and has
        an allowed extension.

        Returns True if all checks pass.
        """
        path = Path(filepath)
        allowed = allowed_extensions or self.allowed_extensions

        # Existence
        if not path.exists():
            logger.warning("Validation failed – file not found: %s", filepath)
            return False

        # Extension
        if allowed and path.suffix.lower() not in allowed:
            logger.warning(
                "Validation failed – extension '%s' not in %s",
                path.suffix,
                allowed,
            )
            return False

        # Size
        file_size = path.stat().st_size
        if file_size > self.max_file_size:
            logger.warning(
                "Validation failed – file too large: %d bytes (max %d)",
                file_size,
                self.max_file_size,
            )
            return False

        if file_size == 0:
            logger.warning("Validation failed – empty file: %s", filepath)
            return False

        logger.debug("File validated: %s", filepath)
        return True

    # ------------------------------------------------------------------
    # File metadata
    # ------------------------------------------------------------------
    def get_file_info(self, filepath: str) -> Dict[str, Any]:
        """Get metadata about a file.

        Returns dict with: name, size, type, rows, columns, modified.
        """
        path = Path(filepath)

        if not path.exists():
            return {"error": f"File not found: {filepath}"}

        stat = path.stat()
        info: Dict[str, Any] = {
            "name": path.name,
            "path": str(path.resolve()),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 3),
            "extension": path.suffix.lower(),
            "type": self._guess_file_type(path.suffix.lower()),
            "modified": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)
            ),
            "created": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(stat.st_ctime)
            ),
        }

        # Try to get row/column counts for data files
        if HAS_PANDAS and path.suffix.lower() in {".csv", ".tsv", ".json", ".parquet"}:
            try:
                df = self.load_dataframe(filepath)
                info["rows"] = len(df)
                info["columns"] = list(df.columns)
                info["n_columns"] = len(df.columns)
            except Exception as exc:
                info["load_error"] = str(exc)

        return info

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------
    def delete_file(self, filepath: str) -> bool:
        """Delete a file from disk.  Returns True on success."""
        path = Path(filepath)
        if not path.exists():
            logger.warning("Cannot delete – file not found: %s", filepath)
            return False

        try:
            os.remove(filepath)
            logger.info("File deleted: %s", filepath)
            return True
        except Exception as exc:
            logger.error("Failed to delete %s: %s", filepath, exc)
            return False

    def list_files(self, directory: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all files in a directory with metadata.

        Parameters
        ----------
        directory:
            Directory to list.  Defaults to the upload directory.
        """
        dir_path = Path(directory) if directory else self.upload_dir

        if not dir_path.exists():
            logger.warning("Directory not found: %s", dir_path)
            return []

        files: List[Dict[str, Any]] = []
        for item in sorted(dir_path.iterdir()):
            if item.is_file():
                try:
                    files.append(self.get_file_info(str(item)))
                except Exception as exc:
                    logger.warning("Error reading file %s: %s", item, exc)
                    files.append({"name": item.name, "error": str(exc)})

        logger.info("Listed %d files in %s", len(files), dir_path)
        return files

    def get_upload_dir(self) -> str:
        """Return the absolute path of the upload directory."""
        return str(self.upload_dir.resolve())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitise_filename(filename: str) -> str:
        """Remove or replace dangerous characters from a filename."""
        # Remove path separators and null bytes
        safe = filename.replace("/", "_").replace("\\", "_").replace("\0", "")
        # Remove leading dots
        safe = safe.lstrip(".")
        # Limit length
        if len(safe) > 255:
            stem = Path(safe).stem[:200]
            suffix = Path(safe).suffix
            safe = stem + suffix
        return safe or "unnamed_file"

    @staticmethod
    def _guess_file_type(extension: str) -> str:
        """Guess human-readable file type from extension."""
        type_map = {
            ".csv": "CSV (Comma-Separated Values)",
            ".tsv": "TSV (Tab-Separated Values)",
            ".json": "JSON (JavaScript Object Notation)",
            ".xlsx": "Excel Workbook",
            ".xls": "Excel Spreadsheet",
            ".parquet": "Apache Parquet",
            ".pkl": "Python Pickle",
            ".h5": "HDF5",
            ".txt": "Plain Text",
        }
        return type_map.get(extension.lower(), "Unknown")
