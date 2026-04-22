"""Dataset management endpoints."""
from fastapi import APIRouter, HTTPException, Query
from typing import Any, Dict

try:
    from app.data.dataset_loader import get_dataset_loader, DatasetLoader
    _HAS_LOADER = True
except ImportError as _exc:
    get_dataset_loader = None  # type: ignore[assignment]
    DatasetLoader = None  # type: ignore[assignment, misc]
    _HAS_LOADER = False
    import logging as _log
    _log.getLogger(__name__).warning("DatasetLoader import failed: %s", _exc)

try:
    from app.data.schema_validator import get_schema_validator
    _HAS_VALIDATOR = True
except ImportError as _exc:
    get_schema_validator = None  # type: ignore[assignment]
    _HAS_VALIDATOR = False
    import logging as _log
    _log.getLogger(__name__).warning("SchemaValidator import failed: %s", _exc)

try:
    from app.config import settings
except ImportError:
    settings = None  # type: ignore[assignment]

import logging
logger = logging.getLogger(__name__)

router = APIRouter()


def _require_loader():
    """Raise 503 if DatasetLoader is unavailable."""
    if not _HAS_LOADER or get_dataset_loader is None:
        raise HTTPException(status_code=503, detail="DatasetLoader is not available.")


@router.get("/list")
async def list_datasets():
    _require_loader()
    loader = get_dataset_loader()
    available = list(loader.list_datasets().keys())
    schemas = {}
    for name in available:
        try:
            schemas[name] = loader.get_dataset_info(name)
        except Exception as exc:
            # Bug 13 fix: log the real error instead of swallowing it silently
            logger.warning("Could not load schema for '%s': %s", name, exc)
            schemas[name] = {"error": f"Could not load schema: {exc}"}
    return {"available": available, "schemas": schemas}


@router.get("/load/{name}")
async def load_data(name: str):
    _require_loader()
    try:
        loader = get_dataset_loader()
        df = loader.load_dataset(name)
        info = loader.get_dataset_info(name)

        # Bug 13 fix: log schema validation errors rather than silently skipping
        validation = {"status": "skipped", "reason": "No matching schema registered"}
        if _HAS_VALIDATOR:
            try:
                validator = get_schema_validator()
                if name in validator.schemas:
                    validation = validator.validate(df, name)
            except Exception as exc:
                logger.warning("Schema validation error for '%s': %s", name, exc)
                validation = {"status": "error", "reason": str(exc)}

        return {
            "name": name,
            "rows": len(df),
            "columns": list(df.columns),
            "validation": validation,
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/schema/{name}")
async def get_schema(name: str):
    try:
        loader = get_dataset_loader()
        return loader.get_dataset_info(name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/sample/{name}")
async def get_dataset_sample(
    name: str,
    n: int = Query(default=10, ge=1, le=500, description="Number of rows to preview"),
) -> Dict[str, Any]:
    """Return first n rows as JSON records. Powers frontend data-preview tables."""
    try:
        loader = get_dataset_loader()
        df = loader.load_dataset(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sample = df.head(n).fillna("").to_dict(orient="records")
    return {
        "dataset": name,
        "n_requested": n,
        "n_returned": len(sample),
        "total_rows": len(df),
        "columns": list(df.columns),
        "sample": sample,
    }


@router.get("/stats/{name}")
async def get_dataset_stats(name: str) -> Dict[str, Any]:
    """Return descriptive statistics. Powers frontend EDA panels."""
    try:
        loader = get_dataset_loader()
        df = loader.load_dataset(name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    numeric_desc = df.describe().round(4).to_dict()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_stats: Dict[str, Any] = {}
    for col in cat_cols:
        vc = df[col].value_counts().head(10).to_dict()
        cat_stats[col] = {
            "n_unique": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in vc.items()},
        }

    return {
        "dataset": name,
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "column_names": list(df.columns),
        "null_counts": {k: int(v) for k, v in df.isnull().sum().items()},
        "numeric_stats": numeric_desc,
        "categorical_stats": cat_stats,
    }
