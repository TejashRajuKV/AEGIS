"""Model management endpoints."""
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.models.schemas import ModelRegisterRequest, ModelInfo
from app.services.model_registry import ModelRegistry

router = APIRouter()

# Module-level singleton
_model_registry: ModelRegistry = ModelRegistry()


@router.post("/register")
async def register_model(req: ModelRegisterRequest):
    model_id = _model_registry.register(
        model_name=req.name,
        model=None,  # No actual model object provided in this API
        model_type=req.model_type,
        metadata={
            "version": req.version,
            "dataset_name": req.dataset_name,
            "metrics": req.metrics,
        },
    )
    return {"status": "registered", "model_id": model_id}


@router.get("/list")
async def list_models():
    models = _model_registry.list_models()
    return {"models": models, "count": len(models)}


@router.get("/{name}")
async def get_model(name: str):
    # Search by name since get() expects a UUID
    matches = _model_registry.find_by_name(name)
    if not matches:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return matches[0]


@router.delete("/{name}")
async def delete_model(name: str):
    # Search by name since remove() expects a UUID
    matches = _model_registry.find_by_name(name)
    if not matches:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    success = _model_registry.remove(matches[0]["model_id"])
    if not success:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return {"status": "deleted", "name": name}


# ── Train Request Schema ───────────────────────────────────────────

class TrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    dataset_name: str = Field(..., description="Dataset to train on (e.g. 'adult_census')")
    target_column: str = Field("income", description="Column to predict")
    sensitive_attributes: Optional[List[str]] = Field(
        None, description="Sensitive attribute columns (informational only)"
    )
    test_size: float = Field(0.2, ge=0.05, le=0.5, description="Train/test split ratio")
    random_state: int = Field(42, description="RNG seed for reproducibility")
    retrain: bool = Field(False, description="Force retraining even if model already cached")
    model_params: Dict[str, Any] = Field(
        default_factory=dict, description="Extra kwargs forwarded to the model constructor"
    )


# Supported model map — extend as needed
_MODEL_MAP = {
    "logistic_regression": lambda rs, **kw: __import__(
        "sklearn.linear_model", fromlist=["LogisticRegression"]
    ).LogisticRegression(max_iter=1000, random_state=rs, **kw),
    "random_forest": lambda rs, **kw: __import__(
        "sklearn.ensemble", fromlist=["RandomForestClassifier"]
    ).RandomForestClassifier(n_estimators=100, random_state=rs, **kw),
    "xgboost": lambda rs, **kw: __import__(
        "xgboost", fromlist=["XGBClassifier"]
    ).XGBClassifier(n_estimators=100, random_state=rs, use_label_encoder=False,
                    eval_metric="logloss", **kw),
}


@router.post("/{model_id}/train")
async def train_model(model_id: str, body: TrainRequest) -> Dict[str, Any]:
    """
    Train a model on a dataset and store it in the registry.

    Supported model_ids: ``logistic_regression``, ``random_forest``, ``xgboost``.
    Train once, reuse everywhere — the fairness audit will pick up the cached model.
    """
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    if model_id not in _MODEL_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_id '{model_id}'. Supported: {list(_MODEL_MAP.keys())}",
        )

    # Load dataset
    from app.data.dataset_loader import get_dataset_loader
    loader = get_dataset_loader()
    try:
        df = loader.load_dataset(body.dataset_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if body.target_column not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=f"Target column '{body.target_column}' not found in dataset. "
                   f"Available columns: {list(df.columns)}",
        )

    # Features: use numeric columns only (safe default)
    feature_cols = [
        c for c in df.columns
        if c != body.target_column
        and df[c].dtype.kind in "iuf"   # int, unsigned int, float
    ]
    if not feature_cols:
        raise HTTPException(
            status_code=422,
            detail="No numeric feature columns found after excluding target column.",
        )

    X = df[feature_cols].fillna(0).values
    y = LabelEncoder().fit_transform(df[body.target_column].values)

    # Bug 11 fix: compute real test accuracy on a held-out split
    from sklearn.model_selection import train_test_split as _tts
    X_train, X_test, y_train, y_test = _tts(
        X, y, test_size=body.test_size, random_state=body.random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Build and fit model
    try:
        model = _MODEL_MAP[model_id](body.random_state, **body.model_params)
        model.fit(X_train_scaled, y_train)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Training failed: {exc}")

    # Evaluate on test set (not training set)
    y_pred_test = model.predict(X_test_scaled)
    test_accuracy  = float(np.mean(y_pred_test == y_test))
    # Training accuracy for reference only
    y_pred_train = model.predict(X_train_scaled)
    train_accuracy = float(np.mean(y_pred_train == y_train))

    # Register in model registry
    model_id_registered = _model_registry.register(
        model_name=model_id,
        model=model,
        model_type=model_id,
        metadata={
            "dataset_name": body.dataset_name,
            "target_column": body.target_column,
            "n_samples": len(y),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "n_features": X_train_scaled.shape[1],
            "feature_cols": feature_cols,
            "train_accuracy": round(train_accuracy, 4),
            "test_accuracy":  round(test_accuracy, 4),
        },
    )

    return {
        "model_id": model_id_registered,
        "status": "trained",
        "dataset": body.dataset_name,
        "target_column": body.target_column,
        "n_samples": len(y),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": X_train_scaled.shape[1],
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy":  round(test_accuracy, 4),
    }
