"""
Model Registry
===============
Tracks and manages registered ML models within the AEGIS system.
Supports registration, retrieval, listing, removal, and active model management.
Uses in-memory dict storage with UUID-based model IDs.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("aegis.services.model_registry")


class ModelRegistry:
    """
    In-memory model registry for tracking ML models.

    Each registered model gets a UUID, metadata, timestamps, and
    can be marked as the active model for the system.
    """

    def __init__(self) -> None:
        """Initialize an empty model registry."""
        self._models: Dict[str, Dict[str, Any]] = {}
        self._active_model_id: Optional[str] = None
        # Bug 23 fix: protect all mutations with an asyncio lock
        self._lock = asyncio.Lock()

        logger.info("ModelRegistry initialized (empty)")

    async def register_async(
        self,
        model_name: str,
        model: Any,
        model_type: str = "sklearn",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Thread-safe async version of register()."""
        async with self._lock:
            return self._register_internal(model_name, model, model_type, metadata)

    def register(
        self,
        model_name: str,
        model: Any,
        model_type: str = "sklearn",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register a model (sync-safe; use register_async from async context)."""
        return self._register_internal(model_name, model, model_type, metadata)

    def _register_internal(
        self,
        model_name: str,
        model: Any,
        model_type: str = "sklearn",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new model in the registry.

        Args:
            model_name: Human-readable name for the model.
            model: The model object (sklearn, xgboost, pytorch, tensorflow, etc.).
            model_type: Framework type: 'sklearn', 'xgboost', 'pytorch', 'tensorflow'.
            metadata: Optional dict of additional metadata (version, description, etc.).

        Returns:
            The generated model_id (UUID string).
        """
        model_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        entry: Dict[str, Any] = {
            "model_id": model_id,
            "model_name": model_name,
            "model": model,
            "model_type": model_type,
            "metadata": metadata or {},
            "registered_at": now,
            "updated_at": now,
            "is_active": False,
        }

        # Store model info (excluding the actual model for safe listing)
        self._models[model_id] = entry

        # If this is the first model, make it active
        if len(self._models) == 1:
            self._active_model_id = model_id
            entry["is_active"] = True
            logger.info("First model registered, set as active: %s (%s)", model_name, model_id)

        logger.info(
            "Model registered: id=%s, name='%s', type=%s",
            model_id, model_name, model_type,
        )
        return model_id

    def get(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a registered model by its ID.

        Args:
            model_id: The UUID of the model.

        Returns:
            Dict with model info including 'model' object, or None if not found.
        """
        entry = self._models.get(model_id)
        if entry is None:
            logger.warning("Model not found: %s", model_id)
            return None
        return entry

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models (without the actual model objects).

        Returns:
            List of dicts with model metadata (model objects excluded for safety).
        """
        result = []
        for model_id, entry in self._models.items():
            safe_entry = {
                "model_id": entry["model_id"],
                "model_name": entry["model_name"],
                "model_type": entry["model_type"],
                "registered_at": entry["registered_at"],
                "updated_at": entry["updated_at"],
                "is_active": entry.get("is_active", False),
                "metadata": entry.get("metadata", {}),
            }
            result.append(safe_entry)
        return result

    async def remove_async(self, model_id: str) -> bool:
        """Async-safe remove."""
        async with self._lock:
            return self._remove_internal(model_id)

    def remove(self, model_id: str) -> bool:
        """Remove a model (sync-safe; use remove_async from async context)."""
        return self._remove_internal(model_id)

    def _remove_internal(self, model_id: str) -> bool:
        """
        Remove a model from the registry.

        If the removed model was active, clears the active model.

        Args:
            model_id: The UUID of the model to remove.

        Returns:
            True if the model was found and removed, False otherwise.
        """
        if model_id not in self._models:
            logger.warning("Cannot remove model not found: %s", model_id)
            return False

        model_name = self._models[model_id]["model_name"]

        # Clear active if this was the active model
        if self._active_model_id == model_id:
            self._active_model_id = None
            logger.info("Removed active model, clearing active: %s", model_id)

        del self._models[model_id]
        logger.info("Model removed: id=%s, name='%s'", model_id, model_name)
        return True

    def get_active(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently active model (sync – use get_active_async from async context).

        Returns:
            Dict with active model info, or None if no active model is set.
        """
        return self._get_active_internal()

    async def get_active_async(self) -> Optional[Dict[str, Any]]:
        """Fix CRIT-06: async-safe get_active that acquires the asyncio lock.

        Use this from async route handlers to prevent data races with
        concurrent register_async() / remove_async() calls.
        """
        async with self._lock:
            return self._get_active_internal()

    def _get_active_internal(self) -> Optional[Dict[str, Any]]:
        """Internal (lock-free) implementation of get_active."""
        if self._active_model_id is None:
            logger.debug("No active model set")
            return None

        entry = self._models.get(self._active_model_id)
        if entry is None:
            # Active ID points to a removed model
            logger.warning("Active model ID %s not found in registry", self._active_model_id)
            self._active_model_id = None
            return None

        return entry

    def set_active(self, model_id: str) -> bool:
        """
        Set a model as the active model (sync – use set_active_async from async context).

        Args:
            model_id: The UUID of the model to set as active.

        Returns:
            True if the model was found and set as active, False otherwise.
        """
        return self._set_active_internal(model_id)

    async def set_active_async(self, model_id: str) -> bool:
        """Fix CRIT-06: async-safe set_active that acquires the asyncio lock.

        Use this from async route handlers to prevent data races with
        concurrent register_async() / remove_async() calls.
        """
        async with self._lock:
            return self._set_active_internal(model_id)

    def _set_active_internal(self, model_id: str) -> bool:
        """Internal (lock-free) implementation of set_active."""
        if model_id not in self._models:
            logger.warning("Cannot set active: model not found: %s", model_id)
            return False

        # Deactivate previous active model
        if self._active_model_id is not None and self._active_model_id in self._models:
            self._models[self._active_model_id]["is_active"] = False

        # Set new active
        self._active_model_id = model_id
        self._models[model_id]["is_active"] = True
        self._models[model_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

        model_name = self._models[model_id]["model_name"]
        logger.info("Active model set: id=%s, name='%s'", model_id, model_name)
        return True

    def find_by_name(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Find models by name (partial match, case-insensitive).

        Args:
            model_name: Name to search for.

        Returns:
            List of matching model entries.
        """
        matches = []
        name_lower = model_name.lower()
        for model_id, entry in self._models.items():
            if name_lower in entry["model_name"].lower():
                matches.append({
                    "model_id": entry["model_id"],
                    "model_name": entry["model_name"],
                    "model_type": entry["model_type"],
                    "registered_at": entry["registered_at"],
                    "is_active": entry.get("is_active", False),
                })
        return matches

    def update_metadata(self, model_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a registered model.

        Args:
            model_id: The UUID of the model.
            metadata: New metadata dict (merged with existing).

        Returns:
            True if model found and updated, False otherwise.
        """
        if model_id not in self._models:
            return False

        self._models[model_id]["metadata"].update(metadata)
        self._models[model_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("Metadata updated for model: %s", model_id)
        return True

    def count(self) -> int:
        """Return the number of registered models."""
        return len(self._models)

    def clear(self) -> None:
        """Remove all models from the registry."""
        self._models.clear()
        self._active_model_id = None
        logger.info("ModelRegistry cleared")

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the registry state."""
        models = self.list_models()
        type_counts: Dict[str, int] = {}
        for m in models:
            t = m.get("model_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        active = self.get_active()
        return {
            "total_models": len(models),
            "active_model": {
                "model_id": active["model_id"],
                "model_name": active["model_name"],
                "model_type": active["model_type"],
            } if active else None,
            "model_types": type_counts,
            "models": models,
        }
