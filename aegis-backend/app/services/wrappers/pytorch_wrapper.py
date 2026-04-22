"""
PyTorchModelWrapper – wraps PyTorch models for AEGIS fairness evaluation.

Supports prediction, probability output, batching, and device management.
Uses try-except for optional PyTorch import.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    HAS_TORCH = False


class PyTorchModelWrapper:
    """Wraps a PyTorch :class:`nn.Module` for use in the AEGIS pipeline.

    Provides a uniform interface (duck-typed with :class:`BaseWrapper`)
    for prediction, probability output, and evaluation.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` instance.
    input_shape:
        Expected input shape (e.g. ``(n_features,)``).
    device:
        ``'cpu'`` or ``'cuda'``.
    batch_size:
        Default batch size for batched inference.
    """

    def __init__(
        self,
        model: Any,
        input_shape: Optional[Tuple[int, ...]] = None,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is not installed. Install it with: pip install torch"
            )

        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Expected torch.nn.Module, got {type(model).__name__}"
            )

        self.model = model
        self.input_shape = input_shape
        self.device = torch.device(device)
        self.batch_size = batch_size

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        # Count parameters
        self._num_params = sum(p.numel() for p in self.model.parameters())

        logger.info(
            "PyTorchModelWrapper – device=%s, params=%d, batch_size=%d",
            self.device,
            self._num_params,
            self.batch_size,
        )

    # ------------------------------------------------------------------
    # Prediction interface
    # ------------------------------------------------------------------
    def predict(self, X: Any) -> "np.ndarray":
        """Return class predictions for input *X*.

        Parameters
        ----------
        X :
            Input array-like of shape ``(n_samples, *input_shape)``.
            Accepts numpy arrays, lists, or torch tensors.

        Returns
        -------
        np.ndarray of shape ``(n_samples,)`` with predicted class labels.
        """
        proba = self.predict_proba(X)
        if HAS_NUMPY:
            return np.argmax(proba, axis=1)
        return [int(list(row).index(max(row))) for row in proba]

    def predict_proba(self, X: Any) -> "np.ndarray":
        """Return class probabilities for input *X*.

        Parameters
        ----------
        X :
            Input data (numpy array, list, or torch tensor).

        Returns
        -------
        np.ndarray of shape ``(n_samples, n_classes)``.
        """
        self.model.eval()
        all_outputs: List["np.ndarray"] = []

        # Convert to tensor
        tensor = self._to_tensor(X)

        # Batched inference
        n_samples = tensor.shape[0]
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch = tensor[start:end].to(self.device)

            with torch.no_grad():
                outputs = self.model(batch)

            # Handle different output formats
            if isinstance(outputs, torch.Tensor):
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
            elif isinstance(outputs, (tuple, list)):
                # Some models return (logits, ...) tuples
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                if not isinstance(logits, torch.Tensor):
                    logits = torch.tensor(logits)
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            else:
                probs = torch.softmax(torch.tensor(outputs), dim=-1).cpu().numpy()

            if HAS_NUMPY:
                all_outputs.append(np.array(probs))
            else:
                all_outputs.append(np.array(probs) if HAS_NUMPY else probs)

        if HAS_NUMPY:
            return np.vstack(all_outputs)
        return all_outputs[0]

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------
    def to(self, device: str) -> "PyTorchModelWrapper":
        """Move the model to a different device.

        Parameters
        ----------
        device:
            ``'cpu'`` or ``'cuda'``.

        Returns
        -------
        self (for method chaining)
        """
        self.device = torch.device(device)
        self.model.to(self.device)
        logger.info("Model moved to %s", self.device)
        return self

    # ------------------------------------------------------------------
    # Model information
    # ------------------------------------------------------------------
    def get_model_info(self) -> Dict[str, Any]:
        """Return model architecture and parameter information."""
        info: Dict[str, Any] = {
            "framework": "pytorch",
            "architecture": self._get_architecture_name(),
            "num_parameters": self._num_params,
            "device": str(self.device),
            "input_shape": self.input_shape,
            "batch_size": self.batch_size,
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "layers": self._count_layers(),
        }

        # Try to get output shape
        if self.input_shape is not None:
            try:
                dummy = torch.zeros(1, *self.input_shape).to(self.device)
                with torch.no_grad():
                    out = self.model(dummy)
                if isinstance(out, torch.Tensor):
                    info["output_shape"] = tuple(out.shape)
                elif isinstance(out, (tuple, list)) and len(out) > 0:
                    first = out[0] if isinstance(out[0], torch.Tensor) else torch.tensor(out[0])
                    info["output_shape"] = tuple(first.shape)
            except Exception as exc:
                info["output_shape_error"] = str(exc)

        return info

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        X: Any,
        y: Any,
        loss_fn: str = "cross_entropy",
    ) -> Dict[str, float]:
        """Evaluate the model on given data.

        Parameters
        ----------
        X :
            Input features.
        y :
            Ground truth labels (integer class indices).
        loss_fn :
            ``'cross_entropy'`` or ``'mse'``.

        Returns
        -------
        Dict with ``accuracy``, ``loss``, and ``n_samples``.
        """
        self.model.eval()
        tensor_X = self._to_tensor(X)
        tensor_y = self._to_labels(y, tensor_X.shape[0])

        all_preds: List[Any] = []
        all_losses: List[float] = []
        correct = 0
        total = 0

        n_samples = tensor_X.shape[0]
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_X = tensor_X[start:end].to(self.device)
            batch_y = tensor_y[start:end].to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_X)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                # Loss
                if loss_fn == "mse":
                    loss = torch.nn.functional.mse_loss(outputs.float(), batch_y.float())
                else:
                    loss = torch.nn.functional.cross_entropy(
                        outputs.float(), batch_y.long()
                    )
                all_losses.append(loss.item())

                # Accuracy
                preds = torch.argmax(outputs, dim=-1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.shape[0]
                all_preds.append(preds.cpu())

        avg_loss = sum(all_losses) / max(len(all_losses), 1)
        accuracy = correct / max(total, 1)

        metrics = {
            "accuracy": round(accuracy, 4),
            "loss": round(avg_loss, 4),
            "n_samples": total,
        }
        logger.info("Evaluation: accuracy=%.4f, loss=%.4f (n=%d)", accuracy, avg_loss, total)
        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, X: Any) -> "torch.Tensor":
        """Convert input to a torch tensor on the correct device."""
        if isinstance(X, torch.Tensor):
            return X.float()
        if HAS_NUMPY and isinstance(X, np.ndarray):
            return torch.tensor(X, dtype=torch.float32)
        return torch.tensor(X, dtype=torch.float32)

    def _to_labels(self, y: Any, expected_size: int) -> "torch.Tensor":
        """Convert labels to a 1-D long tensor."""
        if isinstance(y, torch.Tensor):
            return y.long().flatten()
        if HAS_NUMPY and isinstance(y, np.ndarray):
            return torch.tensor(y, dtype=torch.long).flatten()
        return torch.tensor(y, dtype=torch.long).flatten()

    def _get_architecture_name(self) -> str:
        """Get a human-readable model architecture name."""
        return self.model.__class__.__name__

    def _count_layers(self) -> int:
        """Count the number of nn.Module sub-layers."""
        return sum(1 for _ in self.model.modules())

    def __repr__(self) -> str:
        return (
            f"PyTorchModelWrapper(model={self._get_architecture_name()}, "
            f"device={self.device}, params={self._num_params})"
        )
