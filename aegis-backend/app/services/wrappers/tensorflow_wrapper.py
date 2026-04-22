"""
TensorFlowModelWrapper – wraps TensorFlow/Keras models for AEGIS fairness evaluation.

Supports prediction, probability output, batched inference, and model introspection.
Uses try-except for optional TensorFlow import.
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
    import tensorflow as tf
    from tensorflow import keras

    HAS_TF = True
except ImportError:
    tf = None  # type: ignore[assignment]
    keras = None  # type: ignore[assignment]
    HAS_TF = False


class TensorFlowModelWrapper:
    """Wraps a TensorFlow/Keras model for use in the AEGIS pipeline.

    Provides a uniform interface for prediction, probability output,
    model information, and evaluation.

    Parameters
    ----------
    model:
        A ``tf.keras.Model`` or ``tf.Module`` instance.
    input_shape:
        Expected input shape (e.g. ``(n_features,)``).
    batch_size:
        Default batch size for batched inference.
    """

    def __init__(
        self,
        model: Any,
        input_shape: Optional[Tuple[int, ...]] = None,
        batch_size: int = 64,
    ) -> None:
        if not HAS_TF:
            raise ImportError(
                "TensorFlow is not installed. Install it with: pip install tensorflow"
            )

        if not isinstance(model, (keras.Model, tf.Module)):
            raise TypeError(
                f"Expected tf.keras.Model or tf.Module, got {type(model).__name__}"
            )

        self.model = model
        self.input_shape = input_shape
        self.batch_size = batch_size

        # Count parameters
        self._num_params = sum(
            int(w.numpy().size) for w in model.trainable_variables
        ) + sum(
            int(w.numpy().size) for w in model.non_trainable_variables
        )

        # Set to inference mode
        if hasattr(self.model, "eval"):
            self.model.eval()

        logger.info(
            "TensorFlowModelWrapper – params=%d, batch_size=%d, input_shape=%s",
            self._num_params,
            self.batch_size,
            self.input_shape,
        )

    def predict(self, X: Any) -> "np.ndarray":
        """Return class predictions for input *X*.

        Parameters
        ----------
        X :
            Input array-like of shape ``(n_samples, *input_shape)``.
            Accepts numpy arrays, lists, or tf tensors.

        Returns
        -------
        np.ndarray of shape ``(n_samples,)`` with predicted class labels.
        """
        if not HAS_TF:
            raise ImportError("TensorFlow is not installed")

        proba = self.predict_proba(X)
        if HAS_NUMPY:
            return np.argmax(proba, axis=1)
        return [int(list(row).index(max(row))) for row in proba]

    def predict_proba(self, X: Any) -> "np.ndarray":
        """Return class probabilities for input *X*.

        Parameters
        ----------
        X :
            Input data (numpy array, list, or tf tensor).

        Returns
        -------
        np.ndarray of shape ``(n_samples, n_classes)``.
        """
        if not HAS_TF:
            raise ImportError("TensorFlow is not installed")

        all_outputs: List[Any] = []

        # Convert to tensor
        tensor = self._to_tensor(X)
        n_samples = tensor.shape[0]

        # Batched inference
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch = tensor[start:end]

            # Run inference
            outputs = self.model(batch, training=False)

            # Apply softmax if outputs are logits (not already probabilities)
            if isinstance(outputs, tf.Tensor):
                outputs_np = outputs.numpy()
                # Check if outputs look like probabilities (sum to ~1 per row)
                if outputs_np.ndim == 2 and outputs_np.shape[1] > 1:
                    row_sums = outputs_np.sum(axis=1)
                    if not np.allclose(row_sums, 1.0, atol=0.1):
                        # Apply softmax
                        outputs_np = tf.nn.softmax(outputs, axis=-1).numpy()
                    all_outputs.append(outputs_np)
                elif outputs_np.ndim == 1:
                    # Binary case: sigmoid
                    probs = tf.nn.sigmoid(outputs).numpy()
                    all_outputs.append(np.column_stack([1.0 - probs, probs]))
                else:
                    all_outputs.append(outputs_np)
            elif isinstance(outputs, (tuple, list)):
                logits = outputs[0] if outputs else outputs
                if not isinstance(logits, tf.Tensor):
                    logits = tf.constant(logits, dtype=tf.float32)
                probs = tf.nn.softmax(logits, axis=-1).numpy()
                all_outputs.append(probs)
            else:
                all_outputs.append(tf.constant(outputs, dtype=tf.float32).numpy())

        if HAS_NUMPY:
            return np.vstack(all_outputs)
        return all_outputs[0]

    def get_model_info(self) -> Dict[str, Any]:
        """Return model architecture and parameter information.

        Returns
        -------
        Dict with framework, architecture name, parameter counts, layers, etc.
        """
        if not HAS_TF:
            raise ImportError("TensorFlow is not installed")

        info: Dict[str, Any] = {
            "framework": "tensorflow",
            "architecture": self.model.__class__.__name__,
            "num_parameters": self._num_params,
            "input_shape": self.input_shape,
            "batch_size": self.batch_size,
            "trainable_parameters": sum(
                int(w.numpy().size) for w in self.model.trainable_variables
            ),
            "non_trainable_parameters": sum(
                int(w.numpy().size) for w in self.model.non_trainable_variables
            ),
        }

        # Get layer information
        if hasattr(self.model, "layers"):
            layer_names = []
            for layer in self.model.layers:
                layer_info = {
                    "name": layer.name,
                    "class": layer.__class__.__name__,
                }
                if hasattr(layer, "output_shape"):
                    layer_info["output_shape"] = str(layer.output_shape)
                if hasattr(layer, "count_params"):
                    layer_info["params"] = layer.count_params()
                layer_names.append(layer_info)
            info["layers"] = layer_names
            info["n_layers"] = len(layer_names)
        else:
            info["n_layers"] = 0

        # Try to get output shape
        if self.input_shape is not None:
            try:
                dummy = tf.zeros((1,) + tuple(self.input_shape), dtype=tf.float32)
                out = self.model(dummy, training=False)
                if isinstance(out, tf.Tensor):
                    info["output_shape"] = list(out.shape)
                elif isinstance(out, (tuple, list)) and len(out) > 0:
                    first = out[0] if isinstance(out[0], tf.Tensor) else tf.constant(out[0])
                    info["output_shape"] = list(first.shape)
            except Exception as exc:
                info["output_shape_error"] = str(exc)

        # Get model summary as string if available
        if hasattr(self.model, "summary"):
            import io
            string_buffer = io.StringIO()
            self.model.summary(print_fn=lambda x: string_buffer.write(x + "\n"))
            info["summary"] = string_buffer.getvalue()

        return info

    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        """Evaluate the model on given data.

        Parameters
        ----------
        X :
            Input features.
        y :
            Ground truth labels (integer class indices).

        Returns
        -------
        Dict with ``accuracy``, ``loss``, and ``n_samples``.
        """
        if not HAS_TF:
            raise ImportError("TensorFlow is not installed")

        tensor_X = self._to_tensor(X)
        tensor_y = self._to_labels(y)

        all_preds: List[Any] = []
        correct = 0
        total = 0
        total_loss = 0.0
        n_batches = 0

        n_samples = tensor_X.shape[0]
        n_classes = self._estimate_n_classes(tensor_y)

        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_X = tensor_X[start:end]
            batch_y = tensor_y[start:end]

            # Run inference
            outputs = self.model(batch_X, training=False)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # Compute loss (sparse categorical crossentropy)
            if isinstance(outputs, tf.Tensor):
                outputs_np = outputs.numpy()
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    batch_y, outputs_np, from_logits=True
                )
                total_loss += float(tf.reduce_mean(loss).numpy())
                n_batches += 1

                # Compute accuracy
                preds = np.argmax(outputs_np, axis=-1)
            else:
                preds = np.array(outputs)

            y_batch = batch_y.numpy() if hasattr(batch_y, "numpy") else np.array(batch_y)
            correct += int(np.sum(preds == y_batch))
            total += len(y_batch)
            all_preds.append(preds)

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / max(total, 1)

        metrics = {
            "accuracy": round(accuracy, 4),
            "loss": round(avg_loss, 4),
            "n_samples": total,
        }
        logger.info("TF Evaluation: accuracy=%.4f, loss=%.4f (n=%d)", accuracy, avg_loss, total)
        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, X: Any) -> "tf.Tensor":
        """Convert input to a tf.Tensor."""
        if isinstance(X, tf.Tensor):
            return tf.cast(X, tf.float32)
        if HAS_NUMPY and isinstance(X, np.ndarray):
            return tf.constant(X, dtype=tf.float32)
        return tf.constant(X, dtype=tf.float32)

    def _to_labels(self, y: Any) -> "tf.Tensor":
        """Convert labels to a 1-D int32 tensor."""
        if isinstance(y, tf.Tensor):
            return tf.cast(tf.reshape(y, [-1]), tf.int32)
        if HAS_NUMPY and isinstance(y, np.ndarray):
            return tf.constant(y.reshape(-1), dtype=tf.int32)
        return tf.constant(y, dtype=tf.int32)

    def _estimate_n_classes(self, y: "tf.Tensor") -> int:
        """Estimate the number of classes from labels."""
        if HAS_NUMPY:
            y_np = y.numpy() if hasattr(y, "numpy") else np.array(y)
            return int(len(np.unique(y_np)))
        return 2

    def __repr__(self) -> str:
        return (
            f"TensorFlowModelWrapper(model={self.model.__class__.__name__}, "
            f"params={self._num_params}, batch_size={self.batch_size})"
        )
