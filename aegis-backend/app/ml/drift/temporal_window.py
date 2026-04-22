"""
Temporal Window
================
Sliding window manager for drift detection.
Provides reference and test windows from a data stream.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)


class TemporalWindow:
    """
    Sliding window that splits a data stream into reference and test windows.

    Usage:
        window = TemporalWindow(reference_size=500, test_size=100)
        window.fit(reference_data)
        for batch in data_stream:
            result = window.update(batch)
            if result["is_ready"]:
                ref, test = window.get_windows()
    """

    def __init__(
        self,
        reference_size: int = 500,
        test_size: int = 100,
        stride: int = 50,
    ):
        """
        Args:
            reference_size: Number of samples in the reference window.
            test_size: Number of samples in each test window.
            stride: Step size for sliding the test window.
        """
        self.reference_size = reference_size
        self.test_size = test_size
        self.stride = stride

        self.reference_data: Optional[np.ndarray] = None
        self.test_buffer: deque = deque(maxlen=test_size)
        self.total_seen: int = 0
        self.is_reference_set: bool = False

    def fit(self, reference_data: np.ndarray) -> None:
        """
        Set the reference window.

        Args:
            reference_data: Reference distribution data.
        """
        self.reference_data = np.asarray(reference_data, dtype=np.float64)
        if self.reference_data.ndim > 1:
            self.reference_data = self.reference_data.ravel()
        self.reference_data = self.reference_data[-self.reference_size:]
        self.is_reference_set = True
        self.test_buffer.clear()
        self.total_seen = 0
        logger.info(f"Reference window set with {len(self.reference_data)} samples")

    def update(self, new_data: np.ndarray) -> Dict:
        """
        Add new data to the test buffer.

        Args:
            new_data: New observations (n,) or (n, d).

        Returns:
            Dict with is_ready, buffer_size, reference_size.
        """
        data = np.asarray(new_data, dtype=np.float64).ravel()
        for val in data:
            self.test_buffer.append(float(val))
            self.total_seen += 1

        return {
            "is_ready": len(self.test_buffer) >= self.test_size,
            "buffer_size": len(self.test_buffer),
            "total_seen": self.total_seen,
            "is_reference_set": self.is_reference_set,
        }

    def get_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current reference and test windows.

        Returns:
            Tuple of (reference_array, test_array).

        Raises:
            ValueError if not ready.
        """
        if not self.is_reference_set:
            raise ValueError("Reference window not set. Call fit() first.")
        if len(self.test_buffer) < self.test_size:
            raise ValueError("Test buffer not full yet.")

        return (
            self.reference_data.copy(),
            np.array(list(self.test_buffer)),
        )

    def advance(self, new_data: Optional[np.ndarray] = None) -> Dict:
        """
        Slide the test window forward.

        Args:
            new_data: Optional new data to add.

        Returns:
            Updated state dict.
        """
        if new_data is not None:
            self.update(new_data)

        # Drop oldest elements to simulate sliding
        drop_count = self.stride
        while drop_count > 0 and self.test_buffer:
            self.test_buffer.popleft()
            drop_count -= 1

        return self.get_status()

    def get_status(self) -> Dict:
        """Get current window status."""
        return {
            "is_reference_set": self.is_reference_set,
            "reference_size": len(self.reference_data) if self.reference_data is not None else 0,
            "test_buffer_size": len(self.test_buffer),
            "is_ready": (
                self.is_reference_set and len(self.test_buffer) >= self.test_size
            ),
            "total_seen": self.total_seen,
        }

    def reset(self) -> None:
        """Reset all state."""
        self.reference_data = None
        self.test_buffer.clear()
        self.total_seen = 0
        self.is_reference_set = False
