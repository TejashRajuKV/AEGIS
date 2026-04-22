"""
Latent Space Interpolator
==========================
Interpolates between latent representations for visualization
and analysis of the CVAE's learned manifold.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger("aegis.neural.interpolator")


class LatentInterpolator:
    """
    Interpolates in the CVAE latent space for generating
    smooth transitions between samples.
    """

    def __init__(self, cvae: Any, device: str = "cpu"):
        """
        Initialize the interpolator.

        Args:
            cvae: Trained ConditionalVAE model.
            device: Torch device.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")

        self.cvae = cvae
        self.device = torch.device(device)
        self.cvae.to(self.device)
        self.cvae.eval()

    def interpolate(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        n_steps: int = 10,
        condition_a: Optional[np.ndarray] = None,
        condition_b: Optional[np.ndarray] = None,
        method: str = "linear",
    ) -> List[np.ndarray]:
        """
        Interpolate between two samples in latent space.

        Args:
            sample_a: First sample (n_features,).
            sample_b: Second sample (n_features,).
            n_steps: Number of interpolation steps.
            condition_a: Condition for sample_a.
            condition_b: Condition for sample_b.
            method: Interpolation method ('linear' or 'slerp').

        Returns:
            List of decoded samples along the interpolation path.
        """
        self.cvae.eval()
        with torch.no_grad():
            x_a = torch.FloatTensor(sample_a).unsqueeze(0).to(self.device)
            x_b = torch.FloatTensor(sample_b).unsqueeze(0).to(self.device)

            cond_a = (
                torch.FloatTensor(condition_a).unsqueeze(0).to(self.device)
                if condition_a is not None
                else torch.zeros(1, self.cvae.condition_dim, device=self.device)
            )
            cond_b = (
                torch.FloatTensor(condition_b).unsqueeze(0).to(self.device)
                if condition_b is not None
                else torch.zeros(1, self.cvae.condition_dim, device=self.device)
            )

            # Encode both samples
            mu_a, _ = self.cvae.encode(x_a, cond_a)
            mu_b, _ = self.cvae.encode(x_b, cond_b)

            # Interpolate in latent space
            results = []
            for i in range(n_steps + 1):
                t = i / max(n_steps, 1)

                if method == "slerp":
                    z = self._slerp(mu_a, mu_b, t)
                else:
                    z = (1 - t) * mu_a + t * mu_b

                # Interpolate condition too
                cond_t = (1 - t) * cond_a + t * cond_b

                # Decode
                decoded = self.cvae.decode(z, cond_t)
                results.append(decoded.cpu().numpy().flatten())

        return results

    def find_decision_boundary(
        self,
        sample_a: np.ndarray,
        sample_b: np.ndarray,
        classifier: Any,
        n_steps: int = 50,
        condition_a: Optional[np.ndarray] = None,
        condition_b: Optional[np.ndarray] = None,
    ) -> int:
        """
        Find the point along the interpolation path where the
        classifier's decision changes.

        Args:
            sample_a: Sample from class A.
            sample_b: Sample from class B.
            classifier: Binary classifier with predict method.
            n_steps: Number of interpolation steps.
            condition_a: Condition for sample_a.
            condition_b: Condition for sample_b.

        Returns:
            Step index where decision boundary is crossed, or -1.
        """
        interp_samples = self.interpolate(
            sample_a, sample_b, n_steps, condition_a, condition_b
        )

        try:
            X = np.array(interp_samples)
            preds = classifier.predict(X)

            prev_pred = preds[0]
            for i in range(1, len(preds)):
                if preds[i] != prev_pred:
                    logger.info(
                        "Decision boundary found at step %d/%d", i, n_steps
                    )
                    return i
        except Exception as e:
            logger.warning("Decision boundary search failed: %s", e)

        return -1

    @staticmethod
    def _slerp(
        v0: "torch.Tensor", v1: "torch.Tensor", t: float,
        DOT_THRESHOLD: float = 0.9995,
    ) -> "torch.Tensor":
        """
        Spherical linear interpolation.

        Args:
            v0: Start vector.
            v1: End vector.
            t: Interpolation parameter (0=v0, 1=v1).

        Returns:
            Interpolated vector.
        """
        v0_norm = v0 / (torch.norm(v0, dim=-1, keepdim=True) + 1e-8)
        v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)

        dot = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
        dot = torch.clamp(dot, -1.0, 1.0)

        # If vectors are nearly parallel, use linear interpolation
        if torch.abs(dot) > DOT_THRESHOLD:
            return v0 + t * (v1 - v0)

        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)

        s0 = torch.sin((1 - t) * theta) / sin_theta
        s1 = torch.sin(t * theta) / sin_theta

        return s0 * v0 + s1 * v1
