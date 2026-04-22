"""
Pareto reward modifier — avoids Goodhart's Law via Pareto domination checks.

In a multi-objective setting a *solution* is a point in the objective space:

    (accuracy, DP_gap, EO_gap, calibration_error)

Lower gaps/errors and higher accuracy are better.  This module maintains an
approximate Pareto front and uses it to scale the base reward:

- If the new solution **Pareto-dominates** at least one point on the front,
  the agent receives a bonus multiplier (up to 2×).
- If the new solution is **dominated** by every point on the front, a
  penalty multiplier (< 1×) is applied.
- Otherwise the multiplier is 1.0 (neutral).
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from app.ml.rl.reward_shaper import FairnessMetrics

logger = logging.getLogger(__name__)


@dataclass
class ParetoConfig:
    """Hyper-parameters for the Pareto reward modifier."""

    domination_bonus: float = 0.5       # bonus added when dominating
    dominated_penalty: float = -0.3     # penalty when fully dominated
    epsilon: float = 1e-6               # numerical tolerance
    max_front_size: int = 200           # memory cap for the Pareto front


@dataclass
class ParetoSolution:
    """A single solution point in the multi-objective space."""

    accuracy: float
    dp_gap: float
    eo_gap: float
    calibration_error: float
    step: int = 0

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.accuracy, self.dp_gap, self.eo_gap, self.calibration_error)

    def to_array(self) -> np.ndarray:
        return np.array(self.to_tuple(), dtype=np.float64)

    @classmethod
    def from_metrics(
        cls, metrics: FairnessMetrics, step: int = 0
    ) -> "ParetoSolution":
        return cls(
            accuracy=metrics.accuracy,
            dp_gap=metrics.dp_gap,
            eo_gap=metrics.eo_gap,
            calibration_error=metrics.calibration_diff,
            step=step,
        )


class ParetoRewardModifier:
    """Implements Pareto-domination checks to avoid Goodhart's Law.

    Parameters
    ----------
    config : Optional[ParetoConfig]
        Modifier hyper-parameters.
    """

    def __init__(self, config: Optional[ParetoConfig] = None) -> None:
        self.config = config if config is not None else ParetoConfig()
        # Pareto front stored as a list of ParetoSolution
        self._pareto_front: List[ParetoSolution] = []
        logger.info("ParetoRewardModifier initialised (ε=%.6f, max_front=%d)",
                     self.config.epsilon, self.config.max_front_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_pareto_dominated(
        self, solution: ParetoSolution, pareto_front: Optional[List[ParetoSolution]] = None
    ) -> bool:
        """Check whether *solution* is dominated by any point on *pareto_front*.

        A solution **a** is dominated by **b** when **b** is at least as good
        on every objective and strictly better on at least one.

        For accuracy, *higher* is better.
        For gap/error metrics, *lower* is better.

        Parameters
        ----------
        solution : ParetoSolution
            Candidate solution.
        pareto_front : Optional[List[ParetoSolution]]
            Reference front.  When *None* the internal front is used.

        Returns
        -------
        bool
            ``True`` if the solution is dominated by at least one point.
        """
        front = pareto_front if pareto_front is not None else self._pareto_front
        if len(front) == 0:
            return False
        eps = self.config.epsilon

        sol = solution.to_array()
        for p in front:
            other = p.to_array()
            # other is at least as good on all objectives
            acc_ok = other[0] >= sol[0] - eps       # accuracy: higher better
            dp_ok = other[1] <= sol[1] + eps        # dp_gap: lower better
            eo_ok = other[2] <= sol[2] + eps        # eo_gap: lower better
            cal_ok = other[3] <= sol[3] + eps       # calibration: lower better

            if acc_ok and dp_ok and eo_ok and cal_ok:
                # At least as good on all — check strictly better on at least one
                strictly_better = (
                    (other[0] > sol[0] + eps)
                    or (other[1] < sol[1] - eps)
                    or (other[2] < sol[2] - eps)
                    or (other[3] < sol[3] - eps)
                )
                if strictly_better:
                    return True
        return False

    def dominates_any(
        self, solution: ParetoSolution, pareto_front: Optional[List[ParetoSolution]] = None
    ) -> bool:
        """Check whether *solution* dominates at least one point on the front.

        Returns
        -------
        bool
            ``True`` if the solution dominates at least one front member.
        """
        front = pareto_front if pareto_front is not None else self._pareto_front
        if len(front) == 0:
            return False
        eps = self.config.epsilon

        sol = solution.to_array()
        for p in front:
            other = p.to_array()
            acc_ok = sol[0] >= other[0] - eps
            dp_ok = sol[1] <= other[1] + eps
            eo_ok = sol[2] <= other[2] + eps
            cal_ok = sol[3] <= other[3] + eps

            if acc_ok and dp_ok and eo_ok and cal_ok:
                strictly_better = (
                    (sol[0] > other[0] + eps)
                    or (sol[1] < other[1] - eps)
                    or (sol[2] < other[2] - eps)
                    or (sol[3] < other[3] - eps)
                )
                if strictly_better:
                    return True
        return False

    def update_pareto_front(
        self, solution: ParetoSolution
    ) -> List[ParetoSolution]:
        """Add *solution* to the Pareto front, pruning dominated points.

        Returns
        -------
        List[ParetoSolution]
            Updated Pareto front.
        """
        # Remove points dominated by the new solution
        pruned = [
            p for p in self._pareto_front
            if not self._dominates(solution, p)
        ]
        # Only add if not dominated by any remaining point
        dominated = False
        eps = self.config.epsilon
        sol = solution.to_array()
        for p in pruned:
            other = p.to_array()
            if (
                other[0] >= sol[0] - eps
                and other[1] <= sol[1] + eps
                and other[2] <= sol[2] + eps
                and other[3] <= sol[3] + eps
            ):
                # other is at least as good — check strictly better
                if (
                    (other[0] > sol[0] + eps)
                    or (other[1] < sol[1] - eps)
                    or (other[2] < sol[2] - eps)
                    or (other[3] < sol[3] - eps)
                ):
                    dominated = True
                    break

        if not dominated:
            pruned.append(solution)

        # Memory cap: keep the best max_front_size solutions by a simple score
        if len(pruned) > self.config.max_front_size:
            scored = []
            for p in pruned:
                arr = p.to_array()
                score = arr[0] - (arr[1] + arr[2] + arr[3]) / 3.0
                scored.append((score, p))
            scored.sort(key=lambda x: x[0], reverse=True)
            pruned = [p for _, p in scored[: self.config.max_front_size]]

        self._pareto_front = pruned
        logger.debug(
            "Pareto front updated: %d members (added=%s)",
            len(self._pareto_front),
            not dominated,
        )
        return self._pareto_front

    def get_pareto_reward_multiplier(
        self, solution: ParetoSolution
    ) -> float:
        """Compute a reward multiplier based on Pareto domination.

        - If *solution* dominates ≥1 front member → bonus (multiplier > 1)
        - If *solution* is dominated by all front members → penalty (multiplier < 1)
        - Otherwise → neutral (multiplier == 1)

        Returns
        -------
        float
            Multiplier in the range ``[0.5, 2.0]``.
        """
        front = self._pareto_front

        if len(front) == 0:
            # Empty front: neutral
            return 1.0

        if self.dominates_any(solution, front):
            multiplier = 1.0 + self.config.domination_bonus
            logger.debug("Pareto bonus: multiplier=%.2f", multiplier)
            return multiplier

        if self.is_pareto_dominated(solution, front):
            multiplier = 1.0 + self.config.dominated_penalty
            logger.debug("Pareto penalty: multiplier=%.2f", multiplier)
            return multiplier

        return 1.0

    def get_pareto_front(self) -> List[ParetoSolution]:
        """Return a copy of the current Pareto front."""
        return list(self._pareto_front)

    def front_size(self) -> int:
        return len(self._pareto_front)

    def reset(self) -> None:
        """Clear the Pareto front."""
        self._pareto_front = []
        logger.info("Pareto front reset.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dominates(self, a: ParetoSolution, b: ParetoSolution) -> bool:
        """Return True if *a* dominates *b* (a is better everywhere or equal)."""
        eps = self.config.epsilon
        aa = a.to_array()
        bb = b.to_array()
        at_least_as_good = (
            aa[0] >= bb[0] - eps  # accuracy higher or equal
            and aa[1] <= bb[1] + eps  # dp_gap lower or equal
            and aa[2] <= bb[2] + eps  # eo_gap lower or equal
            and aa[3] <= bb[3] + eps  # calibration lower or equal
        )
        if not at_least_as_good:
            return False
        strictly_better = (
            (aa[0] > bb[0] + eps)
            or (aa[1] < bb[1] - eps)
            or (aa[2] < bb[2] - eps)
            or (aa[3] < bb[3] - eps)
        )
        return strictly_better
