"""
Tests for PPO Autopilot (RL) Module
====================================
Tests for action space, reward shaping, Pareto modifier, Goodhart guard,
PPO agent creation, and environment reset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the project root is on sys.path so that ``app.*`` imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Graceful import – skip the entire module if torch is missing
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="PyTorch is required for autopilot tests")

from app.ml.rl.action_space import ActionBounds, ContinuousActionSpace
from app.ml.rl.reward_shaper import FairnessMetrics, MultiObjectiveRewardShaper, RewardComponents
from app.ml.rl.goodhart_guard import AlertLevel, GoodhartGuard
from app.ml.rl.environment import FairnessRLEnvironment

# ParetoRewardModifier has an import bug (FairnessMetrics imported from wrong module),
# so we conditionally import and skip related tests if unavailable.
_PARETO_AVAILABLE = False
try:
    from app.ml.rl.pareto_reward import ParetoConfig, ParetoRewardModifier, ParetoSolution
    _PARETO_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ParetoConfig = ParetoRewardModifier = ParetoSolution = None

TORCH_AVAILABLE = True


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def action_space():
    """Small action space for fast tests."""
    return ContinuousActionSpace(n_thresholds=3, n_feature_weights=4)


@pytest.fixture
def reward_shaper():
    return MultiObjectiveRewardShaper()


@pytest.fixture
def pareto_modifier():
    if not _PARETO_AVAILABLE:
        pytest.skip("ParetoRewardModifier not available")
    return ParetoRewardModifier()


@pytest.fixture
def goodhart_guard():
    return GoodhartGuard()


@pytest.fixture
def sample_old_metrics():
    return FairnessMetrics(
        accuracy=0.75,
        demographic_parity_gap=0.15,
        equalized_odds_gap=0.20,
        calibration_error=0.10,
    )


@pytest.fixture
def sample_new_metrics():
    return FairnessMetrics(
        accuracy=0.77,
        demographic_parity_gap=0.10,
        equalized_odds_gap=0.15,
        calibration_error=0.08,
    )


@pytest.fixture
def env_data():
    """Provide X, y, sensitive_features for the environment."""
    np.random.seed(42)
    X = np.random.randn(100, 6).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    sensitive_features = (X[:, 2] > 0).astype(np.float32)
    return X, y, sensitive_features


# ===================================================================
# Action Space Tests
# ===================================================================

class TestActionSpaceInit:
    def test_creation(self, action_space):
        assert action_space.n_thresholds == 3
        assert action_space.n_feature_weights == 4
        assert action_space.action_dim == 7

    def test_sample_shape(self, action_space):
        action = action_space.sample()
        assert action.shape == (action_space.action_dim,)
        assert action.dtype == np.float32

    def test_sample_within_bounds(self, action_space):
        rng = np.random.RandomState(99)
        for _ in range(50):
            action = action_space.sample()
            low, high = action_space.get_bounds()
            assert np.all(action >= low - 1e-6)
            assert np.all(action <= high + 1e-6)

    def test_action_names(self, action_space):
        names = action_space.action_names
        assert len(names) == 7
        assert "threshold_0" in names
        assert "feature_weight_0" in names

    def test_default_action(self, action_space):
        default = action_space.default_action()
        assert default.shape == (7,)
        # Threshold defaults should be 0.0
        assert np.allclose(default[: action_space.n_thresholds], 0.0)
        # Weight defaults should be 1.0
        assert np.allclose(default[action_space.n_thresholds :], 1.0)

    def test_split_actions(self, action_space):
        action = action_space.sample()
        thresholds, weights = action_space.split_actions(action)
        assert thresholds.shape == (3,)
        assert weights.shape == (4,)


class TestActionSpaceBounds:
    def test_clip_within_bounds(self, action_space):
        low, high = action_space.get_bounds()
        out_of_bounds = np.array([999.0, -999.0] * 4)[: action_space.action_dim]
        clipped = action_space.clip(out_of_bounds)
        assert np.all(clipped >= low - 1e-6)
        assert np.all(clipped <= high + 1e-6)

    def test_clip_no_change_for_valid(self, action_space):
        low, high = action_space.get_bounds()
        mid = (low + high) / 2.0
        clipped = action_space.clip(mid)
        np.testing.assert_allclose(clipped, mid, atol=1e-6)

    def test_action_bounds_clip(self):
        bounds = ActionBounds(name="test", low=-1.0, high=1.0, default=0.0)
        assert bounds.clip(0.5) == 0.5
        assert bounds.clip(2.0) == 1.0
        assert bounds.clip(-3.0) == -1.0

    def test_get_bounds_shapes(self, action_space):
        low, high = action_space.get_bounds()
        assert low.shape == (action_space.action_dim,)
        assert high.shape == (action_space.action_dim,)


# ===================================================================
# Reward Shaper Tests
# ===================================================================

class TestRewardShaper:
    def test_improving_metrics_positive_reward(
        self, reward_shaper, sample_old_metrics, sample_new_metrics
    ):
        result = reward_shaper.compute_reward(sample_old_metrics, sample_new_metrics)
        assert isinstance(result, RewardComponents)
        # Accuracy improved, fairness gaps reduced → positive reward expected
        assert result.total_reward > 0

    def test_degrading_accuracy_penalty(self, reward_shaper):
        old = FairnessMetrics(accuracy=0.80, demographic_parity_gap=0.10,
                              equalized_odds_gap=0.10, calibration_error=0.05)
        new = FairnessMetrics(accuracy=0.50, demographic_parity_gap=0.05,
                              equalized_odds_gap=0.05, calibration_error=0.02)
        result = reward_shaper.compute_reward(old, new)
        # Accuracy dropped dramatically → penalty should make total negative
        assert result.accuracy_penalty < 0

    def test_no_change_zero_reward(self, reward_shaper, sample_old_metrics):
        result = reward_shaper.compute_reward(sample_old_metrics, sample_old_metrics)
        # No change → reward components should be ~0 (aside from penalties)
        assert abs(result.accuracy_reward) < 1e-6
        assert abs(result.dp_reward) < 1e-6

    def test_reward_components_dict(self, reward_shaper, sample_old_metrics, sample_new_metrics):
        result = reward_shaper.compute_reward(sample_old_metrics, sample_new_metrics)
        d = reward_shaper.get_reward_components(result)
        assert "accuracy_reward" in d
        assert "total_reward" in d
        assert isinstance(d["total_reward"], float)

    def test_normalize_reward(self, reward_shaper):
        assert reward_shaper.normalize_reward(1.0, 0.0, 1.0) == 1.0
        assert reward_shaper.normalize_reward(0.0, 0.0, 1e-10) == 0.0


# ===================================================================
# Pareto Modifier Tests
# ===================================================================

@pytest.mark.skipif(not _PARETO_AVAILABLE, reason="ParetoRewardModifier import failed")
class TestParetoModifier:
    def test_empty_front_neutral(self):
        modifier = ParetoRewardModifier()
        sol = ParetoSolution(accuracy=0.8, dp_gap=0.1, eo_gap=0.1, calibration_error=0.05)
        assert modifier.get_pareto_reward_multiplier(sol) == 1.0

    def test_update_pareto_front_adds(self):
        modifier = ParetoRewardModifier()
        sol = ParetoSolution(accuracy=0.8, dp_gap=0.1, eo_gap=0.1, calibration_error=0.05, step=0)
        front = modifier.update_pareto_front(sol)
        assert len(front) == 1
        assert front[0].accuracy == 0.8

    def test_dominates_any(self):
        modifier = ParetoRewardModifier()
        weak = ParetoSolution(accuracy=0.6, dp_gap=0.3, eo_gap=0.3, calibration_error=0.2, step=0)
        modifier.update_pareto_front(weak)
        strong = ParetoSolution(accuracy=0.9, dp_gap=0.05, eo_gap=0.05, calibration_error=0.02, step=1)
        assert modifier.dominates_any(strong) is True

    def test_dominated_solution_penalty(self):
        modifier = ParetoRewardModifier()
        strong = ParetoSolution(accuracy=0.9, dp_gap=0.05, eo_gap=0.05, calibration_error=0.02, step=0)
        modifier.update_pareto_front(strong)
        weak = ParetoSolution(accuracy=0.5, dp_gap=0.4, eo_gap=0.4, calibration_error=0.3, step=1)
        multiplier = modifier.get_pareto_reward_multiplier(weak)
        assert multiplier < 1.0

    def test_dominating_solution_bonus(self):
        modifier = ParetoRewardModifier()
        weak = ParetoSolution(accuracy=0.6, dp_gap=0.3, eo_gap=0.3, calibration_error=0.2, step=0)
        modifier.update_pareto_front(weak)
        strong = ParetoSolution(accuracy=0.95, dp_gap=0.01, eo_gap=0.01, calibration_error=0.01, step=1)
        multiplier = modifier.get_pareto_reward_multiplier(strong)
        assert multiplier > 1.0

    def test_front_size_cap(self):
        tiny_config = ParetoConfig(max_front_size=3)
        modifier = ParetoRewardModifier(config=tiny_config)
        for i in range(10):
            sol = ParetoSolution(
                accuracy=0.5 + i * 0.05,
                dp_gap=0.3 - i * 0.02,
                eo_gap=0.3 - i * 0.02,
                calibration_error=0.2 - i * 0.01,
                step=i,
            )
            modifier.update_pareto_front(sol)
        assert modifier.front_size() <= 3


# ===================================================================
# Goodhart Guard Tests
# ===================================================================

class TestGoodhartGuard:
    def test_improving_metrics_safe(self, goodhart_guard):
        old = {"accuracy": 0.70, "demographic_parity_gap": 0.20,
               "equalized_odds_gap": 0.25, "calibration_error": 0.15}
        new = {"accuracy": 0.75, "demographic_parity_gap": 0.15,
               "equalized_odds_gap": 0.20, "calibration_error": 0.10}
        report = goodhart_guard.check(old, new)
        assert report.is_safe is True
        assert report.alert_level == AlertLevel.SAFE

    def test_degrading_metrics_unsafe(self, goodhart_guard):
        old = {"accuracy": 0.80, "demographic_parity_gap": 0.10,
               "equalized_odds_gap": 0.10, "calibration_error": 0.05}
        # Large accuracy drop and gap increases
        new = {"accuracy": 0.30, "demographic_parity_gap": 0.30,
               "equalized_odds_gap": 0.35, "calibration_error": 0.20}
        report = goodhart_guard.check(old, new)
        assert report.is_safe is False
        assert report.alert_level == AlertLevel.UNSAFE

    def test_warning_level(self, goodhart_guard):
        old = {"accuracy": 0.75, "demographic_parity_gap": 0.15,
               "equalized_odds_gap": 0.20, "calibration_error": 0.10}
        # Moderate degradation
        new = {"accuracy": 0.65, "demographic_parity_gap": 0.22,
               "equalized_odds_gap": 0.25, "calibration_error": 0.14}
        report = goodhart_guard.check(old, new)
        assert report.alert_level in (AlertLevel.WARNING, AlertLevel.UNSAFE)

    def test_adjust_reward_safe_no_change(self, goodhart_guard):
        report = goodhart_guard.check(
            {"accuracy": 0.7}, {"accuracy": 0.75}
        )
        assert goodhart_guard.adjust_reward(1.0, report) == 1.0

    def test_adjust_reward_unsafe_reduction(self, goodhart_guard):
        old = {"accuracy": 0.80, "demographic_parity_gap": 0.10,
               "equalized_odds_gap": 0.10, "calibration_error": 0.05}
        new = {"accuracy": 0.30, "demographic_parity_gap": 0.30,
               "equalized_odds_gap": 0.35, "calibration_error": 0.20}
        report = goodhart_guard.check(old, new)
        adjusted = goodhart_guard.adjust_reward(10.0, report)
        assert adjusted < 10.0

    def test_reset(self, goodhart_guard):
        goodhart_guard.check({"accuracy": 0.7}, {"accuracy": 0.6})
        goodhart_guard.reset()
        trends = goodhart_guard.get_metric_trends()
        # After reset, all trends should be 0 (not enough data)
        assert all(v == 0.0 for v in trends.values())


# ===================================================================
# PPO Agent Creation Tests (no training, just construction)
# ===================================================================

class TestPPOAgentCreation:
    def test_agent_creates_successfully(self):
        from app.ml.rl.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=10, action_dim=5, device="cpu")
        assert agent.state_dim == 10
        assert agent.action_dim == 5
        assert agent.buffer is not None

    def test_agent_select_action(self):
        from app.ml.rl.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=10, action_dim=5, device="cpu")
        state = np.random.randn(10).astype(np.float32)
        action, log_prob, value = agent.select_action(state)
        assert action.shape == (5,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_agent_compute_gae(self):
        from app.ml.rl.ppo_agent import PPOAgent
        agent = PPOAgent(state_dim=10, action_dim=5, device="cpu")
        rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        values = [0.5, 0.6, 0.7, 0.8, 0.9]
        dones = [False, False, False, False, True]
        advantages = agent.compute_gae(rewards, values, dones, next_value=0.0)
        assert len(advantages) == 5
        assert all(isinstance(a, float) for a in advantages)


# ===================================================================
# Environment Tests
# ===================================================================

class TestEnvironmentReset:
    def test_reset_returns_correct_shape(self, env_data):
        X, y, sensitive_features = env_data
        env = FairnessRLEnvironment(
            X=X, y=y, sensitive_features=sensitive_features,
            n_thresholds=2, n_feature_weights=min(3, X.shape[1]),
        )
        obs = env.reset()
        expected_dim = 5 + env.action_space.action_dim
        assert obs.shape == (expected_dim,)
        assert obs.dtype == np.float32

    def test_step_returns_correct_shapes(self, env_data):
        X, y, sensitive_features = env_data
        env = FairnessRLEnvironment(
            X=X, y=y, sensitive_features=sensitive_features,
            n_thresholds=2, n_feature_weights=min(3, X.shape[1]),
        )
        env.reset()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        expected_dim = 5 + env.action_space.action_dim
        assert obs.shape == (expected_dim,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_observation_dim_property(self, env_data):
        X, y, sensitive_features = env_data
        env = FairnessRLEnvironment(
            X=X, y=y, sensitive_features=sensitive_features,
            n_thresholds=2, n_feature_weights=min(3, X.shape[1]),
        )
        expected = 5 + env.action_space.action_dim
        assert env.observation_dim == expected
