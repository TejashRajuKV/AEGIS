"""RL Package - PPO-based fairness autopilot."""


def __getattr__(name):
    """Lazy imports to avoid requiring torch at import time."""
    _lazy = {
        "PPOAgent": "app.ml.rl.ppo_agent",
        "ActorCriticNetwork": "app.ml.rl.ppo_network",
        "RewardShaper": "app.ml.rl.reward_shaper",
        "ParetoRewardChecker": "app.ml.rl.pareto_reward",
        "GoodhartGuard": "app.ml.rl.goodhart_guard",
        "ContinuousActionSpace": "app.ml.rl.action_space",
        "FairnessEnvironment": "app.ml.rl.environment",
        "PPOTrainingLoop": "app.ml.rl.training_loop",
    }
    if name in _lazy:
        import importlib
        module = importlib.import_module(_lazy[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PPOAgent", "ActorCriticNetwork", "RewardShaper", "ParetoRewardChecker",
    "GoodhartGuard", "ContinuousActionSpace", "FairnessEnvironment", "PPOTrainingLoop",
]
