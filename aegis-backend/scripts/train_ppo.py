#!/usr/bin/env python3
"""Train PPO agent for fairness optimization."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.ml.rl.environment import FairnessEnvironment
from app.ml.rl.ppo_agent import PPOAgent
from app.ml.rl.action_space import ContinuousActionSpace
from app.ml.rl.training_loop import PPOTrainingLoop
from app.utils.logger import get_logger


def main():
    logger = get_logger("train_ppo")
    logger.info("Starting PPO training script")

    # Generate data
    np.random.seed(42)
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    protected = (np.random.RandomState(42).rand(len(y_train)) > 0.5).astype(int)

    # Create environment
    env = FairnessEnvironment(X_train, y_train, X_test, y_test, protected)

    # Create agent
    action_space = ContinuousActionSpace(num_features=X_train.shape[1])
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=action_space.action_dim,
        action_space=action_space,
    )

    # Train
    loop = PPOTrainingLoop(agent=agent, env=env, max_steps=200)
    result = loop.train(num_iterations=30, verbose=True)

    logger.info(f"Training complete: {result['improvement']}")
    return result


if __name__ == "__main__":
    main()
