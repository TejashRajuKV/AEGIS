#!/usr/bin/env python3
"""Train CVAE for counterfactual generation."""

import numpy as np
from sklearn.datasets import make_classification

from app.ml.neural.conditional_vae import ConditionalVAE
from app.ml.neural.vae_trainer import VAETrainer
from app.utils.logger import get_logger


def main():
    logger = get_logger("train_cvae")
    logger.info("Starting CVAE training script")

    # Generate data
    np.random.seed(42)
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=5, random_state=42
    )

    # Normalized conditions (protected attribute as condition)
    conditions = np.random.randint(0, 2, size=(len(X), 1)).astype(float)

    # Create and train CVAE
    cvae = ConditionalVAE(
        input_dim=X.shape[1],
        condition_dim=1,
        latent_dim=16,
        hidden_dims=[128, 64],
        beta=0.1,
    )

    trainer = VAETrainer(cvae=cvae, learning_rate=1e-3)
    result = trainer.train(
        X, conditions, epochs=30, batch_size=32, verbose=True
    )

    logger.info(f"Training complete: loss={result['best_val_loss']:.4f}")
    return result


if __name__ == "__main__":
    main()
