"""
Tests for CVAE Counterfactual Generation
==========================================
Tests for VAE encoder, decoder, CVAE loss function, and sampling.
Skipped entirely if PyTorch is not available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

torch = pytest.importorskip("torch", reason="PyTorch is required for counterfactual tests")

from app.ml.neural.vae_encoder import VAEEncoder
from app.ml.neural.vae_decoder import VAEDecoder
from app.ml.neural.conditional_vae import ConditionalVAE


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def encoder():
    return VAEEncoder(
        input_dim=10,
        condition_dim=2,
        latent_dim=8,
        hidden_dims=[32, 16],
        dropout=0.0,
    )


@pytest.fixture
def decoder():
    return VAEDecoder(
        input_dim=10,
        condition_dim=2,
        latent_dim=8,
        hidden_dims=[16, 32],
        dropout=0.0,
        output_activation="sigmoid",
    )


@pytest.fixture
def cvae():
    return ConditionalVAE(
        input_dim=10,
        condition_dim=2,
        latent_dim=8,
        encoder_hidden=[32, 16],
        decoder_hidden=[16, 32],
        dropout=0.0,
        output_activation="sigmoid",
    )


@pytest.fixture
def sample_batch():
    """Generate a small batch of input data and conditions."""
    torch.manual_seed(42)
    x = torch.randn(16, 10)
    condition = torch.randint(0, 2, (16, 2)).float()
    return x, condition


# ===================================================================
# Encoder Tests
# ===================================================================

class TestVAEEncoder:

    def test_forward_shape(self, encoder, sample_batch):
        x, condition = sample_batch
        # Reduce batch to match encoder input_dim
        x_small = x[:, :encoder.input_dim]
        mu, log_var = encoder(x_small, condition)
        assert mu.shape == (16, encoder.latent_dim)
        assert log_var.shape == (16, encoder.latent_dim)

    def test_forward_without_condition(self, encoder, sample_batch):
        # Create an encoder with condition_dim=0 for unconditional use
        uncond_encoder = VAEEncoder(
            input_dim=10, condition_dim=0, latent_dim=8,
            hidden_dims=[32, 16], dropout=0.0,
        )
        x = sample_batch[0][:, :10]
        mu, log_var = uncond_encoder(x)
        assert mu.shape == (16, uncond_encoder.latent_dim)
        assert log_var.shape == (16, uncond_encoder.latent_dim)

    def test_reparameterize_shape(self, encoder, sample_batch):
        x, condition = sample_batch
        x_small = x[:, :encoder.input_dim]
        mu, log_var = encoder(x_small, condition)
        z = encoder.reparameterize(mu, log_var)
        assert z.shape == (16, encoder.latent_dim)

    def test_reparameterize_different_samples(self, encoder, sample_batch):
        x, condition = sample_batch
        x_small = x[:, :encoder.input_dim]
        mu, log_var = encoder(x_small, condition)
        z1 = encoder.reparameterize(mu, log_var)
        z2 = encoder.reparameterize(mu, log_var)
        # Two samples from same distribution should be different
        assert not torch.allclose(z1, z2)

    def test_reparameterize_zero_variance(self, encoder):
        mu = torch.zeros(4, encoder.latent_dim)
        log_var = torch.full((4, encoder.latent_dim), -100.0)  # ≈ 0 variance
        z = encoder.reparameterize(mu, log_var)
        torch.testing.assert_close(z, mu, atol=1e-4, rtol=1e-4)


# ===================================================================
# Decoder Tests
# ===================================================================

class TestVAEDecoder:

    def test_forward_shape(self, decoder, sample_batch):
        z = torch.randn(16, decoder.latent_dim)
        condition = sample_batch[1]
        output = decoder(z, condition)
        assert output.shape == (16, decoder.input_dim)

    def test_forward_without_condition(self, decoder, sample_batch):
        # Without condition, decoder expects latent_dim + condition_dim input
        z = torch.randn(8, decoder.latent_dim + decoder.condition_dim)
        output = decoder(z)
        assert output.shape == (8, decoder.input_dim)

    def test_sigmoid_output_range(self, decoder, sample_batch):
        z = torch.randn(16, decoder.latent_dim)
        output = decoder(z, sample_batch[1])
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_tanh_output_range(self):
        dec = VAEDecoder(
            input_dim=10, condition_dim=0, latent_dim=8,
            hidden_dims=[16, 32], output_activation="tanh",
        )
        z = torch.randn(8, 8)
        output = dec(z)
        assert output.min() >= -1.0
        assert output.max() <= 1.0
        assert output.shape == (8, 10)

    def test_none_output_activation(self):
        dec = VAEDecoder(
            input_dim=10, condition_dim=0, latent_dim=8,
            hidden_dims=[16, 32], output_activation="none",
        )
        z = torch.randn(8, 8)
        output = dec(z)
        assert output.shape == (8, 10)


# ===================================================================
# CVAE Tests
# ===================================================================

class TestConditionalVAE:

    def test_forward_shape(self, cvae, sample_batch):
        x = sample_batch[0][:, :cvae.input_dim]
        condition = sample_batch[1]
        recon, mu, log_var = cvae(x, condition)
        assert recon.shape == (16, cvae.input_dim)
        assert mu.shape == (16, cvae.latent_dim)
        assert log_var.shape == (16, cvae.latent_dim)

    def test_loss_function_keys(self, cvae, sample_batch):
        x = sample_batch[0][:, :cvae.input_dim]
        condition = sample_batch[1]
        recon, mu, log_var = cvae(x, condition)
        losses = cvae.loss_function(x, recon, mu, log_var)
        assert isinstance(losses, dict)
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses
        assert "total_loss" in losses

    def test_loss_values_are_finite(self, cvae, sample_batch):
        x = sample_batch[0][:, :cvae.input_dim]
        condition = sample_batch[1]
        recon, mu, log_var = cvae(x, condition)
        losses = cvae.loss_function(x, recon, mu, log_var)
        for key, val in losses.items():
            assert torch.isfinite(val).all(), f"{key} is not finite: {val}"

    def test_loss_total_is_sum(self, cvae, sample_batch):
        x = sample_batch[0][:, :cvae.input_dim]
        condition = sample_batch[1]
        recon, mu, log_var = cvae(x, condition)
        losses = cvae.loss_function(x, recon, mu, log_var)
        expected = losses["reconstruction_loss"] + losses["kl_loss"]
        torch.testing.assert_close(losses["total_loss"], expected, atol=1e-5, rtol=1e-5)

    def test_kl_weight_scales_loss(self, cvae, sample_batch):
        x = sample_batch[0][:, :cvae.input_dim]
        condition = sample_batch[1]
        recon, mu, log_var = cvae(x, condition)
        losses_default = cvae.loss_function(x, recon, mu, log_var, kl_weight=1.0)
        losses_scaled = cvae.loss_function(x, recon, mu, log_var, kl_weight=5.0)
        # KL loss value is the same regardless of weight; only total_loss changes
        torch.testing.assert_close(losses_scaled["kl_loss"], losses_default["kl_loss"])
        # Total loss should be higher with higher KL weight
        assert losses_scaled["total_loss"] > losses_default["total_loss"]

    def test_sample_shape(self, cvae, sample_batch):
        condition = sample_batch[1][:1]  # single condition
        samples = cvae.sample(num_samples=32, condition=condition, device="cpu")
        assert samples.shape == (32, cvae.input_dim)

    def test_sample_output_range_sigmoid(self, cvae, sample_batch):
        condition = sample_batch[1][:1]
        samples = cvae.sample(num_samples=50, condition=condition, device="cpu")
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

    def test_sample_expands_single_condition(self, cvae, sample_batch):
        condition = sample_batch[1][:1]  # shape (1, 2)
        samples = cvae.sample(num_samples=10, condition=condition, device="cpu")
        assert samples.shape[0] == 10

    def test_generate_counterfactual_shape(self, cvae, sample_batch):
        x = sample_batch[0][:4, :cvae.input_dim]
        orig_cond = sample_batch[1][:4]
        # Different target condition
        target_cond = 1.0 - orig_cond
        cf = cvae.generate_counterfactual(x, orig_cond, target_cond)
        assert cf.shape == (4, cvae.input_dim)

    def test_counterfactual_differs_from_input(self, cvae, sample_batch):
        x = sample_batch[0][:4, :cvae.input_dim]
        orig_cond = sample_batch[1][:4]
        target_cond = 1.0 - orig_cond
        cf = cvae.generate_counterfactual(x, orig_cond, target_cond)
        # Counterfactual should generally differ from input
        # (unless the model hasn't been trained, some may match)
        num_different = (cf != x).float().sum(dim=1)
        assert num_different.sum() > 0  # at least some differences

    def test_encode_decode_consistency(self, cvae, sample_batch):
        x = sample_batch[0][:, :cvae.input_dim]
        condition = sample_batch[1]
        mu, log_var = cvae.encode(x, condition)
        z = cvae.reparameterize(mu, log_var)
        recon = cvae.decode(z, condition)
        assert recon.shape == x.shape
