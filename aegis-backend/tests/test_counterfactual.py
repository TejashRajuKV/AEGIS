"""Test counterfactual generation."""

import importlib
import numpy as np
import pytest

torch_available = bool(importlib.util.find_spec("torch"))
skip_no_torch = pytest.mark.skipif(not torch_available, reason="torch not installed")


class TestCVAE:
    @skip_no_torch
    def test_cvae_forward(self):
        import torch
        from app.ml.neural.conditional_vae import ConditionalVAE

        cvae = ConditionalVAE(input_dim=10, condition_dim=1, latent_dim=4)
        x = torch.randn(8, 10)
        c = torch.randn(8, 1)

        recon, mu, log_var = cvae(x, c)
        assert recon.shape == (8, 10)
        assert mu.shape == (8, 4)
        assert log_var.shape == (8, 4)

    @skip_no_torch
    def test_cvae_loss(self):
        import torch
        from app.ml.neural.conditional_vae import ConditionalVAE

        cvae = ConditionalVAE(input_dim=10, condition_dim=1, latent_dim=4)
        x = torch.randn(8, 10)
        c = torch.randn(8, 1)

        recon, mu, log_var = cvae(x, c)
        losses = cvae.loss_function(x, recon, mu, log_var)

        assert "total_loss" in losses
        assert "reconstruction_loss" in losses
        assert "kl_loss" in losses

    @skip_no_torch
    def test_cvae_generate(self):
        import torch
        from app.ml.neural.conditional_vae import ConditionalVAE

        cvae = ConditionalVAE(input_dim=10, condition_dim=1, latent_dim=4)
        z = torch.randn(4, 4)
        c = torch.randn(4, 1)

        samples = cvae.generate(z, c)
        assert samples.shape == (4, 10)


class TestVAEEncoder:
    @skip_no_torch
    def test_encoder_forward(self):
        import torch
        from app.ml.neural.vae_encoder import VAEEncoder

        encoder = VAEEncoder(input_dim=10, condition_dim=1, latent_dim=4)
        x = torch.randn(8, 10)
        c = torch.randn(8, 1)

        mu, log_var = encoder(x, c)
        assert mu.shape == (8, 4)
        assert log_var.shape == (8, 4)


class TestVAEDecoder:
    @skip_no_torch
    def test_decoder_forward(self):
        import torch
        from app.ml.neural.vae_decoder import VAEDecoder

        decoder = VAEDecoder(output_dim=10, condition_dim=1, latent_dim=4)
        z = torch.randn(8, 4)
        c = torch.randn(8, 1)

        output = decoder(z, c)
        assert output.shape == (8, 10)


class TestCounterfactualGenerator:
    @skip_no_torch
    def test_counterfactual_init(self):
        import torch
        from app.ml.neural.conditional_vae import ConditionalVAE
        from app.ml.neural.counterfactual_generator import CounterfactualGenerator

        cvae = ConditionalVAE(input_dim=10, condition_dim=1, latent_dim=4)
        gen = CounterfactualGenerator(
            cvae=cvae,
            feature_names=[f"f{i}" for i in range(10)],
        )
        assert gen is not None


class TestLatentInterpolator:
    @skip_no_torch
    def test_interpolator(self):
        import torch
        from app.ml.neural.conditional_vae import ConditionalVAE
        from app.ml.neural.latent_interpolator import LatentInterpolator

        cvae = ConditionalVAE(input_dim=5, condition_dim=1, latent_dim=2)
        interp = LatentInterpolator(cvae)

        result = interp.interpolate(
            np.random.randn(5),
            np.random.randn(5),
            0.0, 1.0,
            num_steps=5,
        )
        assert len(result) == 6  # num_steps + 1
