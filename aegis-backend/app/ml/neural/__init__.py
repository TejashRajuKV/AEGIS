"""Neural Package - Conditional VAE for counterfactual generation."""


def __getattr__(name):
    _lazy = {
        "ConditionalVAE": "app.ml.neural.conditional_vae",
        "VAEEncoder": "app.ml.neural.vae_encoder",
        "VAEDecoder": "app.ml.neural.vae_decoder",
        "CounterfactualGenerator": "app.ml.neural.counterfactual_generator",
        "LatentInterpolator": "app.ml.neural.latent_interpolator",
        "VAETrainer": "app.ml.neural.vae_trainer",
    }
    if name in _lazy:
        import importlib
        module = importlib.import_module(_lazy[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConditionalVAE", "VAEEncoder", "VAEDecoder",
    "CounterfactualGenerator", "LatentInterpolator", "VAETrainer",
]
