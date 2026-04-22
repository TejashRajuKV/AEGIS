"""Drift Detection Package."""


def __getattr__(name):
    _lazy = {
        "CUSUMDetector": "app.ml.drift.cusum_detector",
        "WassersteinDetector": "app.ml.drift.wasserstein_detector",
        "DistributionComparator": "app.ml.drift.distribution_comparator",
        "TemporalWindow": "app.ml.drift.temporal_window",
        "DriftAlertManager": "app.ml.drift.drift_alert",
        "DriftEnsembleDetector": "app.ml.drift.drift_ensemble",
    }
    if name in _lazy:
        import importlib
        module = importlib.import_module(_lazy[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CUSUMDetector", "WassersteinDetector", "DistributionComparator",
    "TemporalWindow", "DriftAlertManager", "DriftEnsembleDetector",
]
