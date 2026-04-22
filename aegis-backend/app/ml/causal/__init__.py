"""Causal Discovery Package."""


def __getattr__(name):
    _lazy = {
        "DAGGNN": "app.ml.causal.dag_gnn",
        "DAGGNNModel": "app.ml.causal.dag_gnn_model",
        "ProxyChainDetector": "app.ml.causal.proxy_chain_detector",
    }
    if name in _lazy:
        import importlib
        module = importlib.import_module(_lazy[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DAGGNN", "DAGGNNModel", "ProxyChainDetector"]
