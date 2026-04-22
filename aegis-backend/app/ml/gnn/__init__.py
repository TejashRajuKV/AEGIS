"""GNN Package - Graph Neural Network components for causal discovery."""


def __getattr__(name):
    _lazy = {
        "NOTEARSLayer": "app.ml.gnn.dag_gnn_layers",
        "DAGGNNLayer": "app.ml.gnn.dag_gnn_layers",
        "NodeEncoder": "app.ml.gnn.node_encoder",
        "EdgeDecoder": "app.ml.gnn.edge_decoder",
        "GraphAttentionLayer": "app.ml.gnn.graph_attention",
        "MultiHeadGraphAttention": "app.ml.gnn.graph_attention",
        "CausalGNNTrainer": "app.ml.gnn.causal_gnn_trainer",
    }
    if name in _lazy:
        import importlib
        module = importlib.import_module(_lazy[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NOTEARSLayer", "DAGGNNLayer", "NodeEncoder", "EdgeDecoder",
    "GraphAttentionLayer", "MultiHeadGraphAttention", "CausalGNNTrainer",
]
