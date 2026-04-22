"""Causal discovery endpoint."""
from fastapi import APIRouter, HTTPException
from app.models.schemas import CausalDiscoveryRequest, CausalDiscoveryResponse
from app.ml.causal.pc_algorithm import PCAlgorithm
# DAGGNNDiscovery is imported lazily inside the endpoint (requires PyTorch)
from app.data.dataset_loader import get_dataset_loader
from app.utils.logger import get_logger
import numpy as np
import pandas as pd

logger = get_logger("causal_route")
router = APIRouter()


@router.post("/discover", response_model=CausalDiscoveryResponse)
async def discover_causal(req: CausalDiscoveryRequest):
    try:
        loader = get_dataset_loader()
        df = loader.load_dataset(req.dataset_name)
        info = loader.get_dataset_info(req.dataset_name)
        target = info["target_column"]
        protected = info["sensitive_attributes"][0] if info.get("sensitive_attributes") else None

        # Select numeric columns for causal discovery
        num_cols = [c for c in df.columns if df[c].dtype in ["int64", "float64", "int32", "float32"]]
        if len(num_cols) < 3:
            num_cols = list(df.columns)[:10]

        data = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        feature_names = num_cols

        edges = []
        adj_matrix = None
        proxy_chains = []
        num_nodes = len(num_cols)
        num_edges = 0

        if req.method == "dag_gnn":
            try:
                from app.ml.causal.dag_gnn import DAGGNNDiscovery
            except ImportError as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"DAG-GNN requires PyTorch which is not installed: {e}",
                )
            dag_gnn = DAGGNNDiscovery(
                input_dim=len(num_cols),
                max_epochs=req.max_epochs,
                hidden_dim=32,
            )
            result = dag_gnn.discover(data, feature_names=feature_names)
            result_dict = result.to_dict()

            # Build edges from CausalGraph result
            for src, tgt, weight in result.edge_list:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "weight": weight,
                    "is_significant": abs(weight) >= req.threshold,
                })

            adj_matrix = result_dict.get("adjacency_matrix")
            num_nodes = result.num_nodes
            num_edges = result.num_edges

            # Find proxy chains
            if result.graph is not None:
                from app.ml.causal.proxy_chain_detector import ProxyChainDetector
                if protected and target in feature_names:
                    detector = ProxyChainDetector()
                    proxy_result = detector.detect(
                        graph=result.graph,
                        sensitive_nodes=[protected],
                        outcome_node=target,
                    )
                    proxy_dict = proxy_result.to_dict()
                    proxy_chains = [
                        {
                            "chain": pc.get("chain", []),
                            "total_indirect_effect": pc.get("total_indirect_effect", 0.0),
                            "strength": pc.get("strength", "weak"),
                        }
                        for pc in proxy_dict.get("proxy_chains", [])
                    ]
        else:
            # PC algorithm
            pc = PCAlgorithm(alpha=req.threshold)
            result = pc.discover(df[num_cols])

            for e in result.get("edges", []):
                edges.append({
                    "source": e.get("source", e.get("u", "")),
                    "target": e.get("target", e.get("v", "")),
                    "weight": e.get("weight", 1.0),
                    "is_significant": True,
                })
            num_edges = len(edges)
            if result.get("adjacency_matrix") is not None:
                adj_matrix = result["adjacency_matrix"]

        return CausalDiscoveryResponse(
            dataset_name=req.dataset_name,
            method=req.method,
            edges=edges,
            proxy_chains=proxy_chains,
            num_nodes=num_nodes,
            num_edges=num_edges,
            adjacency_matrix=adj_matrix,
        )
    except Exception as e:
        logger.error(f"Causal discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
