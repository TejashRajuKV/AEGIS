"""
AEGIS Text Bias Detection Module

Provides tools for auditing LLM text outputs for demographic bias
using embedding-based cosine distance analysis.
"""

from app.ml.text_bias.llm_wrapper import LLMWrapper
from app.ml.text_bias.embedding_extractor import EmbeddingExtractor
from app.ml.text_bias.cosine_distance import CosineDistanceCalculator
from app.ml.text_bias.prompt_framer import PromptFramer
from app.ml.text_bias.bias_scorer import TextBiasScorer, BiasScore, BiasLevel, DatasetBiasSummary
from app.ml.text_bias.text_auditor import (
    TextAuditor,
    SingleAuditResult,
    CategoryAuditResult,
    FullAuditReport,
)

__all__ = [
    "LLMWrapper",
    "EmbeddingExtractor",
    "CosineDistanceCalculator",
    "PromptFramer",
    "TextBiasScorer",
    "BiasScore",
    "BiasLevel",
    "DatasetBiasSummary",
    "TextAuditor",
    "SingleAuditResult",
    "CategoryAuditResult",
    "FullAuditReport",
]
