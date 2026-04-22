"""
Tests for Text Bias Auditing Module
=====================================
Tests for prompt framing, cosine distance, bias scoring, and embedding extraction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.ml.text_bias.prompt_framer import PromptFramer, BiasCategory, PromptPair
from app.ml.text_bias.cosine_distance import CosineDistanceCalculator
from app.ml.text_bias.bias_scorer import TextBiasScorer, BiasLevel, BiasScore, _classify_bias, _normalize_score
from app.ml.text_bias.embedding_extractor import EmbeddingExtractor, PoolingStrategy


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def prompt_framer():
    return PromptFramer(seed=42)


@pytest.fixture
def cosine_calc():
    return CosineDistanceCalculator(eps=1e-10)


@pytest.fixture
def bias_scorer():
    return TextBiasScorer()


@pytest.fixture
def embedding_extractor():
    return EmbeddingExtractor(pooling="mean", normalize=True)


@pytest.fixture
def sample_embeddings():
    rng = np.random.RandomState(42)
    emb_a = rng.randn(384).astype(np.float64)
    emb_b = rng.randn(384).astype(np.float64)
    # Normalize them to unit vectors for deterministic cosine sim
    emb_a /= np.linalg.norm(emb_a)
    emb_b /= np.linalg.norm(emb_b)
    return emb_a, emb_b


@pytest.fixture
def identical_embeddings():
    rng = np.random.RandomState(42)
    v = rng.randn(384).astype(np.float64)
    v /= np.linalg.norm(v)
    return v, v.copy()


@pytest.fixture
def opposite_embeddings():
    rng = np.random.RandomState(42)
    v = rng.randn(384).astype(np.float64)
    v /= np.linalg.norm(v)
    return v, -v


# ===================================================================
# Prompt Framer Tests
# ===================================================================

class TestPromptFramer:

    def test_create_pair_returns_prompt_pair(self, prompt_framer):
        pair = prompt_framer.create_pair("gender")
        assert isinstance(pair, PromptPair)
        assert pair.category == "gender"
        assert len(pair.id) == 8
        assert "{demographic}" not in pair.prompt_a
        assert "{profession}" not in pair.prompt_a
        assert len(pair.prompt_a) > 0
        assert len(pair.prompt_b) > 0

    def test_create_pair_different_demographics(self, prompt_framer):
        pair = prompt_framer.create_pair("race")
        assert pair.demographic_a != pair.demographic_b

    def test_create_pair_custom_demographics(self, prompt_framer):
        pair = prompt_framer.create_pair(
            "gender",
            demographic_a="man",
            demographic_b="woman",
            profession="engineer",
        )
        assert pair.demographic_a == "man"
        assert pair.demographic_b == "woman"
        assert "engineer" in pair.prompt_a.lower()
        assert "engineer" in pair.prompt_b.lower()

    def test_create_pair_invalid_category_raises(self, prompt_framer):
        with pytest.raises(ValueError):
            prompt_framer.create_pair("nonexistent_category")

    def test_get_all_categories(self, prompt_framer):
        categories = prompt_framer.get_all_categories()
        assert isinstance(categories, list)
        assert len(categories) >= 5
        assert "gender" in categories
        assert "race" in categories

    def test_get_all_categories_is_static(self):
        cats1 = PromptFramer.get_all_categories()
        cats2 = PromptFramer.get_all_categories()
        assert cats1 == cats2

    def test_get_demographics(self, prompt_framer):
        demos = prompt_framer.get_demographics("gender")
        assert isinstance(demos, list)
        assert "man" in demos
        assert "woman" in demos

    def test_get_templates(self, prompt_framer):
        templates = prompt_framer.get_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0

    def test_stereoset_pairs(self, prompt_framer):
        pairs = prompt_framer.create_stereoset_pairs("gender")
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        assert isinstance(pairs[0], tuple)
        assert len(pairs[0]) == 2

    def test_frame_with_context(self, prompt_framer):
        result = prompt_framer.frame_with_context(
            "She was a great leader.",
            "woman",
            context_type="professional",
        )
        assert "professional" in result.lower()
        assert "woman" in result
        assert "She was a great leader." in result

    def test_generate_audit_set(self, prompt_framer):
        audit = prompt_framer.generate_audit_set(categories=["gender"], templates_per_category=2)
        assert isinstance(audit, list)
        assert len(audit) > 0
        for item in audit:
            assert "category" in item
            assert "prompt_a" in item
            assert "prompt_b" in item


# ===================================================================
# Cosine Distance Tests
# ===================================================================

class TestCosineDistance:

    def test_identical_vectors_zero_distance(self, cosine_calc, identical_embeddings):
        a, b = identical_embeddings
        dist = cosine_calc.compute(a, b)
        assert abs(dist) < 1e-6

    def test_opposite_vectors_distance_two(self, cosine_calc, opposite_embeddings):
        a, b = opposite_embeddings
        dist = cosine_calc.compute(a, b)
        assert dist > 1.5  # Should be ~2.0

    def test_similarity_range(self, cosine_calc, sample_embeddings):
        a, b = sample_embeddings
        sim = cosine_calc.compute_similarity(a, b)
        assert -1.0 <= sim <= 1.0

    def test_distance_range(self, cosine_calc, sample_embeddings):
        a, b = sample_embeddings
        dist = cosine_calc.compute(a, b)
        assert 0.0 <= dist <= 2.0

    def test_batch_compute(self, cosine_calc):
        rng = np.random.RandomState(0)
        list_a = [rng.randn(10) for _ in range(5)]
        list_b = [rng.randn(10) for _ in range(5)]
        distances = cosine_calc.compute_batch(list_a, list_b)
        assert len(distances) == 5
        for d in distances:
            assert 0.0 <= d <= 2.0 + 1e-6

    def test_batch_mismatched_lengths_raises(self, cosine_calc):
        with pytest.raises(ValueError, match="lengths differ"):
            cosine_calc.compute_batch([np.array([1, 2])], [np.array([1, 2]), np.array([3, 4])])

    def test_pairwise_matrix(self, cosine_calc):
        rng = np.random.RandomState(0)
        embeddings = [rng.randn(5) for _ in range(4)]
        mat = cosine_calc.compute_pairwise(embeddings)
        assert mat.shape == (4, 4)
        # Diagonal should be 0 (distance to self)
        for i in range(4):
            assert abs(mat[i][i]) < 1e-6

    def test_permutation_test_significant(self, cosine_calc):
        # Very different groups → should be significant
        dist_a = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        dist_b = [0.0, 0.1, 0.05, 0.02, 0.08, 0.03]
        is_sig, p_val = cosine_calc.is_significant(dist_a, dist_b, n_permutations=200, seed=42)
        assert is_sig is True
        assert p_val < 0.05


# ===================================================================
# Bias Scorer Tests
# ===================================================================

class TestBiasScorer:

    def test_score_pair_returns_bias_score(self, bias_scorer, sample_embeddings):
        a, b = sample_embeddings
        score = bias_scorer.score_pair(a, b)
        assert isinstance(score, BiasScore)
        assert 0.0 <= score.cosine_distance <= 2.0 + 1e-6
        assert 0.0 <= score.normalized_score <= 100.0
        assert isinstance(score.bias_level, BiasLevel)

    def test_identical_embeddings_none_level(self, bias_scorer, identical_embeddings):
        a, b = identical_embeddings
        score = bias_scorer.score_pair(a, b)
        assert score.bias_level == BiasLevel.NONE
        assert score.normalized_score < 10.0

    def test_precomputed_distance(self, bias_scorer):
        score = bias_scorer.score_pair(np.array([1.0]), np.array([1.0]), cosine_distance=0.5)
        assert score.cosine_distance == 0.5
        assert score.bias_level == BiasLevel.MEDIUM

    def test_classify_bias_levels(self):
        assert _classify_bias(0.0) == BiasLevel.NONE
        assert _classify_bias(0.05) == BiasLevel.NONE
        assert _classify_bias(0.15) == BiasLevel.LOW
        assert _classify_bias(0.4) == BiasLevel.MEDIUM
        assert _classify_bias(0.7) == BiasLevel.HIGH
        assert _classify_bias(1.1) == BiasLevel.SEVERE

    def test_normalize_score(self):
        assert _normalize_score(0.0) < 10.0
        assert _normalize_score(0.5) > 40.0
        assert _normalize_score(1.0) > 85.0
        assert _normalize_score(2.0) >= 95.0

    def test_score_dataset(self, bias_scorer):
        results = [
            {"cosine_distance": 0.01, "category": "gender"},
            {"cosine_distance": 0.05, "category": "gender"},
            {"cosine_distance": 0.50, "category": "race"},
            {"cosine_distance": 1.20, "category": "race"},
        ]
        summary = bias_scorer.score_dataset(results)
        assert summary.total_pairs == 4
        assert summary.mean_distance > 0.0
        assert summary.bias_index > 0.0
        assert len(summary.recommendations) > 0
        assert "gender" in summary.per_category
        assert "race" in summary.per_category

    def test_score_dataset_empty(self, bias_scorer):
        summary = bias_scorer.score_dataset([])
        assert summary.total_pairs == 0
        assert "No results" in summary.recommendations[0]

    def test_compute_bias_index(self):
        assert TextBiasScorer.compute_bias_index([]) == 0.0
        assert TextBiasScorer.compute_bias_index([50.0]) == 50.0


# ===================================================================
# Embedding Extractor Tests
# ===================================================================

class TestEmbeddingExtractor:

    def test_mean_pooling(self, embedding_extractor):
        token_embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        pooled = embedding_extractor.pool_embeddings(token_embeddings, strategy="mean")
        expected = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        np.testing.assert_allclose(pooled, expected, atol=1e-6)

    def test_max_pooling(self, embedding_extractor):
        token_embeddings = np.array([
            [1.0, 2.0, 0.5],
            [0.5, 3.0, 1.0],
            [0.2, 0.5, 4.0],
        ], dtype=np.float64)
        pooled = embedding_extractor.pool_embeddings(token_embeddings, strategy="max")
        assert pooled.shape == (3,)
        # Max across rows: [1.0, 3.0, 4.0], then L2-normalized
        expected_raw = np.array([1.0, 3.0, 4.0])
        expected = expected_raw / np.linalg.norm(expected_raw)
        np.testing.assert_allclose(pooled, expected, atol=1e-6)

    def test_cls_pooling(self, embedding_extractor):
        token_embeddings = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ], dtype=np.float64)
        pooled = embedding_extractor.pool_embeddings(token_embeddings, strategy="cls")
        # CLS takes first row [1, 2, 3], then L2-normalizes
        expected_raw = np.array([1.0, 2.0, 3.0])
        expected = expected_raw / np.linalg.norm(expected_raw)
        np.testing.assert_allclose(pooled, expected, atol=1e-6)

    def test_1d_vector_passes_through(self, embedding_extractor):
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        pooled = embedding_extractor.pool_embeddings(vec)
        assert pooled.shape == (4,)
        # Should be normalized
        norm = np.linalg.norm(pooled)
        assert abs(norm - 1.0) < 1e-6

    def test_extract_embeddings_single(self, embedding_extractor):
        emb_a, emb_b = embedding_extractor.extract_embeddings(
            "The doctor is a man.",
            "The doctor is a woman.",
        )
        assert emb_a.shape == (384,)
        assert emb_b.shape == (384,)

    def test_extract_embeddings_batch(self, embedding_extractor):
        texts_a = ["Hello world", "Goodbye world"]
        texts_b = ["Hi there", "Bye there"]
        emb_a, emb_b = embedding_extractor.extract_embeddings(texts_a, texts_b)
        assert emb_a.shape == (2, 384)
        assert emb_b.shape == (2, 384)

    def test_infer_dim(self, embedding_extractor):
        assert EmbeddingExtractor.infer_dim(np.zeros(10)) == 10
        assert EmbeddingExtractor.infer_dim(np.zeros((5, 20))) == 20

    def test_no_normalize(self):
        extractor = EmbeddingExtractor(normalize=False)
        vec = np.array([3.0, 4.0], dtype=np.float64)
        pooled = extractor.pool_embeddings(vec)
        assert abs(np.linalg.norm(pooled) - 5.0) < 1e-6  # Not normalized
