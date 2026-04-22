"""
LLMWrapper – wraps any LLM (OpenAI, Anthropic, or local) for text generation
and embedding extraction.  Falls back to a lightweight local model when APIs
are unavailable.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports – all wrapped so the module loads without them
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    logger.debug("python-dotenv not available; .env will not be auto-loaded")

try:
    import numpy as np

    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

try:
    import openai

    HAS_OPENAI = True
except Exception:
    openai = None  # type: ignore[assignment]
    HAS_OPENAI = False

try:
    import anthropic

    HAS_ANTHROPIC = True
except Exception:
    anthropic = None  # type: ignore[assignment]
    HAS_ANTHROPIC = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore[assignment]
    HAS_ST = False


# ---------------------------------------------------------------------------
# Tiny hash-based local embedding fallback (no external model required)
# ---------------------------------------------------------------------------
def _simple_hash_embed(text: str, dim: int = 384) -> List[float]:
    """Deterministic pseudo-embedding via character hashing.
    Produces a unit vector – sufficient for demo / offline mode."""
    import hashlib

    vec = [0.0] * dim
    # Use sliding-window character bigram hashing for better distribution
    for i in range(len(text) - 1):
        bigram = text[i : i + 2].encode("utf-8")
        h = hashlib.sha256(bigram).hexdigest()
        for j, ch in enumerate(h[:8]):
            idx = (i + ord(ch)) % dim
            vec[idx] += (int(ch, 16) - 7.5) / 7.5
    # Normalise
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class LLMWrapper:
    """Unified wrapper for LLM text generation and embeddings.

    Supports three backends:
    1. **Anthropic** (Claude) – via ``ANTHROPIC_API_KEY``
    2. **OpenAI** – via ``OPENAI_API_KEY``
    3. **Local** (SentenceTransformers) – auto-selected when APIs are missing

    The wrapper always provides ``generate``, ``embed``, and batch variants,
    falling back gracefully when a backend is missing.
    """

    # Default local embedding model (small, fast)
    LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        device: str = "cpu",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # ---- Provider detection ------------------------------------------------
        if provider:
            self.provider = provider.lower()
        elif os.getenv("ANTHROPIC_API_KEY") and HAS_ANTHROPIC:
            self.provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY") and HAS_OPENAI:
            self.provider = "openai"
        else:
            self.provider = "local"

        # ---- Initialise backend -------------------------------------------------
        self._client: Any = None
        self._embed_model: Any = None
        self._embed_dim: int = 384  # default for local fallback

        if self.provider == "anthropic" and HAS_ANTHROPIC:
            key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
            self.model_name = self.model_name or "claude-sonnet-4-20250514"
            self._client = anthropic.Anthropic(api_key=key)
            logger.info("LLMWrapper – initialised Anthropic client (model=%s)", self.model_name)

        elif self.provider == "openai" and HAS_OPENAI:
            key = api_key or os.getenv("OPENAI_API_KEY", "")
            self.model_name = self.model_name or "gpt-4o-mini"
            self._client = openai.OpenAI(api_key=key)
            logger.info("LLMWrapper – initialised OpenAI client (model=%s)", self.model_name)

        else:
            self.provider = "local"
            logger.info("LLMWrapper – using local fallback provider")

        # ---- Embedding model (try sentence-transformers first) ------------------
        self._init_embed_model()

    # ------------------------------------------------------------------
    # Embedding model initialisation
    # ------------------------------------------------------------------
    def _init_embed_model(self) -> None:
        """Load the best available embedding model."""
        if self.provider == "openai" and self._client is not None:
            self._embed_dim = 1536  # text-embedding-ada-002 / text-embedding-3-small
            logger.info("LLMWrapper – will use OpenAI embeddings (dim=%d)", self._embed_dim)
            return

        if HAS_ST:
            try:
                model_id = os.getenv("AEGIS_EMBED_MODEL", self.LOCAL_EMBED_MODEL)
                self._embed_model = SentenceTransformer(model_id, device=self.device)
                self._embed_dim = self._embed_model.get_sentence_embedding_dimension()
                self.provider = "local"  # generation may be local even with API embed
                logger.info(
                    "LLMWrapper – loaded SentenceTransformer '%s' (dim=%d)",
                    model_id,
                    self._embed_dim,
                )
            except Exception as exc:
                logger.warning("Failed to load SentenceTransformer: %s – using hash fallback", exc)
                self._embed_dim = 384
        else:
            logger.info("LLMWrapper – sentence-transformers unavailable; using hash fallback (dim=384)")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a text completion for *prompt*."""
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.provider == "anthropic" and self._client is not None:
                    return self._generate_anthropic(prompt, max_tokens, temperature, system_prompt)
                if self.provider == "openai" and self._client is not None:
                    return self._generate_openai(prompt, max_tokens, temperature, system_prompt)
                return self._generate_local(prompt)
            except Exception as exc:
                logger.warning("Generation attempt %d failed: %s", attempt, exc)
                if attempt == self.max_retries:
                    logger.error("All %d generation attempts exhausted – returning fallback", self.max_retries)
                    return self._generate_local(prompt)
                time.sleep(self.retry_delay * (2 ** (attempt - 1)))
        return self._generate_local(prompt)

    def _generate_anthropic(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        resp = self._client.messages.create(**kwargs)
        # Extract text from content blocks
        text_parts = []
        for block in resp.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts).strip()

    def _generate_openai(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()

    @staticmethod
    def _generate_local(prompt: str) -> str:
        """Deterministic local fallback – returns a canned response."""
        logger.debug("Local fallback generation for prompt (len=%d)", len(prompt))
        return (
            f"[Local fallback] Processing: {prompt[:80]}{'...' if len(prompt) > 80 else ''} "
            f"This is a placeholder response from the AEGIS local generator."
        )

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """Generate completions for a list of prompts (sequential)."""
        results: List[str] = []
        for i, prompt in enumerate(prompts):
            logger.debug("generate_batch – prompt %d/%d", i + 1, len(prompts))
            results.append(self.generate(prompt, max_tokens, temperature, system_prompt))
        return results

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def embed(self, text: str) -> "np.ndarray":
        """Return embedding vector for *text* as a NumPy array."""
        if self.provider == "openai" and self._client is not None and HAS_OPENAI:
            try:
                return self._embed_openai([text])[0]
            except Exception as exc:
                logger.warning("OpenAI embedding failed (%s); falling back to local", exc)

        if self._embed_model is not None and HAS_ST:
            vec = self._embed_model.encode(text, normalize_embeddings=True, show_progress_bar=False)
            if not HAS_NUMPY:
                vec = list(vec)
            return np.array(vec) if HAS_NUMPY else np.array(vec)  # type: ignore[arg-type]

        # Hash fallback
        raw = _simple_hash_embed(text, self._embed_dim)
        return np.array(raw, dtype=np.float64) if HAS_NUMPY else np.array(raw)  # type: ignore[arg-type]

    def embed_batch(self, texts: List[str]) -> List["np.ndarray"]:
        """Return list of embedding vectors for *texts*."""
        if self.provider == "openai" and self._client is not None and HAS_OPENAI:
            try:
                return self._embed_openai(texts)
            except Exception as exc:
                logger.warning("OpenAI batch embedding failed (%s); falling back", exc)

        if self._embed_model is not None and HAS_ST:
            vecs = self._embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            if not HAS_NUMPY:
                return [np.array(list(v)) for v in vecs]
            return [np.asarray(v) for v in vecs]

        return [self.embed(t) for t in texts]

    def _embed_openai(self, texts: List[str]) -> List["np.ndarray"]:
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        resp = self._client.embeddings.create(model=model, input=texts)
        vecs = [np.array(d.embedding, dtype=np.float64) for d in resp.data]
        return vecs

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embedding vectors."""
        return self._embed_dim

    def is_available(self) -> bool:
        """Return True if the primary generation backend is reachable."""
        if self.provider in ("anthropic", "openai"):
            return self._client is not None
        # Local is always "available"
        return True
