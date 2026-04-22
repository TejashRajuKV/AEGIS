"""
LLMClient – Claude / LLM API integration using the Anthropic SDK.

Provides synchronous and async generation with exponential-backoff
rate limiting and a mock fallback for demo/offline mode.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import anthropic

    HAS_ANTHROPIC = True
except Exception:
    anthropic = None  # type: ignore[assignment]
    HAS_ANTHROPIC = False

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    logger.debug("python-dotenv not available")


# ---------------------------------------------------------------------------
# Mock responses for offline / demo mode
# ---------------------------------------------------------------------------
_MOCK_RESPONSES: Dict[str, str] = {
    "default": (
        "Based on the analysis, I recommend reviewing the model's training data "
        "for demographic imbalances and applying re-sampling or re-weighting "
        "techniques to improve fairness across protected groups."
    ),
    "bias_fix": (
        "To mitigate the identified bias, implement the following:\n"
        "1. Apply adversarial debiasing to the training pipeline.\n"
        "2. Use equalized odds post-processing.\n"
        "3. Add fairness constraints to the loss function."
    ),
    "json_example": json.dumps(
        {"recommendation": "apply_reweighting", "confidence": 0.85},
        indent=2,
    ),
}


class LLMClient:
    """Anthropic Claude API client with retry logic and mock fallback.

    Parameters
    ----------
    api_key:
        Anthropic API key.  If None, reads ``ANTHROPIC_API_KEY`` env var.
    model:
        Model identifier (e.g. ``'claude-sonnet-4-20250514'``).
    max_tokens:
        Default maximum tokens for generation.
    max_retries:
        Maximum retry attempts with exponential backoff.
    base_delay:
        Initial delay in seconds between retries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay

        self._client: Optional[Any] = None
        self._available: Optional[bool] = None

        if HAS_ANTHROPIC:
            key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
            if key:
                try:
                    self._client = anthropic.Anthropic(api_key=key)
                    self._available = True
                    logger.info("LLMClient – initialised Anthropic client (model=%s)", model)
                except Exception as exc:
                    logger.warning("Failed to initialise Anthropic client: %s", exc)
                    self._available = False
            else:
                logger.info("LLMClient – no API key found; using mock mode")
                self._available = False
        else:
            logger.info("LLMClient – anthropic SDK not installed; using mock mode")
            self._available = False

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Return True if the Anthropic client is ready."""
        if self._available is None:
            # Fix CRIT-07: determine availability from whether the client was
            # successfully constructed, NOT by making a live API probe.
            # A probe blocks for up to 60 s on network failure and would freeze
            # the first /api/code_fix/ request entirely.
            self._available = self._client is not None
        return self._available

    # ------------------------------------------------------------------
    # Synchronous generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate a text response.

        Falls back to mock responses if the API is unavailable.
        """
        if not self.is_available() or self._client is None:
            logger.info("LLMClient – using mock response (API unavailable)")
            return self._mock_response(prompt)

        tokens = max_tokens or self.max_tokens
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                resp = self._client.messages.create(**kwargs)
                text_parts = []
                for block in resp.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                result = "\n".join(text_parts).strip()
                logger.debug("LLMClient – generated %d chars (attempt %d)", len(result), attempt)
                return result

            except Exception as exc:  # includes anthropic.APIError variants
                last_exc = exc
                delay = self.base_delay * (2 ** (attempt - 1))
                # Fix MED-05: guard anthropic error types — if the SDK is not
                # installed, anthropic is None and attribute access would raise
                # AttributeError.  Use generic Exception matching instead.
                is_rate_limit = (
                    HAS_ANTHROPIC
                    and anthropic is not None
                    and isinstance(exc, anthropic.RateLimitError)  # type: ignore[attr-defined]
                )
                if is_rate_limit:
                    logger.warning(
                        "Rate limited (attempt %d/%d); waiting %.1fs",
                        attempt, self.max_retries, delay,
                    )
                else:
                    logger.warning(
                        "API error (attempt %d/%d): %s; retrying in %.1fs",
                        attempt, self.max_retries, exc, delay,
                    )
                time.sleep(delay)

        logger.warning("All retries exhausted; falling back to mock response")
        if last_exc:
            logger.debug("Last error: %s", last_exc)
        return self._mock_response(prompt)

    # ------------------------------------------------------------------
    # JSON generation
    # ------------------------------------------------------------------
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a response and parse it as JSON.

        If the response is not valid JSON, attempts to extract JSON from
        the text.  Falls back to a default JSON structure.
        """
        sys = system_prompt or "You must respond with valid JSON only. No markdown, no explanation."
        raw = self.generate(prompt, system_prompt=sys)

        # Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
        import re
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Braces extraction
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse JSON from response; returning default structure")
        return {"raw_response": raw, "parsed": False}

    # ------------------------------------------------------------------
    # Async generation (for WebSocket streaming)
    # ------------------------------------------------------------------
    async def async_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> str:
        """Async wrapper around :meth:`generate`.

        Bug 24 fix: `generate()` uses `time.sleep()` for retries. Running it
        inside `run_in_executor` isolates the blocking sleep to a thread-pool
        worker so the asyncio event loop is never frozen.
        """
        # Fix MED-01: use get_running_loop() — get_event_loop() is deprecated in
        # Python 3.10+ and emits DeprecationWarning when called from async context.
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(
                prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )

    async def async_generate_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> str:
        """Async retry loop using asyncio.sleep (non-blocking).

        Preferred over async_generate when running directly in an async
        context where blocking the thread pool is also undesirable.
        """
        if not self.is_available() or self._client is None:
            return self._mock_response(prompt)

        tokens = max_tokens or self.max_tokens
        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # Fix MED-01: get_running_loop() instead of deprecated get_event_loop()
                loop = asyncio.get_running_loop()
                kwargs: dict = {
                    "model": self.model,
                    "max_tokens": tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                resp = await loop.run_in_executor(None, lambda: self._client.messages.create(**kwargs))
                text_parts = [block.text for block in resp.content if hasattr(block, "text")]
                return "\n".join(text_parts).strip()

            except Exception as exc:
                last_error = str(exc)
                delay = self.base_delay * (2 ** (attempt - 1))
                logger.warning("async retry %d/%d failed: %s; waiting %.1fs",
                               attempt, self.max_retries, exc, delay)
                await asyncio.sleep(delay)  # non-blocking sleep

        logger.warning("All async retries exhausted; using mock response")
        return self._mock_response(prompt)

    # ------------------------------------------------------------------
    # Mock responses
    # ------------------------------------------------------------------
    @staticmethod
    def _mock_response(prompt: str) -> str:
        """Return a contextual mock response for demo/offline mode."""
        prompt_lower = prompt.lower()
        if "fix" in prompt_lower or "mitigate" in prompt_lower or "debias" in prompt_lower:
            return _MOCK_RESPONSES["bias_fix"]
        if "json" in prompt_lower or "format" in prompt_lower:
            return _MOCK_RESPONSES["json_example"]
        return _MOCK_RESPONSES["default"]
