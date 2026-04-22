"""
AutoFixGenerator – generates Python fix code when bias is detected.

Uses the :class:`LLMClient` (Claude) to produce actionable mitigation code
with explanations and expected improvements.
"""

import ast
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FixResult:
    """Result of an auto-fix generation request."""

    fix_type: str
    code: str
    explanation: str
    expected_improvement: str
    imports_needed: List[str]
    is_valid_syntax: bool
    syntax_error: Optional[str] = None


# ---------------------------------------------------------------------------
# System prompts for each fix type
# ---------------------------------------------------------------------------
_SYSTEM_PROMPTS = {
    "preprocessing": (
        "You are an AI fairness expert. Given a bias report about a machine learning "
        "model, generate Python code that implements a PREPROCESSING fix to mitigate "
        "the identified bias. The fix should work with scikit-learn pipelines.\n\n"
        "Requirements:\n"
        "- Output ONLY the Python code inside a single code block\n"
        "- Include necessary imports at the top\n"
        "- Add a docstring explaining the fix\n"
        "- The code should be a callable function or class\n"
        "- Use standard libraries (sklearn, numpy, pandas) only\n"
    ),
    "threshold": (
        "You are an AI fairness expert. Given a bias report about a machine learning "
        "model, generate Python code that implements THRESHOLD ADJUSTMENT to mitigate "
        "bias. This should include finding optimal thresholds per demographic group.\n\n"
        "Requirements:\n"
        "- Output ONLY the Python code inside a single code block\n"
        "- Include necessary imports at the top\n"
        "- Add a docstring explaining the approach\n"
        "- The code should accept model predictions and protected attribute labels\n"
        "- Use standard libraries (sklearn, numpy, scipy) only\n"
    ),
    "reweighting": (
        "You are an AI fairness expert. Given a bias report about a machine learning "
        "model, generate Python code that implements FEATURE REWEIGHTING or SAMPLE "
        "REWEIGHTING to mitigate the identified bias.\n\n"
        "Requirements:\n"
        "- Output ONLY the Python code inside a single code block\n"
        "- Include necessary imports at the top\n"
        "- Add a docstring explaining the approach\n"
        "- The code should work with pandas DataFrames\n"
        "- Use standard libraries (sklearn, numpy, pandas, scipy) only\n"
    ),
    "general": (
        "You are an AI fairness expert. Given a bias report about a machine learning "
        "model, generate Python code that mitigates the identified bias.\n\n"
        "Requirements:\n"
        "- Output ONLY the Python code inside a single code block\n"
        "- Include necessary imports at the top\n"
        "- Add a docstring explaining the fix\n"
        "- Use standard libraries (sklearn, numpy, pandas) only\n"
    ),
}


class AutoFixGenerator:
    """Generates Python fix code when bias is detected in a model.

    Uses the Claude LLM API to produce contextual, actionable mitigation
    code. Falls back to built-in template-based fixes when the API is
    unavailable.
    """

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        """
        Parameters
        ----------
        llm_client:
            An :class:`LLMClient` instance.  If None, a default one is created.
        """
        if llm_client is not None:
            self._llm = llm_client
        else:
            # Lazy import to avoid circular dependency issues
            from app.services.llm_client import LLMClient

            self._llm = LLMClient()
        logger.info("AutoFixGenerator initialised (llm_available=%s)", self._llm.is_available())

    # ------------------------------------------------------------------
    # Main generation
    # ------------------------------------------------------------------
    def generate_fix(
        self,
        bias_report: Dict[str, Any],
        model_type: str = "sklearn",
        code_context: str = "",
    ) -> FixResult:
        """Generate a fix based on a bias report.

        Parameters
        ----------
        bias_report:
            Dict containing bias metrics, affected features/groups, and
            severity information.
        model_type:
            Type of model ('sklearn', 'pytorch', 'tensorflow', 'xgboost').
        code_context:
            Existing code context to make the fix more targeted.

        Returns
        -------
        FixResult with generated code, explanation, and metadata.
        """
        bias_index = bias_report.get("bias_index", 0)
        categories = bias_report.get("categories_affected", [])
        severity = bias_report.get("severity", "unknown")

        # Determine fix type based on bias characteristics
        fix_type = self._determine_fix_type(bias_report)

        # Build prompt
        prompt = self._build_prompt(bias_report, model_type, code_context, fix_type)
        system_prompt = _SYSTEM_PROMPTS.get(fix_type, _SYSTEM_PROMPTS["general"])

        # Generate code
        raw_response = self._llm.generate(prompt, system_prompt=system_prompt)
        code = self._extract_code(raw_response)

        # Validate
        is_valid, error_msg = self.validate_fix_syntax(code)

        # If invalid, fall back to template
        if not is_valid:
            logger.warning("Generated code has syntax errors; using template fallback")
            code = self._template_fix(fix_type, bias_report, model_type)
            is_valid, error_msg = self.validate_fix_syntax(code)
            # Fix HIGH-08: if the template also produces invalid code, surface it
            # explicitly rather than returning broken code silently.
            if not is_valid:
                logger.error(
                    "Template fallback also produced invalid syntax (%s); "
                    "returning best-effort code with is_valid_syntax=False",
                    error_msg,
                )

        imports = self._extract_imports(code)
        explanation = self._generate_explanation(fix_type, bias_report, severity)
        improvement = self._estimate_improvement(bias_index, severity)

        return FixResult(
            fix_type=fix_type,
            code=code,
            explanation=explanation,
            expected_improvement=improvement,
            imports_needed=imports,
            is_valid_syntax=is_valid,
            syntax_error=error_msg,
        )

    # ------------------------------------------------------------------
    # Specialised fix generators
    # ------------------------------------------------------------------
    def generate_preprocessing_fix(self, bias_report: Dict[str, Any]) -> FixResult:
        """Generate a preprocessing pipeline fix."""
        report = dict(bias_report)
        report["_fix_hint"] = "preprocessing"
        return self.generate_fix(report)

    def generate_threshold_fix(self, bias_report: Dict[str, Any]) -> FixResult:
        """Generate a threshold adjustment fix."""
        report = dict(bias_report)
        report["_fix_hint"] = "threshold"
        return self.generate_fix(report)

    def generate_reweighting_fix(self, bias_report: Dict[str, Any]) -> FixResult:
        """Generate a feature/sample reweighting fix."""
        report = dict(bias_report)
        report["_fix_hint"] = "reweighting"
        return self.generate_fix(report)

    # ------------------------------------------------------------------
    # Code validation
    # ------------------------------------------------------------------
    @staticmethod
    def validate_fix_syntax(code: str) -> tuple:
        """Validate that generated code has correct Python syntax.

        Returns
        -------
        (is_valid, error_message)
        """
        if not code or not code.strip():
            return False, "Empty code"
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as exc:
            return False, f"Line {exc.lineno}: {exc.msg}"
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _determine_fix_type(bias_report: Dict[str, Any]) -> str:
        """Choose the most appropriate fix type based on the bias report."""
        hint = bias_report.get("_fix_hint", "")
        if hint in ("preprocessing", "threshold", "reweighting"):
            return hint

        metrics = bias_report.get("metrics", {})
        bias_index = bias_report.get("bias_index", 0)
        categories = bias_report.get("categories_affected", [])

        if "disparate_impact" in metrics or "demographic_parity" in metrics:
            if bias_index > 60:
                return "preprocessing"
            return "reweighting"

        if "equalized_odds" in metrics or any("threshold" in c.lower() for c in categories):
            return "threshold"

        # Default: use preprocessing for high bias, reweighting otherwise
        return "preprocessing" if bias_index > 50 else "reweighting"

    @staticmethod
    def _build_prompt(
        bias_report: Dict[str, Any],
        model_type: str,
        code_context: str,
        fix_type: str,
    ) -> str:
        """Build the generation prompt."""
        import json

        context_section = ""
        if code_context.strip():
            context_section = f"\n\nExisting code context:\n```\n{code_context[:2000]}\n```\n"

        prompt = (
            f"## Bias Report\n\n"
            f"```json\n{json.dumps(bias_report, indent=2, default=str)[:3000]}\n```\n\n"
            f"## Model Type\n{model_type}\n\n"
            f"## Requested Fix Type\n{fix_type}\n"
            f"{context_section}\n"
            f"Generate a Python fix for this bias issue."
        )
        return prompt

    @staticmethod
    def _extract_code(response: str) -> str:
        """Extract Python code from an LLM response (strip markdown fences)."""
        import re

        # Try fenced code block
        code_match = re.search(r"```python\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Try any code fence
        code_match = re.search(r"```\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # Return the full response if no fences found
        return response.strip()

    @staticmethod
    def _extract_imports(code: str) -> List[str]:
        """Extract import statements from generated code."""
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except Exception:
            # Fallback: regex
            import re
            imports = re.findall(r"^(?:import|from)\s+([\w.]+)", code, re.MULTILINE)
        return sorted(set(imports))

    @staticmethod
    def _generate_explanation(
        fix_type: str,
        bias_report: Dict[str, Any],
        severity: str,
    ) -> str:
        """Generate a human-readable explanation for the fix."""
        # Bug 19 fix: coerce to float safely so f-string :.1f never crashes
        try:
            bias_index = float(bias_report.get("bias_index", 0) or 0)
        except (TypeError, ValueError):
            bias_index = 0.0

        raw_affected = bias_report.get("categories_affected", ["multiple groups"])
        if not isinstance(raw_affected, (list, tuple)) or not raw_affected:
            raw_affected = ["multiple groups"]
        affected = [str(a) for a in raw_affected]

        explanations = {
            "preprocessing": (
                f"This preprocessing fix addresses {severity} bias (index={bias_index:.1f}) "
                f"affecting {', '.join(affected)}. It modifies the input data pipeline "
                f"to reduce disparate impact before model training, using techniques "
                f"such as resampling and feature transformation."
            ),
            "threshold": (
                f"This threshold adjustment fix targets {severity} bias (index={bias_index:.1f}) "
                f"for {', '.join(affected)}. It computes optimal classification thresholds "
                f"per demographic group to equalise outcomes while maintaining overall accuracy."
            ),
            "reweighting": (
                f"This reweighting fix mitigates {severity} bias (index={bias_index:.1f}) "
                f"across {', '.join(affected)}. It adjusts sample or feature weights "
                f"in the training data to ensure fair representation and reduce "
                f"demographic disparities."
            ),
        }
        return explanations.get(
            fix_type,
            f"Generated fix for {severity} bias (index={bias_index:.1f}) affecting {', '.join(affected)}.",
        )

    @staticmethod
    def _estimate_improvement(bias_index: float, severity: str) -> str:
        """Estimate expected improvement from the fix.

        Bug 20 fix: higher bias index means MORE room to improve (higher reduction %).
        Original logic was inverted.
        """
        # Coerce safely
        try:
            bias_index = float(bias_index or 0)
        except (TypeError, ValueError):
            bias_index = 0.0

        # High bias → high improvement potential (was backwards before)
        if bias_index > 70:
            low = round(bias_index * 0.5, 0)
            high = round(bias_index * 0.7, 0)
            return f"Expected to reduce bias index by 50-70% (from {bias_index:.0f} to ~{low:.0f}-{high:.0f})"
        elif bias_index > 40:
            low = round(bias_index * 0.6, 0)
            high = round(bias_index * 0.8, 0)
            return f"Expected to reduce bias index by 40-60% (from {bias_index:.0f} to ~{low:.0f}-{high:.0f})"
        else:
            low = round(bias_index * 0.7, 0)
            high = round(bias_index * 0.9, 0)
            return f"Expected to reduce bias index by 30-50% (from {bias_index:.0f} to ~{low:.0f}-{high:.0f})"

    # ------------------------------------------------------------------
    # Template-based fallback (when LLM is unavailable)
    # ------------------------------------------------------------------
    @staticmethod
    def _template_fix(
        fix_type: str,
        bias_report: Dict[str, Any],
        model_type: str,
    ) -> str:
        """Return a template-based fix when LLM generation fails."""
        protected_attr = bias_report.get("protected_attribute", "protected_group")
        target = bias_report.get("target_column", "label")

        if fix_type == "preprocessing":
            return (
                '"""\n'
                "Auto-generated preprocessing fix for bias mitigation.\n"
                "Applies resampling to balance protected groups in training data.\n"
                '"""\n'
                "import pandas as pd\n"
                "import numpy as np\n"
                "from sklearn.utils import resample\n\n\n"
                "def apply_preprocessing_fix(X, y, protected_attribute):\n"
                '    """\n'
                "    Resample the training data to balance demographic groups.\n\n"
                "    Parameters\n"
                "    ----------\n"
                "    X : pd.DataFrame or np.ndarray\n"
                "        Feature matrix.\n"
                "    y : array-like\n"
                "        Target labels.\n"
                "    protected_attribute : str or array-like\n"
                "        Name of the protected attribute column or array of values.\n\n"
                "    Returns\n"
                "    -------\n"
                "    X_balanced, y_balanced : resampled feature matrix and labels\n"
                '    """\n'
                "    if isinstance(X, pd.DataFrame):\n"
                "        groups = X[protected_attribute] if isinstance(protected_attribute, str) else protected_attribute\n"
                "    else:\n"
                "        groups = np.array(protected_attribute)\n\n"
                "    unique_groups = np.unique(groups)\n"
                "    max_size = max(np.sum(groups == g) for g in unique_groups)\n\n"
                "    X_resampled_list = []\n"
                "    y_resampled_list = []\n\n"
                "    for g in unique_groups:\n"
                "        mask = groups == g\n"
                "        X_g = X[mask]\n"
                "        y_g = y[mask]\n"
                "        if len(X_g) < max_size:\n"
                "            X_g, y_g = resample(X_g, y_g, n_samples=max_size, random_state=42)\n"
                "        X_resampled_list.append(X_g)\n"
                "        y_resampled_list.append(y_g)\n\n"
                "    if isinstance(X, pd.DataFrame):\n"
                "        return pd.concat(X_resampled_list), np.concatenate(y_resampled_list)\n"
                "    return np.vstack(X_resampled_list), np.concatenate(y_resampled_list)\n"
            )

        elif fix_type == "threshold":
            return (
                '"""\n'
                "Auto-generated threshold adjustment fix for bias mitigation.\n"
                "Computes optimal per-group classification thresholds.\n"
                '"""\n'
                "import numpy as np\n"
                "from sklearn.metrics import accuracy_score\n\n\n"
                "def compute_fair_thresholds(y_true, y_proba, groups, base_threshold=0.5):\n"
                '    """\n'
                "    Compute per-group thresholds that equalise positive prediction rates.\n\n"
                "    Parameters\n"
                "    ----------\n"
                "    y_true : array-like\n"
                "        Ground truth labels.\n"
                "    y_proba : array-like\n"
                "        Predicted probabilities.\n"
                "    groups : array-like\n"
                "        Protected group membership.\n"
                "    base_threshold : float\n"
                "        Default threshold.\n\n"
                "    Returns\n"
                "    -------\n"
                "    dict : mapping from group to optimal threshold\n"
                '    """\n'
                "    thresholds = {}\n"
                "    unique_groups = np.unique(groups)\n\n"
                "    for g in unique_groups:\n"
                "        mask = groups == g\n"
                "        proba_g = y_proba[mask]\n"
                "        true_g = y_true[mask]\n\n"
                "        best_t = base_threshold\n"
                "        best_acc = 0.0\n"
                "        for t in np.arange(0.1, 0.9, 0.05):\n"
                "            preds = (proba_g >= t).astype(int)\n"
                "            acc = accuracy_score(true_g, preds)\n"
                "            if acc > best_acc:\n"
                "                best_acc = acc\n"
                "                best_t = t\n"
                "        thresholds[g] = round(best_t, 3)\n\n"
                "    return thresholds\n\n\n"
                "def apply_fair_thresholds(y_proba, groups, thresholds):\n"
                '    """Apply group-specific thresholds to probability predictions."""\n'
                "    predictions = np.zeros(len(y_proba))\n"
                "    for g, t in thresholds.items():\n"
                "        mask = groups == g\n"
                "        predictions[mask] = (y_proba[mask] >= t).astype(int)\n"
                "    return predictions\n"
            )

        else:  # reweighting
            return (
                '"""\n'
                "Auto-generated reweighting fix for bias mitigation.\n"
                "Assigns sample weights to balance demographic representation.\n"
                '"""\n'
                "import numpy as np\n\n\n"
                "def compute_fairness_weights(groups, y):\n"
                '    """\n'
                "    Compute sample weights that equalise representation across groups.\n\n"
                "    Parameters\n"
                "    ----------\n"
                "    groups : array-like\n"
                "        Protected group membership.\n"
                "    y : array-like\n"
                "        Target labels.\n\n"
                "    Returns\n"
                "    -------\n"
                "    weights : np.ndarray of shape (n_samples,)\n"
                '    """\n'
                "    groups = np.asarray(groups)\n"
                "    y = np.asarray(y)\n"
                "    n = len(groups)\n"
                "    weights = np.ones(n)\n\n"
                "    unique_groups = np.unique(groups)\n"
                "    group_counts = {g: np.sum(groups == g) for g in unique_groups}\n"
                "    max_count = max(group_counts.values())\n\n"
                "    for g, count in group_counts.items():\n"
                "        mask = groups == g\n"
                "        weights[mask] = max_count / count\n\n"
                "    # Normalise weights\n"
                "    weights = weights / np.mean(weights)\n"
                "    return weights\n"
            )
