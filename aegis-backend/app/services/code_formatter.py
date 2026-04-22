"""
CodeFormatter – formats generated Python code with proper indentation,
docstrings, imports, function wrapping, and diff generation.

Uses the ``ast`` module for safe code manipulation where possible.
"""

import ast
import difflib
import logging
import re
import textwrap
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# Default section order for generated code
_DEFAULT_IMPORTS = [
    "import numpy as np",
    "import pandas as pd",
    "from sklearn.base import BaseEstimator, TransformerMixin",
]


class CodeFormatter:
    """Formats and manipulates generated Python code for the AEGIS platform."""

    def __init__(
        self,
        indent_size: int = 4,
        max_line_length: int = 100,
    ) -> None:
        self.indent_size = indent_size
        self.max_line_length = max_line_length
        self._indent_str = " " * indent_size

    # ------------------------------------------------------------------
    # Core formatting
    # ------------------------------------------------------------------
    def format_code(self, code: str) -> str:
        """Format code with consistent indentation and structure.

        - Strips leading/trailing whitespace
        - Ensures consistent indentation
        - Removes excessive blank lines (≥3 consecutive → 2)
        - Adds a trailing newline
        """
        if not code or not code.strip():
            return ""

        # Strip and normalise line endings
        lines = code.strip().splitlines()

        # Detect base indentation
        base_indent = self._detect_base_indent(lines)
        if base_indent > 0:
            lines = [l[base_indent:] if l.startswith(" " * base_indent) else l for l in lines]

        # Dedent (handles mixed indentation)
        dedented = textwrap.dedent("\n".join(lines)).splitlines()

        # Collapse multiple blank lines
        formatted: List[str] = []
        blank_count = 0
        for line in dedented:
            if line.strip() == "":
                blank_count += 1
                if blank_count <= 2:
                    formatted.append("")
            else:
                blank_count = 0
                formatted.append(line)

        # Strip trailing blank lines, ensure single trailing newline
        result = "\n".join(formatted).rstrip() + "\n"
        return result

    # ------------------------------------------------------------------
    # Docstring management
    # ------------------------------------------------------------------
    def add_docstring(self, code: str, description: str) -> str:
        """Add or replace the module-level docstring.

        If the code already starts with a docstring, replaces it.
        Otherwise, prepends one.
        """
        if not description.strip():
            return code

        formatted_code = self.format_code(code)
        docstring = self._make_docstring(description)

        # Check if code already has a module-level docstring
        try:
            tree = ast.parse(formatted_code)
            if (
                tree.body
                and isinstance(tree.body[0], ast.Expr)
                and isinstance(tree.body[0].value, ast.Constant)
            ):
                # Replace existing docstring
                # Find where the docstring ends
                first_node_end = getattr(tree.body[0], "end_lineno", None)
                if first_node_end:
                    lines = formatted_code.splitlines()
                    new_lines = [docstring] + lines[first_node_end:]
                    return "\n".join(new_lines)
        except SyntaxError:
            pass

        # Prepend docstring
        return docstring + "\n" + formatted_code

    # ------------------------------------------------------------------
    # Import management
    # ------------------------------------------------------------------
    def add_imports(self, code: str, imports: List[str]) -> str:
        """Add import statements to the top of the code.

        Deduplicates against existing imports and inserts after any
        existing docstring.
        """
        if not imports:
            return code

        formatted_code = self.format_code(code)
        existing_imports = self._extract_import_lines(formatted_code)

        # Filter out duplicates
        new_imports = []
        for imp in imports:
            imp_stripped = imp.strip()
            if imp_stripped and imp_stripped not in existing_imports:
                new_imports.append(imp_stripped)
                existing_imports.add(imp_stripped)

        if not new_imports:
            return formatted_code

        # Find insertion point (after docstring, before first non-import)
        lines = formatted_code.splitlines()
        insert_idx = self._find_import_section_end(lines)

        import_block = "\n".join(new_imports) + "\n"
        new_lines = lines[:insert_idx] + [import_block] + lines[insert_idx:]
        return "\n".join(new_lines)

    # ------------------------------------------------------------------
    # Function wrapping
    # ------------------------------------------------------------------
    def wrap_in_function(
        self,
        code: str,
        func_name: str,
        params: Optional[List[str]] = None,
    ) -> str:
        """Wrap code inside a function definition.

        Parameters
        ----------
        code:
            The code body to wrap.
        func_name:
            Name of the function.
        params:
            List of parameter names (e.g. ``['X', 'y', 'groups']``).
            If None, uses a default ``**kwargs``.
        """
        formatted_code = self.format_code(code)

        # Build parameter list
        if params:
            param_str = ", ".join(params)
        else:
            param_str = "**kwargs"

        func_header = f"def {func_name}({param_str}):\n"
        docstring = self._make_docstring(
            f"Auto-generated function: {func_name}",
            indent=self._indent_str,
        )
        docstring += "\n"

        # Indent the body
        body_lines = []
        for line in formatted_code.rstrip().splitlines():
            if line.strip():
                body_lines.append(self._indent_str + line)
            else:
                body_lines.append("")

        return func_header + docstring + "\n".join(body_lines) + "\n"

    # ------------------------------------------------------------------
    # Diff generation
    # ------------------------------------------------------------------
    @staticmethod
    def generate_diff(original: str, modified: str) -> str:
        """Generate a unified diff between original and modified code.

        Returns
        -------
        Unified diff string, or a message if files are identical.
        """
        orig_lines = original.splitlines(keepends=True)
        mod_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            orig_lines,
            mod_lines,
            fromfile="original",
            tofile="modified",
            lineterm="\n",
        )
        diff_text = "".join(diff)
        return diff_text if diff_text else "No differences found."

    # ------------------------------------------------------------------
    # Patch application
    # ------------------------------------------------------------------
    def apply_fix(self, original_code: str, fix_code: str) -> str:
        """Apply a fix by replacing the original code with the fix.

        If the fix is a function body, tries to intelligently merge
        imports and replace the relevant section.

        Returns
        -------
        Patched code string.
        """
        formatted_original = self.format_code(original_code)
        formatted_fix = self.format_code(fix_code)

        # Check if the fix is a function definition
        fix_funcs = self._extract_function_names(formatted_fix)

        if fix_funcs:
            # Replace matching functions in the original
            patched = self._replace_functions(formatted_original, formatted_fix, fix_funcs)
            # Merge imports
            fix_imports = self._extract_import_lines(formatted_fix)
            original_imports = self._extract_import_lines(formatted_original)
            new_imports = fix_imports - original_imports
            if new_imports:
                patched = self.add_imports(patched, sorted(new_imports))
            return patched

        # No functions found – return the fix as-is (prepend original imports)
        original_imports = self._extract_import_lines(formatted_original)
        fix_imports = self._extract_import_lines(formatted_fix)
        missing_imports = original_imports - fix_imports
        if missing_imports:
            return self.add_imports(formatted_fix, sorted(missing_imports))

        return formatted_fix

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_base_indent(lines: List[str]) -> int:
        """Detect the smallest non-zero indentation level."""
        min_indent = float("inf")
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if 0 < indent < min_indent:
                    min_indent = indent
        return min_indent if min_indent != float("inf") else 0

    @staticmethod
    def _make_docstring(text: str, indent: str = "") -> str:
        """Create a triple-quoted docstring."""
        wrapped = textwrap.fill(text, width=76)
        lines = wrapped.splitlines()
        if len(lines) == 1:
            return f'{indent}"""{lines[0]}"""'
        parts = [f'{indent}"""\n{indent}{lines[0]}']
        parts.extend([f"{indent}{l}" for l in lines[1:]])
        parts.append(f'{indent}"""')
        return "\n".join(parts)

    @staticmethod
    def _extract_import_lines(code: str) -> set:
        """Extract all import statements from code as a set."""
        imports = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = ", ".join(a.name for a in node.names)
                    imports.add(f"import {names}")
                elif isinstance(node, ast.ImportFrom):
                    names = ", ".join(a.name for a in node.names)
                    if node.module:
                        imports.add(f"from {node.module} import {names}")
        except SyntaxError:
            # Fallback to regex
            for match in re.finditer(r"^(import .+|from .+ import .+)$", code, re.MULTILINE):
                imports.add(match.group(0))
        return imports

    @staticmethod
    def _find_import_section_end(lines: List[str]) -> int:
        """Find the line index after the last import statement (after docstring)."""
        in_docstring = False
        docstring_end = 0

        # Find docstring end
        for i, line in enumerate(lines):
            stripped = line.strip()
            if '"""' in stripped or "'''" in stripped:
                if not in_docstring:
                    in_docstring = True
                    # Check if single-line docstring
                    if stripped.count('"""') >= 2 or stripped.count("'''") >= 2:
                        in_docstring = False
                        docstring_end = i + 1
                else:
                    in_docstring = False
                    docstring_end = i + 1

        if not docstring_end:
            docstring_end = 0

        # Find last import after docstring
        last_import = docstring_end
        for i in range(docstring_end, len(lines)):
            stripped = lines[i].strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import = i + 1

        return last_import

    @staticmethod
    def _extract_function_names(code: str) -> List[str]:
        """Extract top-level function names from code."""
        names = []
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    names.append(node.name)
        except SyntaxError:
            # Regex fallback
            names = re.findall(r"^def (\w+)\s*\(", code, re.MULTILINE)
        return names

    @staticmethod
    def _replace_functions(
        original: str, fix: str, func_names: List[str]
    ) -> str:
        """Replace functions in original with versions from fix."""
        try:
            orig_tree = ast.parse(original)
            fix_tree = ast.parse(fix)
        except SyntaxError:
            return fix

        # Build map of fix functions by name
        fix_funcs = {}
        for node in fix_tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in func_names:
                fix_funcs[node.name] = node

        # Replace in original
        new_body = []
        for node in orig_tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in fix_funcs:
                new_body.append(fix_funcs[node.name])
            else:
                new_body.append(node)

        # Reconstruct (this will lose comments but is safe)
        return ast.unparse(ast.Module(body=new_body, type_ignores=[])) + "\n"
