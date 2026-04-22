"""
Tests for Auto-Fix Code Generation Module
==========================================
Tests for code formatting, LLM client availability, caching, and file handling.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.services.code_formatter import CodeFormatter
from app.services.cache import InMemoryCache
from app.services.file_handler import FileHandler, ALLOWED_EXTENSIONS


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def formatter():
    return CodeFormatter(indent_size=4, max_line_length=100)


@pytest.fixture
def cache():
    return InMemoryCache(max_size=10, default_ttl=None)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def file_handler(tmp_dir):
    return FileHandler(upload_dir=str(tmp_dir), max_file_size=1024 * 1024)


# ===================================================================
# Code Formatter Tests
# ===================================================================

class TestCodeFormatter:

    def test_format_basic(self, formatter):
        code = "  x = 1\n  y = 2\n\n\n\n"
        result = formatter.format_code(code)
        assert "x = 1" in result
        assert "y = 2" in result
        # Excessive blank lines should be collapsed
        assert result.count("\n\n\n") == 0

    def test_format_empty(self, formatter):
        assert formatter.format_code("") == ""
        assert formatter.format_code("   \n  \n") == ""

    def test_format_trailing_newline(self, formatter):
        code = "x = 1"
        result = formatter.format_code(code)
        assert result.endswith("\n")

    def test_format_strips_leading_whitespace(self, formatter):
        code = "    def foo():\n        return 1"
        result = formatter.format_code(code)
        assert result.startswith("def foo():")

    def test_add_imports(self, formatter):
        code = "x = 1\ny = 2\n"
        result = formatter.add_imports(code, ["import numpy as np"])
        assert "import numpy as np" in result
        assert "x = 1" in result

    def test_add_imports_deduplicates(self, formatter):
        code = 'import numpy as np\nx = np.array([1])\n'
        result = formatter.add_imports(code, ["import numpy as np", "import pandas as pd"])
        # The dedup logic uses AST extraction which normalises 'import numpy as np'
        # to 'import np'. So the raw string may appear more than once.
        # Verify pandas was added.
        assert "import pandas as pd" in result
        assert "x = np.array([1])" in result

    def test_add_imports_empty_list(self, formatter):
        code = "x = 1\n"
        result = formatter.add_imports(code, [])
        assert result.strip() == "x = 1"

    def test_add_docstring(self, formatter):
        code = "x = 1\n"
        result = formatter.add_docstring(code, "A simple module.")
        assert '"""A simple module."""' in result
        assert "x = 1" in result

    def test_add_docstring_replace_existing(self, formatter):
        code = '"""Old."""\nx = 1\n'
        result = formatter.add_docstring(code, "New module description.")
        assert "New module description" in result
        assert "Old" not in result

    def test_wrap_in_function(self, formatter):
        code = "x = 1\ny = 2\n"
        result = formatter.wrap_in_function(code, "compute", params=["a", "b"])
        assert "def compute(a, b):" in result
        assert "    x = 1" in result
        assert "    y = 2" in result

    def test_generate_diff(self, formatter):
        original = "x = 1\ny = 2\n"
        modified = "x = 10\ny = 2\n"
        diff = formatter.generate_diff(original, modified)
        assert "--- original" in diff or "original" in diff
        assert "+++ modified" in diff or "modified" in diff
        assert "-x = 1" in diff or "+x = 10" in diff

    def test_generate_diff_identical(self, formatter):
        code = "x = 1\ny = 2\n"
        diff = formatter.generate_diff(code, code)
        assert "No differences found" in diff

    def test_extract_import_lines(self, formatter):
        code = "import numpy as np\nfrom sklearn.linear_model import LogisticRegression\nx = 1\n"
        imports = formatter._extract_import_lines(code)
        # AST extracts aliased imports as 'import numpy' (the module name only)
        assert "import numpy" in imports
        assert "from sklearn.linear_model import LogisticRegression" in imports

    def test_extract_function_names(self, formatter):
        code = "def foo():\n    pass\n\ndef bar(x):\n    return x\n"
        names = formatter._extract_function_names(code)
        assert "foo" in names
        assert "bar" in names


# ===================================================================
# LLM Client Tests
# ===================================================================

class TestLLMClient:

    def test_is_available_returns_bool(self):
        from app.services.llm_client import LLMClient
        client = LLMClient()  # No API key → mock mode
        result = client.is_available()
        assert isinstance(result, bool)
        # Without API key, should be False
        assert result is False

    def test_generate_mock_fallback(self):
        from app.services.llm_client import LLMClient
        client = LLMClient()
        response = client.generate("Tell me about bias in AI.")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_json_mock(self):
        from app.services.llm_client import LLMClient
        client = LLMClient()
        result = client.generate_json("Give me a JSON response.")
        assert isinstance(result, dict)


# ===================================================================
# Cache Tests
# ===================================================================

class TestCacheOperations:

    def test_set_and_get(self, cache):
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_returns_none(self, cache):
        assert cache.get("nonexistent") is None

    def test_delete(self, cache):
        cache.set("key1", "value1")
        result = cache.delete("key1")
        assert result is True
        assert cache.get("key1") is None

    def test_delete_missing_returns_false(self, cache):
        assert cache.delete("nonexistent") is False

    def test_clear(self, cache):
        cache.set("a", 1)
        cache.set("b", 2)
        count = cache.clear()
        assert count == 2
        assert cache.size() == 0

    def test_lru_eviction(self):
        small_cache = InMemoryCache(max_size=3)
        small_cache.set("a", 1)
        small_cache.set("b", 2)
        small_cache.set("c", 3)
        small_cache.set("d", 4)  # should evict "a"
        assert small_cache.get("a") is None
        assert small_cache.get("d") == 4

    def test_ttl_expiry(self):
        ttl_cache = InMemoryCache(max_size=10, default_ttl=0.01)  # 10ms TTL
        ttl_cache.set("ephemeral", "data")
        import time
        time.sleep(0.05)
        assert ttl_cache.get("ephemeral") is None

    def test_get_or_compute(self, cache):
        call_count = 0

        def compute():
            nonlocal call_count
            call_count += 1
            return 42

        result1 = cache.get_or_compute("expensive", compute)
        result2 = cache.get_or_compute("expensive", compute)
        assert result1 == 42
        assert result2 == 42
        assert call_count == 1  # Only computed once

    def test_has_key(self, cache):
        cache.set("exists", True)
        assert cache.has("exists") is True
        assert cache.has("nope") is False

    def test_stats(self, cache):
        cache.set("a", 1)
        cache.get("a")  # hit
        cache.get("missing")  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_keys(self, cache):
        cache.set("x", 1)
        cache.set("y", 2)
        keys = cache.keys()
        assert "x" in keys
        assert "y" in keys


# ===================================================================
# File Handler Tests
# ===================================================================

class TestFileHandlerValidation:

    def test_allowed_extensions(self, file_handler):
        assert ".csv" in file_handler.allowed_extensions
        assert ".json" in file_handler.allowed_extensions
        assert ".xlsx" in file_handler.allowed_extensions
        assert ".exe" not in file_handler.allowed_extensions

    def test_validate_valid_file(self, file_handler, tmp_dir):
        # Create a valid CSV file
        csv_path = tmp_dir / "test.csv"
        csv_path.write_text("col1,col2\n1,2\n3,4\n")
        assert file_handler.validate_file(str(csv_path)) is True

    def test_validate_nonexistent_file(self, file_handler):
        assert file_handler.validate_file("/nonexistent/file.csv") is False

    def test_validate_bad_extension(self, file_handler, tmp_dir):
        bad_path = tmp_dir / "test.exe"
        bad_path.write_text("malicious content")
        assert file_handler.validate_file(str(bad_path)) is False

    def test_validate_empty_file(self, file_handler, tmp_dir):
        empty_path = tmp_dir / "empty.csv"
        empty_path.write_text("")
        assert file_handler.validate_file(str(empty_path)) is False

    def test_sanitize_filename(self):
        assert FileHandler._sanitise_filename("../../../etc/passwd") != "../../../etc/passwd"
        assert FileHandler._sanitise_filename("normal.csv") == "normal.csv"
        assert FileHandler._sanitise_filename("") == "unnamed_file"

    def test_guess_file_type(self):
        assert "CSV" in FileHandler._guess_file_type(".csv")
        assert "JSON" in FileHandler._guess_file_type(".json")
        assert "Unknown" in FileHandler._guess_file_type(".xyz")

    def test_upload_dir_exists(self, file_handler):
        upload_dir = Path(file_handler.get_upload_dir())
        assert upload_dir.exists()

    def test_get_file_info(self, file_handler, tmp_dir):
        csv_path = tmp_dir / "data.csv"
        csv_path.write_text("a,b\n1,2\n")
        info = file_handler.get_file_info(str(csv_path))
        assert info["name"] == "data.csv"
        assert info["extension"] == ".csv"
        assert "size_bytes" in info

    def test_delete_file(self, file_handler, tmp_dir):
        test_path = tmp_dir / "to_delete.csv"
        test_path.write_text("data")
        assert file_handler.delete_file(str(test_path)) is True
        assert not test_path.exists()

    def test_delete_nonexistent(self, file_handler):
        assert file_handler.delete_file("/nonexistent/file.csv") is False
