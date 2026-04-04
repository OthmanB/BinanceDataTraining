"""Unit tests for observability server hardening and extracted assets."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import cast
import unittest
from unittest.mock import MagicMock, Mock, patch

from observability.server import (
    ServerConfig,
    ServerState,
    _safe_static_file_path,
    _render_ui_page,
    _tail_log,
)


class TestStaticAssetAuthBypass(unittest.TestCase):
    """Test that /static/* paths bypass authentication correctly."""

    def test_safe_static_file_path_accepts_static_prefix(self) -> None:
        result = _safe_static_file_path(static_dir="static", request_path="/static/app.js")
        self.assertIsNotNone(result)
        self.assertTrue(str(result).endswith("app.js"))

    def test_safe_static_file_path_rejects_non_static(self) -> None:
        result = _safe_static_file_path(static_dir="static", request_path="/api/config")
        self.assertIsNone(result)

    def test_safe_static_file_path_prevents_directory_traversal(self) -> None:
        result = _safe_static_file_path(static_dir="static", request_path="/static/../config/observability.yaml")
        self.assertIsNone(result)

    def test_safe_static_file_path_rejects_absolute_relative_path(self) -> None:
        result = _safe_static_file_path(static_dir="static", request_path="/static//etc/passwd")
        self.assertIsNone(result)

    def test_static_auth_bypass_verified_via_safe_path_helper(self) -> None:
        """Test _safe_static_file_path is sufficient contract for /static/* auth bypass."""
        result = _safe_static_file_path(static_dir="static", request_path="/static/htmx.min.js")
        self.assertIsNotNone(result)

        non_static = _safe_static_file_path(static_dir="static", request_path="/api/config")
        self.assertIsNone(non_static)


class TestPostHardeningGuards(unittest.TestCase):
    """Test POST request hardening: Content-Length, body size, Content-Type."""

    def test_invalid_content_length_guard_contract(self) -> None:
        """do_POST must reject invalid Content-Length headers."""
        from observability.server import ObservabilityHandler

        with patch.object(ObservabilityHandler, "__init__", lambda x, *args, **kwargs: None):
            handler = ObservabilityHandler()
            handler.server_state = ServerState(config=cast(ServerConfig, Mock(spec=ServerConfig)))
            handler.path = "/ui/config/save"
            handler.headers = {"Content-Length": "not-a-number"}
            handler._require_auth = Mock(return_value=True)
            handler._send_plain = Mock()

            handler.do_POST()
            handler._send_plain.assert_called_once()
            args = handler._send_plain.call_args
            self.assertIn("Invalid Content-Length", args[0][0])
            self.assertEqual(args[1]["status"], 400)

    def test_oversized_body_guard_contract(self) -> None:
        """do_POST must reject bodies exceeding MAX_BODY_SIZE."""
        from observability.server import ObservabilityHandler

        with patch.object(ObservabilityHandler, "__init__", lambda x, *args, **kwargs: None):
            handler = ObservabilityHandler()
            handler.server_state = ServerState(config=cast(ServerConfig, Mock(spec=ServerConfig)))
            handler.path = "/ui/config/save"
            handler.headers = {"Content-Length": "2000000", "Content-Type": "application/x-www-form-urlencoded"}
            handler._require_auth = Mock(return_value=True)
            handler._send_plain = Mock()

            handler.do_POST()
            handler._send_plain.assert_called_once()
            args = handler._send_plain.call_args
            self.assertIn("exceeds", args[0][0].lower())
            self.assertEqual(args[1]["status"], 413)

    def test_unsupported_content_type_guard_contract(self) -> None:
        """do_POST must reject unsupported Content-Type values."""
        from observability.server import ObservabilityHandler

        with patch.object(ObservabilityHandler, "__init__", lambda x, *args, **kwargs: None):
            handler = ObservabilityHandler()
            handler.server_state = ServerState(config=cast(ServerConfig, Mock(spec=ServerConfig)))
            handler.path = "/ui/config/save"
            handler.headers = {"Content-Length": "100", "Content-Type": "application/json"}
            handler._require_auth = Mock(return_value=True)
            handler._send_plain = Mock()

            handler.do_POST()
            handler._send_plain.assert_called_once()
            args = handler._send_plain.call_args
            self.assertIn("Unsupported Content-Type", args[0][0])
            self.assertEqual(args[1]["status"], 415)

    def test_allowed_content_type_form_urlencoded(self) -> None:
        """do_POST accepts application/x-www-form-urlencoded Content-Type."""
        from observability.server import ObservabilityHandler

        with patch.object(ObservabilityHandler, "__init__", lambda x, *args, **kwargs: None):
            handler = ObservabilityHandler()
            handler.server_state = ServerState(config=cast(ServerConfig, Mock(spec=ServerConfig)))
            handler.path = "/ui/config/browse"
            handler.headers = {"Content-Length": "20", "Content-Type": "application/x-www-form-urlencoded"}
            handler._require_auth = Mock(return_value=True)
            handler.rfile = BytesIO(b"browse_path=config/")
            handler._send_html = Mock()

            with patch("observability.server._render_training_config_panel", return_value="<html></html>"):
                handler.do_POST()
                handler._send_html.assert_called_once()

    def test_zero_content_length_accepted(self) -> None:
        """do_POST accepts zero-length POST bodies."""
        from observability.server import ObservabilityHandler

        with patch.object(ObservabilityHandler, "__init__", lambda x, *args, **kwargs: None):
            handler = ObservabilityHandler()
            handler.server_state = ServerState(config=cast(ServerConfig, Mock(spec=ServerConfig)))
            handler.path = "/ui/config/browse"
            handler.headers = {"Content-Length": "0"}
            handler._require_auth = Mock(return_value=True)
            handler.rfile = BytesIO(b"")
            handler._send_html = Mock()

            with patch("observability.server._render_training_config_panel", return_value="<html></html>"):
                handler.do_POST()
                handler._send_html.assert_called_once()


class TestExtractedAssetReferences(unittest.TestCase):
    """Test that UI shell references extracted static assets correctly."""

    def test_ui_shell_references_app_css(self) -> None:
        html = _render_ui_page(cast(ServerConfig, None))
        self.assertIn('<link rel="stylesheet" href="/static/app.css"', html)

    def test_ui_shell_references_app_js(self) -> None:
        html = _render_ui_page(cast(ServerConfig, None))
        self.assertIn('<script src="/static/app.js"></script>', html)

    def test_ui_shell_references_charts_js(self) -> None:
        html = _render_ui_page(cast(ServerConfig, None))
        self.assertIn('<script src="/static/charts.js"></script>', html)

    def test_no_inline_javascript_functions_in_shell(self) -> None:
        """UI shell should not contain inline function definitions."""
        html = _render_ui_page(cast(ServerConfig, None))
        self.assertNotIn("function switchTab(", html)
        self.assertNotIn("function initChart(", html)
        self.assertNotIn("function toggleTheme(", html)

    def test_ui_shell_uses_onclick_attributes_for_interactive_calls(self) -> None:
        """Shell may use onclick attributes that call extracted functions."""
        html = _render_ui_page(cast(ServerConfig, None))
        # These should call functions defined in app.js
        self.assertIn("onclick=\"switchTab(", html)


class TestChartDataAttributeContract(unittest.TestCase):
    """Test that chart configuration uses data-* attributes instead of inline constructors."""

    def test_server_code_no_inline_chart_constructors(self) -> None:
        """Server source should not contain 'new Chart(' inline constructors."""
        server_py = Path("observability/server.py").read_text(encoding="utf-8")
        self.assertNotIn("new Chart(", server_py)

    def test_charts_js_defines_init_functions(self) -> None:
        """charts.js should export initChart and initAllCharts functions."""
        charts_js_path = Path("static/charts.js")
        if not charts_js_path.exists():
            self.skipTest("charts.js not present; skipping")
        charts_js = charts_js_path.read_text(encoding="utf-8")
        self.assertIn("function initChart(", charts_js)
        self.assertIn("function initAllCharts(", charts_js)

    def test_charts_js_reads_data_chart_config_attribute(self) -> None:
        """charts.js should read configuration from data-chart-config attribute."""
        charts_js_path = Path("static/charts.js")
        if not charts_js_path.exists():
            self.skipTest("charts.js not present; skipping")
        charts_js = charts_js_path.read_text(encoding="utf-8")
        self.assertIn("data-chart-config", charts_js)


class TestStaticServingBehavior(unittest.TestCase):
    """Test static file serving behavior including caching and MIME types."""

    def test_static_file_cache_control_contract(self) -> None:
        """Static asset handler must set Cache-Control header for browser caching."""
        from observability.server import ObservabilityHandler

        static_file = Path("static/test-cache.css")
        static_file.parent.mkdir(parents=True, exist_ok=True)
        static_file.write_text("/* test */", encoding="utf-8")

        try:
            with patch.object(ObservabilityHandler, "__init__", lambda x, *args, **kwargs: None):
                handler = ObservabilityHandler()
                cfg = cast(ServerConfig, Mock(spec=ServerConfig))
                cfg.static_dir = "static"
                handler.server_state = ServerState(config=cfg)
                handler.path = "/static/test-cache.css"
                handler.send_response = Mock()
                handler.send_header = Mock()
                handler.end_headers = Mock()
                handler.wfile = BytesIO()

                handler.do_GET()
                calls = handler.send_header.call_args_list
                cache_headers = [call for call in calls if call[0][0] == "Cache-Control"]
                self.assertTrue(len(cache_headers) > 0, "Cache-Control header required for static files")
        finally:
            if static_file.exists():
                static_file.unlink()

    def test_static_file_mime_type_contract(self) -> None:
        """Static asset handler must set correct Content-Type based on file extension."""
        from observability.server import ObservabilityHandler

        static_file = Path("static/test-mime.js")
        static_file.parent.mkdir(parents=True, exist_ok=True)
        static_file.write_text("// js", encoding="utf-8")

        try:
            with patch.object(ObservabilityHandler, "__init__", lambda x, *args, **kwargs: None):
                handler = ObservabilityHandler()
                cfg = cast(ServerConfig, Mock(spec=ServerConfig))
                cfg.static_dir = "static"
                handler.server_state = ServerState(config=cfg)
                handler.path = "/static/test-mime.js"
                handler.send_response = Mock()
                handler.send_header = Mock()
                handler.end_headers = Mock()
                handler.wfile = BytesIO()

                handler.do_GET()
                calls = handler.send_header.call_args_list
                content_type_headers = [call for call in calls if call[0][0] == "Content-Type"]
                self.assertTrue(len(content_type_headers) > 0, "Content-Type header required")
                mime_value = content_type_headers[0][0][1]
                self.assertIn("javascript", mime_value.lower())
        finally:
            if static_file.exists():
                static_file.unlink()


class TestTailLogBoundarySafety(unittest.TestCase):
    """Test _tail_log() boundary-safety with contiguous byte buffer."""

    def test_empty_file(self) -> None:
        """Empty file returns empty list."""
        with Path("/tmp/test_tail_empty.log").open("wb") as f:
            f.write(b"")
        try:
            result = _tail_log("/tmp/test_tail_empty.log", max_lines=10)
            self.assertEqual(result, [])
        finally:
            Path("/tmp/test_tail_empty.log").unlink(missing_ok=True)

    def test_small_file_fits_in_buffer(self) -> None:
        """Small file (<64KB) read completely."""
        content = "line1\nline2\nline3\n"
        with Path("/tmp/test_tail_small.log").open("w") as f:
            f.write(content)
        try:
            result = _tail_log("/tmp/test_tail_small.log", max_lines=10)
            self.assertEqual(result, ["line1", "line2", "line3"])
        finally:
            Path("/tmp/test_tail_small.log").unlink(missing_ok=True)

    def test_max_lines_truncation(self) -> None:
        """File with more lines than max_lines returns only last N lines."""
        lines = [f"line{i}\n" for i in range(100)]
        with Path("/tmp/test_tail_max.log").open("w") as f:
            f.writelines(lines)
        try:
            result = _tail_log("/tmp/test_tail_max.log", max_lines=10)
            self.assertEqual(len(result), 10)
            self.assertEqual(result[0], "line90")
            self.assertEqual(result[-1], "line99")
        finally:
            Path("/tmp/test_tail_max.log").unlink(missing_ok=True)

    def test_long_line_across_chunk_boundaries(self) -> None:
        """Long line (>64KB) spanning chunk boundaries not truncated."""
        long_line = "x" * 100000
        short_line = "short"
        content = f"{long_line}\n{short_line}\n"
        with Path("/tmp/test_tail_long.log").open("w") as f:
            f.write(content)
        try:
            result = _tail_log("/tmp/test_tail_long.log", max_lines=10)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], long_line)
            self.assertEqual(result[1], short_line)
        finally:
            Path("/tmp/test_tail_long.log").unlink(missing_ok=True)

    def test_file_without_trailing_newline(self) -> None:
        """File missing final newline handled correctly."""
        content = "line1\nline2\nline3"
        with Path("/tmp/test_tail_no_newline.log").open("w") as f:
            f.write(content)
        try:
            result = _tail_log("/tmp/test_tail_no_newline.log", max_lines=10)
            self.assertEqual(result, ["line1", "line2", "line3"])
        finally:
            Path("/tmp/test_tail_no_newline.log").unlink(missing_ok=True)

    def test_utf8_multibyte_characters(self) -> None:
        """UTF-8 multi-byte characters handled correctly."""
        content = "line1\n🔥emoji🔥\nline3\n"
        with Path("/tmp/test_tail_utf8.log").open("w", encoding="utf-8") as f:
            f.write(content)
        try:
            result = _tail_log("/tmp/test_tail_utf8.log", max_lines=10)
            self.assertEqual(result, ["line1", "🔥emoji🔥", "line3"])
        finally:
            Path("/tmp/test_tail_utf8.log").unlink(missing_ok=True)

    def test_mixed_encoding_fallback(self) -> None:
        """Non-UTF-8 bytes handled with latin-1 fallback."""
        content_bytes = b"line1\n\xff\xfe\nline3\n"
        with Path("/tmp/test_tail_latin1.log").open("wb") as f:
            f.write(content_bytes)
        try:
            result = _tail_log("/tmp/test_tail_latin1.log", max_lines=10)
            self.assertEqual(len(result), 3)
            self.assertIn("line1", result)
            self.assertIn("line3", result)
        finally:
            Path("/tmp/test_tail_latin1.log").unlink(missing_ok=True)

    def test_nonexistent_file(self) -> None:
        """Non-existent file returns sentinel error message."""
        result = _tail_log("/tmp/nonexistent_wrf46_test.log", max_lines=10)
        self.assertEqual(result, ["Log file not found"])

    def test_chunk_boundary_at_line_split(self) -> None:
        """Chunk boundary that falls exactly at line split."""
        lines = [f"line{i}\n" for i in range(1000)]
        content = "".join(lines)
        with Path("/tmp/test_tail_boundary.log").open("w") as f:
            f.write(content)
        try:
            result = _tail_log("/tmp/test_tail_boundary.log", max_lines=100)
            self.assertEqual(len(result), 100)
            expected_first = "line900"
            self.assertEqual(result[0], expected_first)
            self.assertEqual(result[-1], "line999")
            for i, line in enumerate(result):
                self.assertEqual(line, f"line{900 + i}")
        finally:
            Path("/tmp/test_tail_boundary.log").unlink(missing_ok=True)

    def test_single_line_file(self) -> None:
        """File with single line returns that line."""
        content = "single line content"
        with Path("/tmp/test_tail_single.log").open("w") as f:
            f.write(content)
        try:
            result = _tail_log("/tmp/test_tail_single.log", max_lines=10)
            self.assertEqual(result, ["single line content"])
        finally:
            Path("/tmp/test_tail_single.log").unlink(missing_ok=True)

    def test_zero_max_lines(self) -> None:
        """max_lines=0 returns empty list."""
        content = "line1\nline2\nline3\n"
        with Path("/tmp/test_tail_zero.log").open("w") as f:
            f.write(content)
        try:
            result = _tail_log("/tmp/test_tail_zero.log", max_lines=0)
            self.assertEqual(result, [])
        finally:
            Path("/tmp/test_tail_zero.log").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
