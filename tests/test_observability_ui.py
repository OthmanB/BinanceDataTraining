"""Regression tests for observability UI HTML/JS rendering."""

from __future__ import annotations

from pathlib import Path
from typing import cast
import unittest

from observability.server import ServerConfig, _render_ui_page


REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


class TestObservabilityUiRendering(unittest.TestCase):
    def test_ui_shell_references_extracted_javascript(self) -> None:
        html = _render_ui_page(cast(ServerConfig, None))

        self.assertIn('<script src="/static/app.js"></script>', html)
        self.assertNotIn("function switchTab(", html)
        self.assertIn("onclick=\"switchTab(this,'dashboard')\"", html)
        self.assertIn("onclick=\"switchTab(this,'config')\"", html)
        self.assertIn("onclick=\"switchTab(this,'logs')\"", html)
        self.assertIn("onclick=\"switchTab(this,'history')\"", html)

    def test_ui_shell_exposes_runtime_data_attributes(self) -> None:
        html = _render_ui_page(cast(ServerConfig, None))

        self.assertIn('data-theme-storage-key="obs-theme"', html)
        self.assertIn('data-api-run-state="/api/run-state"', html)
        self.assertIn('data-api-assets="/api/config/assets"', html)
        self.assertIn('data-nn-params="', html)
        self.assertIn("<!doctype html>", html)
        self.assertIn("</html>", html)

    def test_extracted_app_js_contains_interactive_ui_logic(self) -> None:
        app_js = _repo_path("static", "app.js").read_text(encoding="utf-8")

        self.assertIn("function switchTab(btn, name)", app_js)
        self.assertIn("function syncConnectionsTextarea()", app_js)
        self.assertIn("part.join(\"\\n\")", app_js)
        self.assertIn("lines.join(\"\\n\")", app_js)
        self.assertIn('document.body.addEventListener("htmx:beforeSwap"', app_js)
        self.assertIn('document.body.addEventListener("htmx:responseError"', app_js)
        self.assertIn("window._syncConnTA = syncConnectionsTextarea;", app_js)
        self.assertIn("window.switchTab = switchTab;", app_js)
        self.assertIn("window.toggleTheme = toggleTheme;", app_js)
        self.assertIn("window.filterLogs = filterLogs;", app_js)
        self.assertIn("window.toggleAutoScroll = toggleAutoScroll;", app_js)

    def test_ui_shell_includes_charts_js_script(self) -> None:
        html = _render_ui_page(cast(ServerConfig, None))
        self.assertIn('<script src="/static/charts.js"></script>', html)

    def test_charts_js_contains_initialization_logic(self) -> None:
        charts_js = _repo_path("static", "charts.js").read_text(encoding="utf-8")

        self.assertIn("function initChart(canvas)", charts_js)
        self.assertIn("function initAllCharts()", charts_js)
        self.assertIn('data-chart-config', charts_js)
        self.assertIn('typeof Chart === \'undefined\'', charts_js)
        self.assertIn('htmx:afterSwap', charts_js)
        self.assertIn('window.initChart = initChart;', charts_js)
        self.assertIn('window.initAllCharts = initAllCharts;', charts_js)

    def test_server_no_inline_chart_constructors(self) -> None:
        server_py = _repo_path("observability", "server.py").read_text(encoding="utf-8")
        self.assertNotIn("new Chart(", server_py)

    def test_app_css_contains_responsive_rules(self) -> None:
        app_css = _repo_path("static", "app.css").read_text(encoding="utf-8")
        self.assertIn("@media (max-width: 768px)", app_css)
        self.assertIn(".header { flex-direction: column;", app_css)
        self.assertIn(".header-controls { width: 100%; justify-content: space-between; flex-wrap: wrap; }", app_css)
        self.assertIn(".card-grid { grid-template-columns: 1fr; }", app_css)
        self.assertIn(".tab-bar { width: 100%; flex-wrap: wrap; }", app_css)
        self.assertIn(".config-fields { grid-template-columns: 1fr; }", app_css)


if __name__ == "__main__":
    unittest.main()
