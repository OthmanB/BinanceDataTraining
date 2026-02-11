"""Regression tests for observability UI HTML/JS rendering."""

from __future__ import annotations

import unittest

from observability.server import _render_ui_page


class TestObservabilityUiRendering(unittest.TestCase):
    def test_tab_switch_controls_are_rendered(self) -> None:
        html = _render_ui_page(None)

        self.assertIn("function switchTab(btn,name)", html)
        self.assertIn("onclick=\"switchTab(this,'dashboard')\"", html)
        self.assertIn("onclick=\"switchTab(this,'config')\"", html)
        self.assertIn("onclick=\"switchTab(this,'logs')\"", html)
        self.assertIn("onclick=\"switchTab(this,'history')\"", html)

    def test_connection_yaml_builder_uses_escaped_newlines(self) -> None:
        html = _render_ui_page(None)

        start = html.find("function _syncConnTA(){")
        self.assertGreaterEqual(start, 0)
        end = html.find("function removeConnRow", start)
        self.assertGreater(end, start)
        snippet = html[start:end]

        self.assertIn("part.join('\\n')", snippet)
        self.assertIn("lines.join('\\n')", snippet)
        self.assertNotIn("lines.push('- name:", snippet)


if __name__ == "__main__":
    unittest.main()
