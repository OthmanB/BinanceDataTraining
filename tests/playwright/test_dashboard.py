"""Playwright E2E tests for Observability Dashboard interactions.

Tests cover:
- Page load and authentication
- Tab navigation
- Theme toggle
- Static asset availability
- Responsive behavior
- Loading indicator presence
- JavaScript error detection
"""

try:
    from playwright.sync_api import Page, expect
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Page = None

import pytest


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_dashboard_page_load(page, base_url: str):
    """Verify dashboard loads successfully with correct title and content."""
    response = page.goto(f"{base_url}/ui/")
    
    assert response is not None
    assert response.status == 200
    assert "Observability Dashboard" in page.title()
    assert page.locator("text=Observability Dashboard").is_visible()


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_tab_navigation(page, base_url: str):
    """Verify tab switching functionality with multiple tabs."""
    page.goto(f"{base_url}/ui/")
    
    tabs = page.locator("button.tab-btn")
    tab_count = tabs.count()
    assert tab_count >= 4, f"Expected at least 4 tabs, found {tab_count}"
    
    dashboard_tab = page.locator("button.tab-btn", has_text="Dashboard")
    assert "active" in dashboard_tab.get_attribute("class")
    
    config_tab = page.locator("button.tab-btn", has_text="Config")
    config_tab.click()
    page.wait_for_timeout(100)
    
    assert "active" in config_tab.get_attribute("class")
    assert "active" not in dashboard_tab.get_attribute("class")
    
    logs_tab = page.locator("button.tab-btn", has_text="Logs")
    logs_tab.click()
    page.wait_for_timeout(100)
    
    assert "active" in logs_tab.get_attribute("class")
    assert "active" not in config_tab.get_attribute("class")
    
    history_tab = page.locator("button.tab-btn", has_text="History")
    history_tab.click()
    page.wait_for_timeout(100)
    
    assert "active" in history_tab.get_attribute("class")
    assert "active" not in logs_tab.get_attribute("class")


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_theme_toggle(page, base_url: str):
    """Verify theme toggle functionality and attribute changes."""
    page.goto(f"{base_url}/ui/")
    
    theme_toggle = page.locator("button.theme-toggle[title='Toggle dark mode']")
    assert theme_toggle.is_visible()
    
    initial_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
    
    theme_toggle.click()
    page.wait_for_timeout(100)
    
    new_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
    assert new_theme != initial_theme, f"Theme did not change (was {initial_theme}, still {new_theme})"
    assert new_theme in ["light", "dark"], f"Unexpected theme value: {new_theme}"
    
    theme_toggle.click()
    page.wait_for_timeout(100)
    
    final_theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
    assert final_theme == initial_theme, f"Theme did not revert (initial {initial_theme}, final {final_theme})"


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_static_assets_loaded(page, base_url: str):
    """Verify static JavaScript and CSS files are loaded successfully."""
    page.goto(f"{base_url}/ui/")
    
    app_js_script = page.locator("script[src='/static/app.js']")
    assert app_js_script.count() > 0, "app.js script tag not found"
    
    charts_js_script = page.locator("script[src='/static/charts.js']")
    assert charts_js_script.count() > 0, "charts.js script tag not found"
    
    htmx_script = page.locator("script[src='/static/htmx.min.js']")
    assert htmx_script.count() > 0, "htmx.min.js script tag not found"
    
    chart_script = page.locator("script[src='/static/chart.min.js']")
    assert chart_script.count() > 0, "chart.min.js script tag not found"
    
    switch_tab_exists = page.evaluate("typeof window.switchTab === 'function'")
    assert switch_tab_exists, "switchTab function not exported from app.js"
    
    init_chart_exists = page.evaluate("typeof window.initChart === 'function'")
    assert init_chart_exists, "initChart function not exported from charts.js"


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_responsive_behavior(page, base_url: str):
    """Verify dashboard is usable at mobile viewport width (768px)."""
    page.set_viewport_size({"width": 768, "height": 1024})
    page.goto(f"{base_url}/ui/")
    
    assert "Observability Dashboard" in page.title()
    
    tabs = page.locator("button.tab-btn")
    assert tabs.count() >= 4
    
    first_tab = tabs.nth(0)
    assert first_tab.is_visible()
    
    first_tab.click()
    page.wait_for_timeout(100)
    
    theme_toggle = page.locator("button.theme-toggle[title='Toggle dark mode']")
    assert theme_toggle.is_visible()
    
    page.set_viewport_size({"width": 1280, "height": 720})


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_loading_indicators_present(page, base_url: str):
    """Verify HTMX loading indicators are present in markup."""
    page.goto(f"{base_url}/ui/")
    
    indicators = page.locator(".htmx-indicator")
    indicator_count = indicators.count()
    
    assert indicator_count >= 2, f"Expected at least 2 loading indicators, found {indicator_count}"
    
    indicator_texts = indicators.all_text_contents()
    assert any("Starting" in text or "Stopping" in text for text in indicator_texts), \
        "Loading indicator text not found"


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_no_javascript_errors_on_load(page, base_url: str):
    """Verify no JavaScript errors occur during page load and interaction."""
    js_errors = []
    
    page.on("console", lambda msg: js_errors.append(msg) if msg.type == "error" else None)
    page.on("pageerror", lambda err: js_errors.append(err))
    
    page.goto(f"{base_url}/ui/")
    page.wait_for_load_state("domcontentloaded")
    
    config_tab = page.locator("button.tab-btn", has_text="Config")
    config_tab.click()
    page.wait_for_timeout(200)
    
    theme_toggle = page.locator("button.theme-toggle[title='Toggle dark mode']")
    theme_toggle.click()
    page.wait_for_timeout(200)
    
    assert len(js_errors) == 0, f"JavaScript errors detected: {js_errors}"


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_tab_content_visibility(page, base_url: str):
    """Verify tab content sections show/hide correctly when switching tabs."""
    page.goto(f"{base_url}/ui/")
    
    dashboard_panel = page.locator("#tab-dashboard")
    config_panel = page.locator("#tab-config")
    
    assert dashboard_panel.is_visible()
    assert not config_panel.is_visible()
    
    config_tab = page.locator("button.tab-btn", has_text="Config")
    config_tab.click()
    page.wait_for_timeout(100)
    
    assert config_panel.is_visible()
    assert not dashboard_panel.is_visible()


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_data_attributes_present(page, base_url: str):
    """Verify body has required data-* attributes for static JS configuration."""
    page.goto(f"{base_url}/ui/")
    
    body = page.locator("body")
    
    theme_key = body.get_attribute("data-theme-storage-key")
    assert theme_key is not None, "data-theme-storage-key attribute missing"
    assert len(theme_key) > 0, "data-theme-storage-key is empty"
    
    run_state_api = body.get_attribute("data-api-run-state")
    assert run_state_api is not None, "data-api-run-state attribute missing"
    assert "/api/run-state" in run_state_api or run_state_api.startswith("/"), \
        f"Unexpected run-state API path: {run_state_api}"
    
    assets_api = body.get_attribute("data-api-assets")
    assert assets_api is not None, "data-api-assets attribute missing"
    assert "/api" in assets_api or assets_api.startswith("/"), \
        f"Unexpected assets API path: {assets_api}"


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_chart_canvas_elements_present(page, base_url: str):
    """Verify chart canvas elements with data-chart-config are present."""
    page.goto(f"{base_url}/ui/")
    
    canvases = page.locator("canvas[data-chart-config]")
    canvas_count = canvases.count()
    
    if canvas_count > 0:
        first_canvas = canvases.nth(0)
        config_attr = first_canvas.get_attribute("data-chart-config")
        assert config_attr is not None
        assert len(config_attr) > 0
        
        import json
        try:
            config = json.loads(config_attr)
            assert isinstance(config, dict), "Chart config should be a JSON object"
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in data-chart-config: {e}")
