"""Minimal Playwright fixture smoke test.

Verifies that pytest fixtures can start/stop server and create browser context.
This is NOT the full E2E test suite (Task 15) - just infrastructure validation.
"""
import pytest

try:
    from playwright.sync_api import Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Page = None


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_playwright_fixtures_import():
    """Verify conftest.py imports without errors."""
    import tests.playwright.conftest
    assert hasattr(tests.playwright.conftest, 'observability_server')
    assert hasattr(tests.playwright.conftest, 'browser_context')
    assert hasattr(tests.playwright.conftest, 'page')


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_server_fixture_provides_running_process(observability_server):
    """Verify observability_server fixture starts a process."""
    assert observability_server is not None
    assert observability_server.poll() is None


@pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright not installed")
def test_page_fixture_provides_authenticated_page(page, base_url: str):
    """Verify page fixture can navigate to authenticated dashboard."""
    response = page.goto(f"{base_url}/ui")
    assert response is not None
    assert response.status == 200
    assert "Observability Dashboard" in page.title()
