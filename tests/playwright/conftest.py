"""Pytest fixtures for Playwright E2E tests of observability dashboard.

Provides:
- observability_server: Starts/stops observability server with config
- browser_context: Authenticated browser context with Basic Auth
- page: Playwright page in authenticated context
- base_url: Server base URL
- auth_credentials: Basic Auth credentials (user, password)

All fixtures handle deterministic server startup with health checks and
reliable teardown after test completion.
"""
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page
else:
    Browser = Any
    BrowserContext = Any
    Page = Any

playwright_sync_api = pytest.importorskip("playwright.sync_api", reason="Playwright not installed")
sync_playwright = playwright_sync_api.sync_playwright


# Logging setup
logger = logging.getLogger(__name__)

# Constants
DEFAULT_USER = "obs"
DEFAULT_PASSWORD = "obs"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8008
STARTUP_TIMEOUT_S = 10
HEALTH_CHECK_INTERVAL_S = 0.5
SERVER_SHUTDOWN_TIMEOUT_S = 5


@pytest.fixture(scope="session")
def auth_credentials() -> tuple[str, str]:
    """Return Basic Auth credentials (user, password) for observability server.
    
    Reads from environment variables OBSERVABILITY_USER and OBSERVABILITY_PASSWORD.
    Falls back to defaults if not set.
    
    Returns:
        Tuple[str, str]: (username, password)
    """
    user = os.environ.get("OBSERVABILITY_USER", DEFAULT_USER)
    password = os.environ.get("OBSERVABILITY_PASSWORD", DEFAULT_PASSWORD)
    return user, password


@pytest.fixture(scope="session")
def server_host() -> str:
    return os.environ.get("OBSERVABILITY_HOST", DEFAULT_HOST)


def _pick_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def server_port(server_host: str) -> int:
    env_port = os.environ.get("OBSERVABILITY_PORT")
    if env_port:
        return int(env_port)
    return _pick_free_port(server_host)


@pytest.fixture(scope="session")
def base_url(server_host: str, server_port: int) -> str:
    """Return base URL for observability server.
    
    Reads host/port from config/observability.yaml defaults.
    Can be overridden via OBSERVABILITY_HOST and OBSERVABILITY_PORT env vars.
    
    Returns:
        str: Base URL (e.g., "http://127.0.0.1:8008")
    """
    return f"http://{server_host}:{server_port}"


@pytest.fixture(scope="session")
def observability_server(
    base_url: str,
    auth_credentials: tuple[str, str],
    server_host: str,
    server_port: int,
) -> Generator[subprocess.Popen[str], None, None]:
    """Start observability server as subprocess, wait for readiness, yield process, teardown.
    
    Server startup:
    1. Sets OBSERVABILITY_USER and OBSERVABILITY_PASSWORD env vars
    2. Starts server via `python -m observability.server --config config/observability.yaml`
    3. Polls /healthz endpoint until HTTP 200 or timeout
    4. Yields process handle to tests
    5. Terminates process on teardown with timeout enforcement
    
    Args:
        base_url: Server base URL from base_url fixture
        auth_credentials: (user, password) from auth_credentials fixture
    
    Yields:
        subprocess.Popen: Running server process
    
    Raises:
        RuntimeError: If server fails to start or become healthy within timeout
    """
    user, password = auth_credentials
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "config" / "observability.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Set auth env vars for server process
    env = os.environ.copy()
    env["OBSERVABILITY_USER"] = user
    env["OBSERVABILITY_PASSWORD"] = password
    env["OBSERVABILITY_HOST"] = server_host
    env["OBSERVABILITY_PORT"] = str(server_port)
    
    # Start server
    cmd = [
        sys.executable,
        "-m", "observability.server",
        "--config", str(config_path)
    ]
    
    logger.info(f"Starting observability server: {' '.join(cmd)}")
    log_file = tempfile.NamedTemporaryFile(
        mode="w+",
        encoding="utf-8",
        prefix="observability-server-",
        suffix=".log",
        delete=False,
    )
    log_path = Path(log_file.name)
    process = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait for server to be healthy
        import requests

        healthz_url = f"{base_url}/healthz"
        start_time = time.time()
        ready = False

        while time.time() - start_time < STARTUP_TIMEOUT_S:
            # Check if process crashed
            if process.poll() is not None:
                log_file.flush()
                log_file.seek(0)
                output = log_file.read()
                raise RuntimeError(
                    f"Server process exited with code {process.returncode}\n"
                    f"LOG ({log_path}):\n{output}"
                )

            # Check health endpoint
            try:
                response = requests.get(healthz_url, timeout=2)
                if response.status_code == 200:
                    ready = True
                    logger.info(f"Server healthy at {healthz_url} after {time.time() - start_time:.1f}s")
                    break
            except requests.RequestException:
                pass  # Not ready yet

            time.sleep(HEALTH_CHECK_INTERVAL_S)

        if not ready:
            process.terminate()
            try:
                process.wait(timeout=SERVER_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log_file.flush()
            log_file.seek(0)
            output = log_file.read()
            raise RuntimeError(
                f"Server failed to become healthy at {healthz_url} within {STARTUP_TIMEOUT_S}s\n"
                f"LOG ({log_path}):\n{output}"
            )

        # Yield to tests
        yield process

        # Teardown: terminate server
        logger.info("Terminating observability server")
        process.terminate()
        try:
            process.wait(timeout=SERVER_SHUTDOWN_TIMEOUT_S)
            logger.info(f"Server terminated with exit code {process.returncode}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Server did not terminate within {SERVER_SHUTDOWN_TIMEOUT_S}s, killing")
            process.kill()
            process.wait()
    finally:
        log_file.close()
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to remove Playwright server log file: %s", log_path)


@pytest.fixture(scope="session")
def playwright_browser() -> Generator[Browser, None, None]:
    """Launch Playwright browser (Chromium) for session, yield, close.
    
    Uses sync_playwright context manager to launch browser.
    Browser is shared across all tests in session (faster).
    
    Yields:
        Browser: Playwright browser instance (Chromium)
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        logger.info("Playwright browser launched (Chromium, headless)")
        yield browser
        browser.close()
        logger.info("Playwright browser closed")


@pytest.fixture(scope="session")
def browser(playwright_browser: Browser) -> Generator[Browser, None, None]:
    """Alias for playwright_browser fixture for backward compatibility.
    
    Yields:
        Browser: Playwright browser instance (Chromium)
    """
    yield playwright_browser


@pytest.fixture(scope="function")
def browser_context(
    playwright_browser: Browser,
    base_url: str,
    auth_credentials: tuple[str, str],
    observability_server: subprocess.Popen[str],
) -> Generator[BrowserContext, None, None]:
    """Create authenticated browser context with Basic Auth for each test.
    
    Context includes:
    - HTTP Basic Auth credentials pre-configured
    - Base URL set for relative navigation
    - Isolated cookies/storage per test
    
    Args:
        playwright_browser: Shared browser from playwright_browser fixture
        base_url: Server URL from base_url fixture
        auth_credentials: (user, password) from auth_credentials fixture
        observability_server: Running server process (ensures server is started)
    
    Yields:
        BrowserContext: Isolated context with auth configured
    """
    user, password = auth_credentials
    context = playwright_browser.new_context(
        base_url=base_url,
        http_credentials={"username": user, "password": password}
    )
    logger.debug(f"Browser context created with Basic Auth ({user}:***)")
    yield context
    context.close()
    logger.debug("Browser context closed")


@pytest.fixture(scope="function")
def page(browser_context: BrowserContext) -> Generator[Page, None, None]:
    """Create new page in authenticated context for each test.
    
    Page is isolated per test and automatically authenticated via context.
    
    Args:
        browser_context: Authenticated context from browser_context fixture
    
    Yields:
        Page: Playwright page ready for navigation
    """
    page = browser_context.new_page()
    logger.debug("Page created in authenticated context")
    yield page
    page.close()
    logger.debug("Page closed")
