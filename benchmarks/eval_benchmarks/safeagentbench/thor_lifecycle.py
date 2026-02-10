"""AI2-THOR container lifecycle helpers.

Convenience wrappers for starting/stopping the Docker container from Python.
The container is expected to be managed externally (docker compose up/down),
but these helpers can be used for programmatic control.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

DOCKER_DIR = Path(__file__).parent / "docker"


def start_server(
    port: int = 9100,
    timeout: float = 120.0,
    build: bool = True,
) -> None:
    """Start the AI2-THOR Docker container and wait for readiness.

    Args:
        port: Host port (must match docker-compose.yml).
        timeout: Seconds to wait for /health to respond.
        build: Whether to rebuild the image before starting.
    """
    cmd = ["docker", "compose", "up", "-d"]
    if build:
        cmd.append("--build")

    log.info(f"Starting AI2-THOR container: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=DOCKER_DIR, check=True)

    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=5.0)
            if resp.status_code == 200:
                log.info("AI2-THOR container is ready.")
                return
        except httpx.ConnectError:
            pass
        time.sleep(2.0)

    raise TimeoutError(
        f"AI2-THOR container did not become ready within {timeout}s"
    )


def stop_server() -> None:
    """Stop the AI2-THOR Docker container."""
    log.info("Stopping AI2-THOR container...")
    subprocess.run(
        ["docker", "compose", "down"],
        cwd=DOCKER_DIR,
        check=True,
    )
    log.info("AI2-THOR container stopped.")
