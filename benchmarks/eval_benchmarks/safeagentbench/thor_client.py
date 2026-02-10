"""Async HTTP client for the AI2-THOR Docker action server.

The AI2-THOR simulator runs inside a Docker container (see docker/) as a
Flask HTTP server on port 9100. This client wraps the HTTP endpoints:
  /health, /reset, /execute, /execute_plan, /state, /screenshot

AI2-THOR's Unity engine is single-threaded, so all requests are serialized
with an asyncio.Lock. A singleton pattern ensures one client per URL.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

_clients: dict[str, "AsyncThorClient"] = {}


class AsyncThorClient:
    """Thin async wrapper around the Flask action server."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._lock = asyncio.Lock()
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

    async def health(self) -> dict[str, Any]:
        resp = await self._http.get("/health")
        resp.raise_for_status()
        return resp.json()

    async def reset(self, scene: str) -> dict[str, Any]:
        async with self._lock:
            resp = await self._http.post("/reset", json={"scene": scene})
            resp.raise_for_status()
            return resp.json()

    async def execute(self, instruction: str) -> dict[str, Any]:
        async with self._lock:
            resp = await self._http.post(
                "/execute", json={"instruction": instruction}
            )
            resp.raise_for_status()
            return resp.json()

    async def execute_plan(self, instructions: list[str]) -> list[dict[str, Any]]:
        async with self._lock:
            resp = await self._http.post(
                "/execute_plan", json={"instructions": instructions}
            )
            resp.raise_for_status()
            return resp.json()["results"]

    async def state(self) -> list[dict[str, Any]]:
        async with self._lock:
            resp = await self._http.get("/state")
            resp.raise_for_status()
            return resp.json()["objects"]

    async def screenshot_b64(self) -> str:
        """Return base64-encoded PNG screenshot."""
        async with self._lock:
            resp = await self._http.get("/screenshot")
            resp.raise_for_status()
            return resp.json()["image"]

    async def close(self) -> None:
        await self._http.aclose()


def get_thor_client(url: str = "http://localhost:9100") -> AsyncThorClient:
    """Return a singleton ``AsyncThorClient`` for the given URL."""
    if url not in _clients:
        _clients[url] = AsyncThorClient(url)
    return _clients[url]
