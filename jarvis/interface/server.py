"""
Optional API server — allows other applications to call JARVIS.
"""

import asyncio
import json
import logging

from aiohttp import web

from ..core.router import SelfModifyingRouter, RouterConfig
from .cli import JarvisCLI

logger = logging.getLogger(__name__)


class JarvisServer:
    """HTTP API server wrapping the JARVIS core."""

    def __init__(self, cli: JarvisCLI, host: str = "127.0.0.1", port: int = 8420):
        self.cli = cli
        self.host = host
        self.port = port
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        self.app.router.add_post("/query", self._handle_query)
        self.app.router.add_get("/status", self._handle_status)
        self.app.router.add_get("/health", self._handle_health)

    async def _handle_query(self, request: web.Request) -> web.Response:
        data = await request.json()
        query = data.get("query", "")
        if not query:
            return web.json_response({"error": "No query provided"}, status=400)

        response = await self.cli.process_query(query)
        return web.json_response({"response": response})

    async def _handle_status(self, _request: web.Request) -> web.Response:
        history = self.cli.router.get_modification_history()
        return web.json_response({
            "steps": self.cli.router._step_count,
            "modifications": len(history),
            "memory_online": self.cli.memory.available,
            "recent_coherence": (
                [h["coherence"] for h in history[-5:]] if history else []
            ),
        })

    async def _handle_health(self, _request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    def run(self):
        """Start the server."""
        logger.info(f"Starting JARVIS server on {self.host}:{self.port}")
        web.run_app(self.app, host=self.host, port=self.port)
