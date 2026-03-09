"""
MCP client for Watty — the persistent memory and knowledge graph spine.

Connects to the Watty MCP server for long-term memory and geometric coherence.
The Core reads/writes to this for persistent knowledge across sessions.

Key MCP tools:
- vault_query: semantic search over facts and beliefs
- vault_deposit: store new knowledge
- metabolize: extract and deposit knowledge from an interaction
- graph_query: spreading activation search over knowledge graph
- graph_build: build edges between knowledge nodes
- fed_inject: force specific beliefs into context
"""

import json
import logging
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

# Default MCP config location (Windows)
DEFAULT_MCP_CONFIG = Path.home() / "AppData" / "Roaming" / "Claude" / "claude_desktop_config.json"


class WattyClient:
    """
    MCP client for the Watty knowledge graph server.
    Handles connection, tool invocation, and graceful degradation.
    """

    def __init__(
        self,
        server_url: str | None = None,
        config_path: str | Path | None = None,
    ):
        self.server_url = server_url
        self.config_path = Path(config_path) if config_path else DEFAULT_MCP_CONFIG
        self._session: aiohttp.ClientSession | None = None
        self._connected = False

        if not server_url:
            self._load_config()

    def _load_config(self):
        """Load Watty connection info from MCP config."""
        if not self.config_path.exists():
            logger.warning(f"MCP config not found at {self.config_path}")
            return

        try:
            config = json.loads(self.config_path.read_text())
            watty_config = config.get("mcpServers", {}).get("watty", {})
            if watty_config:
                self.server_url = watty_config.get("url", watty_config.get("command"))
                logger.info(f"Loaded Watty config: {self.server_url}")
            else:
                logger.warning("No 'watty' server found in MCP config")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse MCP config: {e}")

    async def connect(self) -> bool:
        """Establish connection to Watty MCP server."""
        if not self.server_url:
            logger.warning("No Watty server URL configured")
            return False

        try:
            self._session = aiohttp.ClientSession()
            # Ping the server
            async with self._session.get(f"{self.server_url}/health") as resp:
                if resp.status == 200:
                    self._connected = True
                    logger.info("Connected to Watty MCP server")
                    return True
        except Exception as e:
            logger.warning(f"Failed to connect to Watty: {e}")
            self._connected = False

        return False

    async def disconnect(self):
        """Close the connection."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def _call_tool(self, tool_name: str, params: dict) -> dict | None:
        """
        Invoke an MCP tool on the Watty server.

        Returns None if the server is unreachable (graceful degradation).
        """
        if not self._connected or not self._session:
            logger.debug(f"Watty offline, skipping {tool_name}")
            return None

        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": params},
                "id": 1,
            }
            async with self._session.post(
                f"{self.server_url}/mcp", json=payload
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("result")
                else:
                    logger.warning(f"Watty tool {tool_name} returned {resp.status}")
                    return None
        except Exception as e:
            logger.warning(f"Watty tool call failed: {e}")
            self._connected = False
            return None

    async def vault_query(self, query: str, limit: int = 5) -> list[str]:
        """Semantic search over facts and beliefs."""
        result = await self._call_tool("vault_query", {
            "query": query, "limit": limit
        })
        if result and isinstance(result, dict):
            return result.get("results", [])
        return []

    async def vault_deposit(self, content: str, metadata: dict | None = None) -> bool:
        """Store new knowledge."""
        result = await self._call_tool("vault_deposit", {
            "content": content, "metadata": metadata or {}
        })
        return result is not None

    async def metabolize(self, interaction: str) -> dict | None:
        """Extract and deposit knowledge from an interaction."""
        return await self._call_tool("metabolize", {"interaction": interaction})

    async def graph_query(
        self, start_node: str, depth: int = 2, limit: int = 10
    ) -> list[dict]:
        """Spreading activation search over knowledge graph."""
        result = await self._call_tool("graph_query", {
            "start_node": start_node, "depth": depth, "limit": limit
        })
        if result and isinstance(result, dict):
            return result.get("nodes", [])
        return []

    async def graph_build(self, source: str, target: str, relation: str) -> bool:
        """Build edges between knowledge nodes."""
        result = await self._call_tool("graph_build", {
            "source": source, "target": target, "relation": relation
        })
        return result is not None

    async def fed_inject(self, belief: str) -> bool:
        """Force a specific belief into the context."""
        result = await self._call_tool("fed_inject", {"belief": belief})
        return result is not None
