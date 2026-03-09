"""
Memory retrieval and storage logic.

Higher-level memory operations that the router uses. Wraps the raw Watty MCP
client with routing-aware logic: the Core decides WHEN to pull memory and
WHAT to store, based on the current Chestahedron state.
"""

import logging

from .watty_client import WattyClient

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages memory operations for the router.

    The router calls into this based on its routing decisions. Not every
    query needs memory — the router learns when memory retrieval helps
    and when it's noise.
    """

    def __init__(self, watty: WattyClient):
        self.watty = watty
        self._cache: dict[str, list[str]] = {}  # Simple query cache
        self._cache_max = 100

    @property
    def available(self) -> bool:
        """Whether the memory system is online."""
        return self.watty.is_connected

    async def retrieve(
        self,
        query: str,
        operation_weights: dict[str, float] | None = None,
        limit: int = 5,
    ) -> list[str]:
        """
        Retrieve relevant memories for a query.

        If operation weights indicate LOAD is dominant, does deeper retrieval
        including graph expansion. Otherwise, just does vault query.

        Args:
            query: the search query
            operation_weights: Chestahedron face weights (optional)
            limit: max results

        Returns:
            list of relevant memory strings
        """
        if not self.available:
            return []

        # Check cache
        if query in self._cache:
            return self._cache[query]

        results = []

        # Basic vault query
        vault_results = await self.watty.vault_query(query, limit=limit)
        results.extend(vault_results)

        # If LOAD operation is strong, also do graph expansion
        load_weight = (operation_weights or {}).get("LOAD", 0.0)
        if load_weight > 0.3 and vault_results:
            # Use first result as graph seed
            graph_nodes = await self.watty.graph_query(
                start_node=vault_results[0], depth=2, limit=limit
            )
            for node in graph_nodes:
                content = node.get("content", str(node))
                if content not in results:
                    results.append(content)

        # Cache results
        if len(self._cache) >= self._cache_max:
            # Evict oldest
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[query] = results

        return results

    async def store(self, content: str, metadata: dict | None = None) -> bool:
        """Store new knowledge in the vault."""
        if not self.available:
            return False
        return await self.watty.vault_deposit(content, metadata)

    async def metabolize_interaction(
        self, user_query: str, response: str, quality: float
    ) -> bool:
        """
        Extract and store knowledge from a completed interaction.
        Only metabolizes interactions above a quality threshold.
        """
        if not self.available:
            return False

        if quality < 0.3:
            logger.debug("Skipping metabolize for low-quality interaction")
            return False

        interaction = f"User: {user_query}\nResponse: {response}"
        result = await self.watty.metabolize(interaction)
        return result is not None

    async def build_connection(
        self, source: str, target: str, relation: str
    ) -> bool:
        """Build a knowledge graph edge."""
        if not self.available:
            return False
        return await self.watty.graph_build(source, target, relation)

    def clear_cache(self):
        """Clear the memory cache."""
        self._cache.clear()
