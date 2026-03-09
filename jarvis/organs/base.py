"""
Base organ class — interface for all frontier model connections.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OrganResponse:
    """Response from an organ call."""
    content: str
    model: str
    usage: dict
    quality_estimate: float  # Self-estimated quality [0, 1]
    raw_response: object = None


class BaseOrgan(ABC):
    """Base class for all organ interfaces."""

    @abstractmethod
    async def call(self, shaped_context: dict, user_query: str) -> OrganResponse:
        """
        Call the organ with shaped context from the router.

        Args:
            shaped_context: context package from the router containing:
                - context_vector: tensor embedding
                - operation_weights: dict of face->weight
                - dominant_operation: primary cognitive operation
                - coherence: geometric coherence score
            user_query: the original user query

        Returns:
            OrganResponse with content and quality estimate
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the organ is reachable."""
        ...
