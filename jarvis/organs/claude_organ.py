"""
Claude API organ — frontier intelligence with context shaping.

The Core doesn't just forward the user's query to Claude. It constructs a
geometrically-shaped context package with:
- Relevant memories from Watty (selected by geometric coherence)
- Topology protocol header (Chestahedron prompt scaffold)
- Weighted operation vector (which of the 7 operations this call needs)
- Constraint set (what the organ should NOT do)
"""

import logging
from typing import Any

import anthropic

from .base import BaseOrgan, OrganResponse
from ..core.chestahedron import FACE_NAMES

logger = logging.getLogger(__name__)

# V9 Topology Protocol scaffold — the geometric prompt structure
TOPOLOGY_PROTOCOL = """You are operating within a geometric cognitive topology.
The current cognitive state is weighted across 7 operations:

{operation_weights}

Dominant operation: {dominant_operation}
Geometric coherence: {coherence:.3f}

{constraints}

Respond within these operational constraints. {instruction}"""


class ClaudeOrgan(BaseOrgan):
    """
    Claude API interface with geometric context shaping.

    The router decides WHEN to call Claude, WHAT context to send,
    and HOW to shape the prompt geometry before sending.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.default_model = default_model
        self.max_tokens = max_tokens

    def _build_operation_weights_str(self, operation_weights: dict) -> str:
        """Format operation weights for the prompt."""
        lines = []
        for face_name in FACE_NAMES:
            weight = operation_weights.get(face_name, 0.0)
            bar = "█" * int(weight * 20)
            lines.append(f"  {face_name:12s} [{bar:<20s}] {weight:.3f}")
        return "\n".join(lines)

    def _build_constraints(self, shaped_context: dict) -> str:
        """Build constraint set based on dominant operation."""
        dominant = shaped_context.get("dominant_operation", "ROUTE")
        constraints = {
            "DIAGNOSE": "Focus on analysis and understanding. Do not generate solutions yet.",
            "LOAD": "Integrate the provided context deeply. Reference specific details.",
            "CONSTRAIN": "Be precise about boundaries. State what is and isn't in scope.",
            "GENERATE": "Produce multiple candidate approaches. Prioritize breadth.",
            "BREAK": "Stress-test the approach. Find failure modes and edge cases.",
            "CUT": "Be concise. Select the best option and compress the response.",
            "ROUTE": "Provide clear, actionable output. Focus on the specific request.",
        }
        return f"Constraint: {constraints.get(dominant, '')}"

    def _build_instruction(self, shaped_context: dict) -> str:
        """Build dynamic instruction based on operation weights."""
        weights = shaped_context.get("operation_weights", {})
        # High DIAGNOSE + LOAD = analysis mode
        if weights.get("DIAGNOSE", 0) > 0.2 and weights.get("LOAD", 0) > 0.2:
            return "Provide thorough analysis with context integration."
        # High GENERATE + BREAK = exploration mode
        if weights.get("GENERATE", 0) > 0.2 and weights.get("BREAK", 0) > 0.2:
            return "Generate candidates and critically evaluate each."
        # High CUT + ROUTE = execution mode
        if weights.get("CUT", 0) > 0.2 and weights.get("ROUTE", 0) > 0.2:
            return "Provide a direct, concise answer."
        return ""

    def _shape_prompt(
        self, shaped_context: dict, user_query: str, memories: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Construct the geometrically-shaped prompt for Claude.

        This is where topology beats raw intelligence — the router shapes
        the context geometry to guide Claude's response.
        """
        op_weights_str = self._build_operation_weights_str(
            shaped_context.get("operation_weights", {})
        )
        constraints = self._build_constraints(shaped_context)
        instruction = self._build_instruction(shaped_context)

        system_prompt = TOPOLOGY_PROTOCOL.format(
            operation_weights=op_weights_str,
            dominant_operation=shaped_context.get("dominant_operation", "ROUTE"),
            coherence=shaped_context.get("coherence", 0.0),
            constraints=constraints,
            instruction=instruction,
        )

        messages = []

        # Inject memories as assistant context if available
        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            messages.append({
                "role": "user",
                "content": f"[Relevant context from memory]\n{memory_text}",
            })
            messages.append({
                "role": "assistant",
                "content": "I've noted the relevant context. How can I help?",
            })

        # User query
        messages.append({"role": "user", "content": user_query})

        return system_prompt, messages

    def _estimate_quality(self, response: anthropic.types.Message) -> float:
        """
        Estimate response quality based on heuristics.
        Real quality comes from user feedback; this is a prior.
        """
        content = response.content[0].text if response.content else ""
        score = 0.5  # Base

        # Length heuristic (not too short, not too long)
        length = len(content)
        if 100 < length < 5000:
            score += 0.1
        elif length < 20:
            score -= 0.2

        # Stop reason
        if response.stop_reason == "end_turn":
            score += 0.1
        elif response.stop_reason == "max_tokens":
            score -= 0.1

        return max(0.0, min(1.0, score))

    async def call(
        self,
        shaped_context: dict,
        user_query: str,
        memories: list[str] | None = None,
        model_override: str | None = None,
    ) -> OrganResponse:
        """
        Call Claude with geometrically-shaped context.

        Args:
            shaped_context: context package from the router
            user_query: the original user query
            memories: optional list of relevant memories from Watty
            model_override: override the default model (e.g. for opus calls)

        Returns:
            OrganResponse
        """
        model = model_override or self.default_model
        system_prompt, messages = self._shape_prompt(
            shaped_context, user_query, memories
        )

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=self.max_tokens,
                system=system_prompt,
                messages=messages,
            )

            content = response.content[0].text if response.content else ""
            quality = self._estimate_quality(response)

            return OrganResponse(
                content=content,
                model=model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                quality_estimate=quality,
                raw_response=response,
            )

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            return OrganResponse(
                content=f"[API Error: {e}]",
                model=model,
                usage={},
                quality_estimate=0.0,
            )

    async def health_check(self) -> bool:
        """Check if Claude API is reachable."""
        try:
            response = self.client.messages.create(
                model=self.default_model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return bool(response.content)
        except Exception:
            return False
