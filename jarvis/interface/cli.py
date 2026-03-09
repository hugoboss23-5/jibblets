"""
CLI interface — conversational terminal for JARVIS.

User talks to Jarvis. Jarvis routes internally, self-modifies, responds.
The interface provides real-time visibility into the router's internal state:
routing decisions, face activations, coherence scores, and self-modification.
"""

import asyncio
import hashlib
import logging
import sys

import torch

from ..core.router import SelfModifyingRouter, RouterConfig, RoutingDecision
from ..core.chestahedron import FACE_NAMES
from ..organs.claude_organ import ClaudeOrgan
from ..spine.watty_client import WattyClient
from ..spine.memory import MemoryManager

logger = logging.getLogger(__name__)


def _simple_tokenize(text: str, vocab_size: int = 30000, max_len: int = 128) -> torch.Tensor:
    """
    Simple hash-based tokenization. Not meant to be good — the intelligence
    comes from the organs. This just gives the router something to work with.
    """
    words = text.lower().split()
    tokens = []
    for w in words[:max_len]:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16) % vocab_size
        tokens.append(h)
    # Pad to max_len
    while len(tokens) < max_len:
        tokens.append(0)
    return torch.tensor([tokens[:max_len]], dtype=torch.long)


class JarvisCLI:
    """
    Conversational terminal interface to JARVIS.
    """

    def __init__(
        self,
        config: RouterConfig | None = None,
        api_key: str | None = None,
        watty_url: str | None = None,
        checkpoint_dir: str = "./jarvis_state",
        verbose: bool = False,
    ):
        self.config = config or RouterConfig()
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose

        # Initialize core
        self.router = SelfModifyingRouter(self.config)
        self.router.init_safety(checkpoint_dir)

        # Simple text encoder (hash-based, intentionally dumb)
        from ..core.router import SimpleTextEncoder
        self.encoder = SimpleTextEncoder(
            vocab_size=30000, embed_dim=self.config.input_dim
        )

        # Initialize organs
        self.claude = ClaudeOrgan(api_key=api_key)

        # Initialize spine
        self.watty = WattyClient(server_url=watty_url)
        self.memory = MemoryManager(self.watty)

        # Conversation state
        self._history: list[dict] = []
        self._last_quality: float | None = None

    def _encode_query(self, text: str) -> torch.Tensor:
        """Convert text to input embedding for the router."""
        tokens = _simple_tokenize(text)
        with torch.no_grad():
            embedding = self.encoder(tokens)
        return embedding

    def _format_face_bar(self, activations: torch.Tensor) -> str:
        """Format face activations as a visual bar chart."""
        lines = []
        for i, name in enumerate(FACE_NAMES):
            w = activations[i].item()
            bar = "█" * int(w * 30)
            lines.append(f"  {name:12s} {bar:<30s} {w:.3f}")
        return "\n".join(lines)

    def _format_decision(self, decision: RoutingDecision) -> str:
        """Format routing decision for display."""
        parts = [
            f"  Route: {decision.route} (confidence: {decision.route_probs.max().item():.3f})",
            f"  Coherence: {decision.shaped_context['coherence']:.3f}",
            f"  Dominant: {decision.shaped_context['dominant_operation']}",
        ]
        if self.verbose and decision.meta.get("loss"):
            loss_info = decision.meta["loss"]
            parts.append(f"  Loss: {loss_info.get('total_loss', 0):.4f}")
            parts.append(f"  Step: {decision.meta.get('step', 0)}")
        return "\n".join(parts)

    async def _execute_route(
        self, decision: RoutingDecision, query: str
    ) -> str:
        """Execute the routing decision and get a response."""
        route = decision.route

        if route == "LOCAL":
            # Router answers locally (badly, but answers)
            return (
                "[Local response — no API call]\n"
                "I can try to help based on routing context, but for a detailed "
                "answer I'd need to call an API organ."
            )

        # Check if we need memory
        memories = []
        if route in ("MEMORY", "MEMORY_THEN_API"):
            memories = await self.memory.retrieve(
                query,
                operation_weights=decision.shaped_context.get("operation_weights"),
            )

        if route == "MEMORY":
            if memories:
                return "[From memory]\n" + "\n".join(f"- {m}" for m in memories)
            return "[No relevant memories found. Consider using API route.]"

        # API routes
        model = None
        if route == "API_OPUS":
            model = "claude-opus-4-20250514"

        response = await self.claude.call(
            shaped_context=decision.shaped_context,
            user_query=query,
            memories=memories if memories else None,
            model_override=model,
        )

        # Feed quality back
        self._last_quality = response.quality_estimate

        # Store interaction in memory if quality is good
        if response.quality_estimate > 0.5:
            await self.memory.metabolize_interaction(
                query, response.content, response.quality_estimate
            )

        model_tag = f"[{response.model}]"
        return f"{model_tag}\n{response.content}"

    async def process_query(self, query: str) -> str:
        """
        Process a user query through the full JARVIS pipeline.

        1. Encode query
        2. Route through self-modifying core
        3. Execute route (local/memory/API)
        4. Return response
        """
        # Encode
        embedding = self._encode_query(query)

        # Route (with self-modification)
        decision = self.router(
            embedding,
            quality_signal=self._last_quality,
            self_modify=True,
        )

        # Show routing info
        print(f"\n--- Routing ---")
        print(self._format_decision(decision))
        if self.verbose:
            print(f"\n--- Face Activations ---")
            print(self._format_face_bar(decision.face_activations))
        print(f"---\n")

        # Execute
        response = await self._execute_route(decision, query)

        # Store in history
        self._history.append({
            "query": query,
            "route": decision.route,
            "coherence": decision.shaped_context["coherence"],
        })

        return response

    async def run(self):
        """Main conversation loop."""
        print("=" * 60)
        print("  JARVIS — Self-Modifying Cognitive Router")
        print("  Type 'quit' to exit, 'status' for router state")
        print("  Type 'verbose' to toggle detailed output")
        print("=" * 60)

        # Try connecting to Watty
        watty_ok = await self.watty.connect()
        if watty_ok:
            print("  Spine (Watty): CONNECTED")
        else:
            print("  Spine (Watty): OFFLINE (graceful degradation)")

        print()

        while True:
            try:
                query = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nShutting down...")
                break

            if not query:
                continue

            if query.lower() == "quit":
                break

            if query.lower() == "verbose":
                self.verbose = not self.verbose
                print(f"Verbose mode: {'ON' if self.verbose else 'OFF'}")
                continue

            if query.lower() == "status":
                self._show_status()
                continue

            if query.lower() == "save":
                self.router.save_state(self.checkpoint_dir)
                print("State saved.")
                continue

            try:
                response = await self.process_query(query)
                print(f"\nJarvis: {response}\n")
            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                print(f"\n[Error: {e}]\n")

        # Save state on exit
        self.router.save_state(self.checkpoint_dir)
        await self.watty.disconnect()
        print("State saved. Goodbye.")

    def _show_status(self):
        """Show router internal status."""
        history = self.router.get_modification_history()
        print(f"\n--- Router Status ---")
        print(f"  Steps: {self.router._step_count}")
        print(f"  Modifications logged: {len(history)}")
        if history:
            recent = history[-5:]
            print(f"  Recent routes: {[h['route'] for h in recent]}")
            coherences = [h['coherence'] for h in recent]
            print(f"  Recent coherence: {[f'{c:.3f}' for c in coherences]}")
            if len(history) > 5:
                early = [h['coherence'] for h in history[:5]]
                late = [h['coherence'] for h in history[-5:]]
                print(f"  Coherence trend: {sum(early)/len(early):.3f} -> {sum(late)/len(late):.3f}")
        print(f"  Memory: {'ONLINE' if self.memory.available else 'OFFLINE'}")
        print(f"---\n")
