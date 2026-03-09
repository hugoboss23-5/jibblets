"""
Internal loss computation for the self-modifying router.

The loss is computed during the forward pass (not as a separate training step)
and drives the self-modification. Three components:

1. Routing accuracy — did the right organ/action get selected?
2. Response quality — user feedback / coherence of the result
3. Geometric coherence — alignment with the Chestahedron manifold
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chestahedron import ChestahedronProjector, NUM_FACES


class RoutingLoss(nn.Module):
    """
    Computes routing accuracy loss based on whether the selected route
    produced a good outcome.

    Uses a running estimate of route quality (exponential moving average)
    to compute a self-supervised loss without requiring external labels.
    """

    def __init__(self, num_routes: int, ema_decay: float = 0.9):
        super().__init__()
        self.num_routes = num_routes
        self.ema_decay = ema_decay

        # Running quality estimates per route
        self.register_buffer(
            "route_quality", torch.zeros(num_routes)
        )
        self.register_buffer(
            "route_counts", torch.zeros(num_routes)
        )

    def update_quality(self, route_idx: int, quality: float):
        """Update the quality estimate for a route based on feedback."""
        self.route_counts[route_idx] += 1
        self.route_quality[route_idx] = (
            self.ema_decay * self.route_quality[route_idx]
            + (1 - self.ema_decay) * quality
        )

    def forward(
        self,
        route_logits: torch.Tensor,
        selected_route: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute routing loss that encourages selecting high-quality routes.

        Args:
            route_logits: (batch, num_routes) unnormalized route scores
            selected_route: (batch,) previously selected route indices, or None

        Returns:
            scalar loss
        """
        route_probs = F.softmax(route_logits, dim=-1)

        # Loss = negative expected quality under current routing policy
        # Encourages putting probability mass on high-quality routes
        quality = self.route_quality.detach()
        if quality.sum() == 0:
            # No feedback yet — return entropy loss to encourage exploration
            entropy = -(route_probs * (route_probs + 1e-8).log()).sum(dim=-1)
            return -entropy.mean() * 0.1  # Mild exploration bonus

        # Normalize quality to [0, 1]
        q_min = quality.min()
        q_max = quality.max()
        if q_max > q_min:
            quality_norm = (quality - q_min) / (q_max - q_min)
        else:
            quality_norm = torch.ones_like(quality) * 0.5

        expected_quality = (route_probs * quality_norm.unsqueeze(0)).sum(dim=-1)
        return -expected_quality.mean()


class CoherenceLoss(nn.Module):
    """
    Penalizes router states that deviate from the Chestahedron manifold.
    Also penalizes topologically invalid face co-activation patterns.
    """

    def __init__(self, coherence_weight: float = 1.0, adjacency_weight: float = 0.5):
        super().__init__()
        self.projector = ChestahedronProjector()
        self.coherence_weight = coherence_weight
        self.adjacency_weight = adjacency_weight

    def forward(self, state_7d: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Compute geometric coherence loss.

        Args:
            state_7d: (batch, 7) router state in Chestahedron space

        Returns:
            loss: scalar coherence loss
            info: dict with coherence metrics
        """
        # Distance from manifold
        coherence = self.projector.coherence_score(state_7d)
        manifold_loss = (1.0 - coherence).mean()

        # Adjacency topology violation
        face_weights = self.projector.face_activations(state_7d)
        adj_loss = self.projector.adjacency_loss(face_weights)

        total = (
            self.coherence_weight * manifold_loss
            + self.adjacency_weight * adj_loss
        )

        info = {
            "coherence_mean": coherence.mean().item(),
            "coherence_min": coherence.min().item(),
            "manifold_loss": manifold_loss.item(),
            "adjacency_loss": adj_loss.item(),
        }

        return total, info


class InternalLoss(nn.Module):
    """
    Combined internal loss for the self-modifying router.
    Computed during forward pass to drive real-time self-modification.
    """

    def __init__(
        self,
        num_routes: int,
        routing_weight: float = 1.0,
        coherence_weight: float = 1.0,
        adjacency_weight: float = 0.5,
        quality_weight: float = 1.0,
    ):
        super().__init__()
        self.routing_loss = RoutingLoss(num_routes)
        self.coherence_loss = CoherenceLoss(coherence_weight, adjacency_weight)
        self.routing_weight = routing_weight
        self.quality_weight = quality_weight

    def forward(
        self,
        route_logits: torch.Tensor,
        state_7d: torch.Tensor,
        quality_signal: float | None = None,
        selected_route: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute the full internal loss.

        Args:
            route_logits: (batch, num_routes) routing logits
            state_7d: (batch, 7) Chestahedron state
            quality_signal: optional quality feedback from last interaction
            selected_route: optional index of last selected route

        Returns:
            total_loss: scalar
            info: dict with all loss components
        """
        # Update quality estimates if feedback is available
        if quality_signal is not None and selected_route is not None:
            self.routing_loss.update_quality(selected_route, quality_signal)

        # Compute components
        r_loss = self.routing_loss(route_logits)
        c_loss, c_info = self.coherence_loss(state_7d)

        total = self.routing_weight * r_loss + self.quality_weight * c_loss

        info = {
            "total_loss": total.item(),
            "routing_loss": r_loss.item(),
            **c_info,
        }

        return total, info
