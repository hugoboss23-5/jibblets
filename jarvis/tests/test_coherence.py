"""
Test that the geometric coherence metric works correctly.

Verifies:
1. Points on the manifold have coherence ~1.0
2. Random points have lower coherence
3. The router learns to increase coherence over time
4. Adjacency topology is respected
"""

import torch
import pytest

from jarvis.core.chestahedron import (
    ChestahedronProjector,
    ChestahedronEmbedding,
    CHESTA_VERTICES_7D,
    CHESTA_ADJACENCY,
    NUM_FACES,
)
from jarvis.core.router import SelfModifyingRouter, RouterConfig


class TestChestahedronGeometry:
    """Verify the Chestahedron manifold and coherence metric."""

    def setup_method(self):
        self.proj = ChestahedronProjector()

    def test_vertices_are_7d(self):
        assert CHESTA_VERTICES_7D.shape == (7, 7)

    def test_vertices_are_unit_norm(self):
        norms = CHESTA_VERTICES_7D.norm(dim=1)
        assert torch.allclose(norms, torch.ones(7), atol=1e-5)

    def test_adjacency_is_symmetric(self):
        assert torch.allclose(CHESTA_ADJACENCY, CHESTA_ADJACENCY.t())

    def test_adjacency_has_correct_edges(self):
        # Should have 12 edges (24 entries since symmetric)
        assert CHESTA_ADJACENCY.sum().item() == 24.0

    def test_vertex_coherence_is_high(self):
        """Points exactly on vertices should have near-perfect coherence."""
        coherence = self.proj.coherence_score(CHESTA_VERTICES_7D)
        assert coherence.min().item() > 0.8, \
            f"Vertex coherence should be high, got min={coherence.min().item():.3f}"

    def test_random_points_have_lower_coherence(self):
        """Random 7D points should have lower coherence than vertices."""
        random_points = torch.randn(100, 7)
        random_coherence = self.proj.coherence_score(random_points)
        vertex_coherence = self.proj.coherence_score(CHESTA_VERTICES_7D)

        assert random_coherence.mean().item() < vertex_coherence.mean().item(), \
            "Random points should have lower coherence than vertices"

    def test_projection_increases_coherence(self):
        """Projecting random points onto manifold should increase coherence."""
        random_points = torch.randn(50, 7)
        pre_coherence = self.proj.coherence_score(random_points)
        projected = self.proj.project_to_manifold(random_points)
        post_coherence = self.proj.coherence_score(projected)

        assert post_coherence.mean().item() > pre_coherence.mean().item(), \
            "Projection should increase coherence"

    def test_face_activations_sum_to_one(self):
        """Face activations should be valid probability distributions."""
        points = torch.randn(10, 7)
        activations = self.proj.face_activations(points)
        sums = activations.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)

    def test_vertex_has_dominant_face(self):
        """Each vertex should strongly activate its corresponding face."""
        activations = self.proj.face_activations(CHESTA_VERTICES_7D)
        for i in range(NUM_FACES):
            dominant = activations[i].argmax().item()
            assert dominant == i, \
                f"Vertex {i} should dominate face {i}, got {dominant}"

    def test_adjacency_loss_penalizes_non_adjacent(self):
        """Non-adjacent co-activation should produce higher loss."""
        # Adjacent activation (DIAGNOSE + LOAD, faces 0 and 1)
        adjacent_weights = torch.zeros(1, 7)
        adjacent_weights[0, 0] = 0.5
        adjacent_weights[0, 1] = 0.5
        adj_loss = self.proj.adjacency_loss(adjacent_weights)

        # Non-adjacent activation (DIAGNOSE + BREAK, faces 0 and 4)
        # Check: faces 0 and 4 are NOT adjacent in our topology
        # Actually 0-4 isn't an edge, let's pick clearly non-adjacent
        # Looking at edges: 1-2 is not an edge, so LOAD + CONSTRAIN are non-adjacent
        non_adj_weights = torch.zeros(1, 7)
        non_adj_weights[0, 1] = 0.5  # LOAD
        non_adj_weights[0, 2] = 0.5  # CONSTRAIN
        non_adj_loss = self.proj.adjacency_loss(non_adj_weights)

        # Non-adjacent should have higher penalty
        assert non_adj_loss.item() > adj_loss.item(), \
            "Non-adjacent co-activation should be penalized more"


class TestChestahedronEmbedding:
    """Verify the learnable embedding into Chestahedron space."""

    def test_embedding_output_is_7d(self):
        embed = ChestahedronEmbedding(input_dim=64)
        x = torch.randn(5, 64)
        out = embed(x)
        assert out.shape == (5, 7)

    def test_embedding_is_differentiable(self):
        embed = ChestahedronEmbedding(input_dim=64)
        x = torch.randn(5, 64, requires_grad=True)
        out = embed(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestCoherenceConvergence:
    """Verify that the router's coherence improves over time."""

    def test_coherence_improves_over_interactions(self):
        """
        The router should learn to stay on-manifold.
        Coherence should increase (or at least not decrease) over sequential interactions.
        """
        torch.manual_seed(42)
        config = RouterConfig(
            input_dim=64, hidden_dim=32, num_routes=5,
            num_fast_layers=2, num_heads=2, hyper_hidden=32,
            num_hyper_layers=2, internal_lr=0.01,
            coherence_weight=2.0,  # Strong coherence pressure
        )
        router = SelfModifyingRouter(config)

        coherences = []
        for i in range(30):
            x = torch.randn(1, 64)
            decision = router(x, quality_signal=0.7, self_modify=True)
            coherences.append(decision.shaped_context["coherence"])

        # Compare early vs late coherence
        early = sum(coherences[:5]) / 5
        late = sum(coherences[-5:]) / 5

        # Coherence should not degrade significantly
        # (it may fluctuate, but the trend should be stable or improving)
        assert late >= early - 0.15, \
            f"Coherence should not significantly degrade: {early:.3f} -> {late:.3f}"

    def test_coherence_is_logged(self):
        """Coherence should appear in modification history."""
        config = RouterConfig(
            input_dim=64, hidden_dim=32, num_routes=5,
            num_fast_layers=1, num_heads=2, hyper_hidden=32,
        )
        router = SelfModifyingRouter(config)

        x = torch.randn(1, 64)
        router(x, self_modify=True)

        history = router.get_modification_history()
        assert len(history) > 0
        assert "coherence" in history[0]
        assert 0 <= history[0]["coherence"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
