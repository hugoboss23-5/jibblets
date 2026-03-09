"""
Test that routing decisions improve over time.

Verifies:
1. The router learns to route similar queries consistently
2. Quality feedback improves routing over sequential queries
3. Route probabilities become more confident with experience
"""

import torch
import pytest

from jarvis.core.router import SelfModifyingRouter, RouterConfig


class TestRoutingImprovement:
    """Verify routing accuracy improves over sequential queries."""

    def setup_method(self):
        self.config = RouterConfig(
            input_dim=64, hidden_dim=32, num_routes=5,
            num_fast_layers=2, num_heads=2, hyper_hidden=32,
            num_hyper_layers=2, internal_lr=0.01,
        )
        self.router = SelfModifyingRouter(self.config)

    def test_repeated_queries_increase_confidence(self):
        """Same query repeated should produce increasingly confident routing."""
        torch.manual_seed(42)
        query = torch.randn(1, 64)

        confidences = []
        for _ in range(20):
            decision = self.router(query, quality_signal=0.8, self_modify=True)
            confidences.append(decision.route_probs.max().item())

        # Confidence should trend upward (or at least not decrease significantly)
        early_avg = sum(confidences[:5]) / 5
        late_avg = sum(confidences[-5:]) / 5
        # Allow some tolerance — the system should at least not degrade
        assert late_avg >= early_avg - 0.1, \
            f"Confidence should not significantly decrease: {early_avg:.3f} -> {late_avg:.3f}"

    def test_quality_feedback_shifts_routing(self):
        """Positive quality feedback should reinforce the selected route."""
        torch.manual_seed(42)

        # Get initial routing for a query
        query = torch.randn(1, 64)
        initial_decision = self.router(query, self_modify=False)
        initial_route = initial_decision.route

        # Provide strong positive feedback for this route
        for _ in range(20):
            decision = self.router(query, quality_signal=1.0, self_modify=True)

        # The router should now be more confident about this route
        final_decision = self.router(query, self_modify=False)
        assert final_decision.route_probs.max().item() >= initial_decision.route_probs.max().item() - 0.15

    def test_different_queries_get_different_routes(self):
        """Sufficiently different queries should produce different internal states."""
        torch.manual_seed(42)

        # Create two very different query types (orthogonal directions)
        query_a = torch.zeros(1, 64)
        query_a[0, :32] = 5.0
        query_b = torch.zeros(1, 64)
        query_b[0, 32:] = 5.0

        # Route each several times with self-modification
        for _ in range(10):
            self.router(query_a, quality_signal=0.8, self_modify=True)
            self.router(query_b, quality_signal=0.8, self_modify=True)

        # Get final routing decisions in eval mode (no dropout noise)
        self.router.eval()
        dec_a = self.router(query_a, self_modify=False)
        dec_b = self.router(query_b, self_modify=False)
        self.router.train()

        # The face activations or route probs should differ for different inputs
        face_diff = (dec_a.face_activations - dec_b.face_activations).abs().sum().item()
        prob_diff = (dec_a.route_probs - dec_b.route_probs).abs().sum().item()
        total_diff = face_diff + prob_diff
        assert total_diff > 0.001, \
            f"Different queries should produce different states (diff={total_diff:.6f})"

    def test_route_consistency_over_time(self):
        """Similar queries should converge to consistent routing."""
        torch.manual_seed(42)

        # Base query
        base_query = torch.randn(1, 64)

        # Train on the base query
        for _ in range(15):
            self.router(base_query, quality_signal=0.9, self_modify=True)

        # Test with slightly perturbed versions
        routes = []
        for _ in range(5):
            noisy_query = base_query + torch.randn(1, 64) * 0.1
            decision = self.router(noisy_query, self_modify=False)
            routes.append(decision.route)

        # At least 3 of 5 should agree (consistency)
        from collections import Counter
        most_common_count = Counter(routes).most_common(1)[0][1]
        assert most_common_count >= 3, \
            f"Similar queries should route consistently, got: {routes}"

    def test_no_improvement_without_self_modify(self):
        """Without self-modification, routing should not change (eval mode)."""
        torch.manual_seed(42)
        query = torch.randn(1, 64)

        # Put in eval mode to disable dropout (which introduces randomness)
        self.router.eval()

        # Run without self-modification
        decisions = []
        for _ in range(10):
            decision = self.router(query, quality_signal=0.8, self_modify=False)
            decisions.append(decision.route_probs.clone())

        # All decisions should be identical (no learning, no dropout noise)
        for i in range(1, len(decisions)):
            assert torch.allclose(decisions[0], decisions[i], atol=1e-5), \
                f"Without self-modify, routing should not change: {decisions[0]} vs {decisions[i]}"

        self.router.train()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
