"""
Test that self-modification actually happens during forward pass.

Verifies:
1. Fast weights change after forward pass
2. Modifiable parameters update while frozen ones don't
3. Weight modifications are logged and reversible
4. Norm constraints are respected
"""

import torch
import pytest

from jarvis.core.router import SelfModifyingRouter, RouterConfig
from jarvis.core.fast_weights import FastWeightMemory, FastWeightLayer
from jarvis.core.safety import WeightSafetyGuard


class TestFastWeightModification:
    """Verify fast weights actually change during forward pass."""

    def setup_method(self):
        self.dim = 64
        self.fw = FastWeightMemory(self.dim, num_heads=4)

    def test_fast_weights_start_at_zero(self):
        assert self.fw.fast_weights.norm().item() == 0.0

    def test_fast_weights_change_after_forward(self):
        x = torch.randn(1, self.dim)
        pre_norm = self.fw.fast_weights.norm().item()
        _, info = self.fw(x, update=True)
        post_norm = self.fw.fast_weights.norm().item()

        assert info["updated"] is True
        assert post_norm > pre_norm, "Fast weights should change after forward pass"

    def test_fast_weights_unchanged_without_update(self):
        x = torch.randn(1, self.dim)
        # First do an update to have non-zero weights
        self.fw(x, update=True)
        pre_state = self.fw.fast_weights.clone()

        # Forward without update
        _, info = self.fw(x, update=False)
        assert info["updated"] is False
        assert torch.allclose(self.fw.fast_weights, pre_state)

    def test_fast_weights_accumulate_over_passes(self):
        norms = []
        for i in range(10):
            x = torch.randn(1, self.dim)
            self.fw(x, update=True)
            norms.append(self.fw.fast_weights.norm().item())

        # Norms should generally increase (with decay preventing explosion)
        assert norms[-1] > norms[0], "Fast weights should accumulate information"

    def test_fast_weight_norm_bounded(self):
        """Fast weights should not explode — per-head norms stay bounded."""
        for _ in range(100):
            x = torch.randn(1, self.dim) * 10  # Large inputs
            self.fw(x, update=True)

        # max_norm is applied per-head, so check per-head norms
        per_head_norms = self.fw.fast_weights.norm(dim=(-2, -1))
        assert per_head_norms.max().item() <= self.fw.max_norm * 1.1, \
            f"Per-head norm {per_head_norms.max().item():.2f} exceeds max_norm {self.fw.max_norm}"

    def test_revert_last_update(self):
        x = torch.randn(1, self.dim)
        self.fw(x, update=True)
        state_after_first = self.fw.fast_weights.clone()

        self.fw(torch.randn(1, self.dim), update=True)
        assert not torch.allclose(self.fw.fast_weights, state_after_first)

        # Revert
        assert self.fw.revert_last_update() is True
        assert torch.allclose(self.fw.fast_weights, state_after_first)


class TestRouterSelfModification:
    """Verify the full router self-modifies during forward pass."""

    def setup_method(self):
        self.config = RouterConfig(
            input_dim=64, hidden_dim=32, num_routes=5,
            num_fast_layers=1, num_heads=2, hyper_hidden=32,
            num_hyper_layers=2,
        )
        self.router = SelfModifyingRouter(self.config)

    def test_router_produces_routing_decision(self):
        x = torch.randn(1, 64)
        decision = self.router(x, self_modify=False)
        assert decision.route in decision.ROUTES
        assert decision.route_probs.shape == (5,)
        assert decision.face_activations.shape == (7,)

    def test_router_modifies_weights_during_forward(self):
        x = torch.randn(1, 64)

        # Snapshot modifiable params before
        pre_states = {}
        for name, param in self.router.named_parameters():
            if self.router._is_modifiable(name):
                pre_states[name] = param.data.clone()

        # Forward with self-modification
        self.router(x, self_modify=True)

        # Check that at least some modifiable params changed
        changed = 0
        for name, param in self.router.named_parameters():
            if name in pre_states:
                if not torch.allclose(param.data, pre_states[name], atol=1e-7):
                    changed += 1

        assert changed > 0, "At least some modifiable parameters should change"

    def test_frozen_params_dont_change(self):
        """Input encoder should be frozen after init_safety."""
        x = torch.randn(1, 64)

        self.router.init_safety()
        # Reset internal optimizer so it picks up the new requires_grad state
        self.router._internal_optimizer = None

        # Snapshot frozen params (those with requires_grad=False after init_safety)
        pre_states = {}
        for name, param in self.router.named_parameters():
            if not param.requires_grad:
                pre_states[name] = param.data.clone()

        self.router(x, self_modify=True)

        for name, param in self.router.named_parameters():
            if name in pre_states:
                assert torch.allclose(param.data, pre_states[name]), \
                    f"Frozen param {name} should not change"

    def test_modification_logged(self):
        x = torch.randn(1, 64)
        self.router(x, self_modify=True)

        history = self.router.get_modification_history()
        assert len(history) == 1
        assert "route" in history[0]
        assert "coherence" in history[0]
        assert "step" in history[0]


class TestSafetyGuard:
    """Verify weight safety boundaries."""

    def setup_method(self):
        self.config = RouterConfig(input_dim=64, hidden_dim=32)
        self.router = SelfModifyingRouter(self.config)
        self.router.init_safety()

    def test_safety_guard_initialized(self):
        assert self.router._safety_guard is not None

    def test_checkpoint_and_rollback(self):
        x = torch.randn(1, 64)
        self.router._safety_guard.checkpoint()

        # Modify
        self.router(x, self_modify=True)

        # Rollback
        assert self.router._safety_guard.rollback() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
