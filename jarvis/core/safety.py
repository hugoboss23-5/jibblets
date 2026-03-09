"""
Safety module — weight modification boundaries and reversibility.

Enforces that only designated "modifiable" parameters can be changed during
self-modification. Core safety weights remain frozen. All modifications are
logged and reversible.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ModificationRecord:
    """Record of a single weight modification event."""
    step: int
    param_name: str
    pre_norm: float
    post_norm: float
    delta_norm: float
    timestamp: float = 0.0


@dataclass
class SafetyState:
    """Tracks safety-relevant state across the router's lifetime."""
    total_modifications: int = 0
    total_reverts: int = 0
    max_delta_observed: float = 0.0
    records: list[ModificationRecord] = field(default_factory=list)
    max_records: int = 1000

    def add_record(self, record: ModificationRecord):
        if len(self.records) >= self.max_records:
            self.records.pop(0)
        self.records.append(record)
        self.total_modifications += 1
        self.max_delta_observed = max(self.max_delta_observed, record.delta_norm)


class WeightSafetyGuard:
    """
    Guards which parameters of a model are modifiable and which are frozen.

    Partitions model parameters into:
    - FROZEN: safety-critical weights that must never change during inference
    - MODIFIABLE: routing/adaptation weights that can self-modify
    - MONITORED: weights that can change but are rate-limited

    Also provides checkpointing and rollback capabilities.
    """

    def __init__(
        self,
        model: nn.Module,
        frozen_prefixes: list[str] | None = None,
        modifiable_prefixes: list[str] | None = None,
        max_delta_per_step: float = 1.0,
        checkpoint_dir: str | None = None,
    ):
        self.model = model
        self.max_delta_per_step = max_delta_per_step
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.state = SafetyState()

        # Default: safety-related prefixes are frozen
        self.frozen_prefixes = frozen_prefixes or [
            "safety_",
            "constraint_",
            "boundary_",
        ]
        self.modifiable_prefixes = modifiable_prefixes or [
            "fast_",
            "hyper",
            "routing_",
            "gate",
            "scale",
            "lr_gate",
            "decay_gate",
        ]

        # Snapshot storage for rollback
        self._checkpoints: list[dict[str, torch.Tensor]] = []
        self._max_checkpoints = 10

        # Cache parameter classification
        self._frozen_params: set[str] = set()
        self._modifiable_params: set[str] = set()
        self._classify_params()

        # Apply initial freeze
        self._apply_freeze()

    def _classify_params(self):
        """Classify all parameters as frozen or modifiable."""
        for name, _ in self.model.named_parameters():
            if any(name.startswith(p) or f".{p}" in name for p in self.frozen_prefixes):
                self._frozen_params.add(name)
            elif any(
                name.startswith(p) or f".{p}" in name for p in self.modifiable_prefixes
            ):
                self._modifiable_params.add(name)
            # Unclassified params default to frozen (conservative)

    def _apply_freeze(self):
        """Set requires_grad=False on frozen parameters."""
        for name, param in self.model.named_parameters():
            if name in self._frozen_params:
                param.requires_grad = False

    def is_modifiable(self, param_name: str) -> bool:
        """Check if a parameter is allowed to be modified."""
        return param_name in self._modifiable_params

    def validate_modification(
        self, param_name: str, old_value: torch.Tensor, new_value: torch.Tensor, step: int
    ) -> bool:
        """
        Validate a proposed weight modification against safety constraints.

        Returns True if the modification is allowed.
        """
        if not self.is_modifiable(param_name):
            logger.warning(f"Blocked modification to frozen param: {param_name}")
            return False

        delta = (new_value - old_value).norm().item()
        if delta > self.max_delta_per_step:
            logger.warning(
                f"Modification to {param_name} exceeds max delta: "
                f"{delta:.4f} > {self.max_delta_per_step}"
            )
            return False

        # Record the modification
        record = ModificationRecord(
            step=step,
            param_name=param_name,
            pre_norm=old_value.norm().item(),
            post_norm=new_value.norm().item(),
            delta_norm=delta,
        )
        self.state.add_record(record)

        return True

    def checkpoint(self) -> int:
        """Save current model state. Returns checkpoint index."""
        state = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if name in self._modifiable_params
        }
        if len(self._checkpoints) >= self._max_checkpoints:
            self._checkpoints.pop(0)
        self._checkpoints.append(state)
        idx = len(self._checkpoints) - 1
        logger.info(f"Checkpoint saved: {idx} ({len(state)} modifiable params)")
        return idx

    def rollback(self, checkpoint_idx: int = -1) -> bool:
        """Restore model to a previous checkpoint."""
        if not self._checkpoints:
            logger.warning("No checkpoints available for rollback")
            return False

        state = self._checkpoints[checkpoint_idx]
        param_dict = dict(self.model.named_parameters())
        for name, saved_value in state.items():
            if name in param_dict:
                param_dict[name].data.copy_(saved_value)

        self.state.total_reverts += 1
        logger.info(f"Rolled back to checkpoint {checkpoint_idx}")
        return True

    def save_to_disk(self, path: str | Path | None = None):
        """Save modifiable weights and safety state to disk."""
        save_path = Path(path) if path else self.checkpoint_dir
        if save_path is None:
            raise ValueError("No save path specified")
        save_path.mkdir(parents=True, exist_ok=True)

        # Save modifiable weights
        modifiable_state = {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
            if name in self._modifiable_params
        }
        torch.save(modifiable_state, save_path / "modifiable_weights.pt")
        logger.info(f"Saved {len(modifiable_state)} modifiable params to {save_path}")

    def load_from_disk(self, path: str | Path | None = None):
        """Load modifiable weights from disk."""
        load_path = Path(path) if path else self.checkpoint_dir
        if load_path is None:
            raise ValueError("No load path specified")

        weights_file = load_path / "modifiable_weights.pt"
        if not weights_file.exists():
            logger.warning(f"No saved weights at {weights_file}")
            return

        saved_state = torch.load(weights_file, weights_only=True)
        param_dict = dict(self.model.named_parameters())

        loaded = 0
        for name, saved_value in saved_state.items():
            if name in param_dict and name in self._modifiable_params:
                param_dict[name].data.copy_(saved_value)
                loaded += 1

        logger.info(f"Loaded {loaded} modifiable params from {load_path}")
