"""
State file management for JARVIS routing cortex.
Pure Python stdlib only — JSON read/write with dataclass.
"""

import json
import hashlib
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .geometry import NUM_FACES

STATE_FILE = "jarvis_state.json"


@dataclass
class JarvisState:
    session_id: str = ""
    step: int = 0
    chestahedron_state: list[float] = field(default_factory=lambda: [1.0 / NUM_FACES] * NUM_FACES)
    routing_history: list[dict] = field(default_factory=list)
    coherence_trajectory: list[float] = field(default_factory=list)
    face_bias: list[float] = field(default_factory=lambda: [0.0] * NUM_FACES)
    tool_success: dict[str, dict[str, Any]] = field(default_factory=dict)
    session_context: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


def new_state(session_id: str | None = None) -> JarvisState:
    """Create a fresh state with a new session ID."""
    if session_id is None:
        now = datetime.now()
        session_id = now.strftime("%Y-%m-%d") + "-" + hashlib.md5(
            now.isoformat().encode()
        ).hexdigest()[:6]
    now_str = datetime.now().isoformat()
    return JarvisState(
        session_id=session_id,
        created_at=now_str,
        updated_at=now_str,
    )


def load_state(path: str = STATE_FILE) -> JarvisState:
    """Load state from JSON file. Returns new state if file missing/corrupt."""
    p = Path(path)
    if not p.exists():
        print(f"[state] No state file at {path}, creating fresh", file=sys.stderr)
        return new_state()
    try:
        data = json.loads(p.read_text())
        state = JarvisState()
        for k, v in data.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"[state] Corrupt state file: {e}, creating fresh", file=sys.stderr)
        return new_state()


def save_state(state: JarvisState, path: str = STATE_FILE):
    """Save state to JSON file."""
    state.updated_at = datetime.now().isoformat()
    p = Path(path)
    p.write_text(json.dumps(asdict(state), indent=2))


def is_state_fresh(path: str = STATE_FILE, max_age_hours: float = 4.0) -> bool:
    """Check if state file exists and is less than max_age_hours old."""
    p = Path(path)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text())
        updated = datetime.fromisoformat(data.get("updated_at", "2000-01-01"))
        age = (datetime.now() - updated).total_seconds() / 3600
        return age < max_age_hours
    except Exception:
        return False


if __name__ == "__main__":
    import tempfile
    import os

    log = lambda msg: print(msg, file=sys.stderr)
    log("=== State Management Self-Test ===")

    # Test in temp dir
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test_state.json")

        # Test new state
        s = new_state("test-001")
        assert s.session_id == "test-001"
        assert len(s.chestahedron_state) == NUM_FACES
        assert s.step == 0
        log("  PASS: new_state")

        # Test save/load
        save_state(s, path)
        s2 = load_state(path)
        assert s2.session_id == "test-001"
        assert s2.step == 0
        log("  PASS: save/load roundtrip")

        # Test mutation
        s2.step = 5
        s2.routing_history.append({"test": True})
        save_state(s2, path)
        s3 = load_state(path)
        assert s3.step == 5
        assert len(s3.routing_history) == 1
        log("  PASS: mutation persists")

        # Test missing file
        s4 = load_state(os.path.join(td, "nonexistent.json"))
        assert s4.step == 0  # Fresh state
        log("  PASS: missing file returns fresh state")

        # Test freshness
        assert is_state_fresh(path, max_age_hours=1.0)
        log("  PASS: freshness check")

    log("=== All state tests passed ===")
