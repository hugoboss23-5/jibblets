"""
Session initialization — run once at session start.

Usage:
    python jarvis/boot.py [--state-file PATH] [--session-id ID]

Output: JSON to stdout with session status.
"""

import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from jarvis.geometry import self_test as geo_self_test, coherence_score, NUM_FACES, FACE_NAMES
    from jarvis.state import load_state, save_state, new_state, is_state_fresh, STATE_FILE
else:
    from .geometry import self_test as geo_self_test, coherence_score, NUM_FACES, FACE_NAMES
    from .state import load_state, save_state, new_state, is_state_fresh, STATE_FILE


def _count_tools() -> int:
    """Count registered tools from tools.yaml."""
    tools_path = Path(__file__).parent / "tools.yaml"
    if not tools_path.exists():
        return 0
    count = 0
    for line in tools_path.read_text().splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if indent == 2 and stripped.endswith(":") and not stripped.startswith("#") and stripped != "tools:":
            count += 1
    return count


def boot(state_file: str = STATE_FILE, session_id: str | None = None) -> dict:
    """
    Initialize a JARVIS session.

    1. Check for existing state file. If found and < 4 hours old, resume.
    2. Otherwise create fresh state.
    3. Run geometry self-test (silent).
    4. Return status JSON.
    """
    resumed = False

    if is_state_fresh(state_file):
        state = load_state(state_file)
        resumed = True
        print(f"[boot] Resuming session {state.session_id} (step {state.step})", file=sys.stderr)
    else:
        state = new_state(session_id)
        save_state(state, state_file)
        print(f"[boot] New session {state.session_id}", file=sys.stderr)

    # Geometry self-test (silent — only warn on failure)
    geo_ok = geo_self_test()
    if not geo_ok:
        print("[boot] WARNING: geometry self-test failed!", file=sys.stderr)

    # Initial coherence
    initial_coh = coherence_score(state.chestahedron_state)

    tools_count = _count_tools()

    return {
        "status": "ready",
        "session_id": state.session_id,
        "resumed": resumed,
        "step": state.step,
        "initial_coherence": round(initial_coh, 4),
        "faces": FACE_NAMES,
        "tools_registered": tools_count,
        "geometry_ok": geo_ok,
        "message": f"Jarvis routing cortex online. {NUM_FACES} faces active. Coherence nominal.",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="JARVIS session boot")
    parser.add_argument("--state-file", default=STATE_FILE, help="State file path")
    parser.add_argument("--session-id", default=None, help="Custom session ID")
    args = parser.parse_args()

    result = boot(args.state_file, args.session_id)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
