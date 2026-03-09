"""
Post-query adaptation — run after Claude completes a response.

Usage:
    python jarvis/track.py --quality 0.8 --query-hash "a1b2c3" --tools-used "web_search,kd_vault_query"

Output: JSON to stdout with adaptation summary.
"""

import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from jarvis.state import load_state, save_state, STATE_FILE
    from jarvis.geometry import NUM_FACES, FACE_NAMES
else:
    from .state import load_state, save_state, STATE_FILE
    from .geometry import NUM_FACES, FACE_NAMES


def track(
    quality: float,
    query_hash: str,
    tools_used: list[str],
    state_file: str = STATE_FILE,
) -> dict:
    """
    Post-query adaptation. Updates face_bias and tool_success based on feedback.

    Returns tracking summary as dict.
    """
    state = load_state(state_file)

    # Find the routing entry for this query_hash
    entry = None
    for h in reversed(state.routing_history):
        if h.get("query_hash") == query_hash:
            entry = h
            break

    if entry is None:
        # No matching routing entry — still track quality
        entry = {
            "step": state.step,
            "query_hash": query_hash,
            "faces": {f: 1.0 / NUM_FACES for f in FACE_NAMES},
            "coherence": 0.5,
        }

    # Record quality in the routing entry
    entry["quality"] = quality
    entry["tools_used"] = tools_used

    # --- ADAPT face_bias ---
    face_weights = entry.get("faces", {})
    bias_updated = False
    for i, face_name in enumerate(FACE_NAMES):
        w = face_weights.get(face_name, 0.0)
        if w > 0.15:  # Dominant face
            if quality >= 0.7:
                # Strengthen: bias += 0.01 * (quality - 0.5)
                state.face_bias[i] += 0.01 * (quality - 0.5)
                bias_updated = True
            elif quality < 0.4:
                # Weaken: bias -= 0.02 * (0.5 - quality)
                state.face_bias[i] -= 0.02 * (0.5 - quality)
                bias_updated = True

    # Normalize face_bias to sum to 0 (adjustment, not absolute)
    bias_mean = sum(state.face_bias) / NUM_FACES
    state.face_bias = [b - bias_mean for b in state.face_bias]

    # Clamp each bias to [-0.3, 0.3]
    state.face_bias = [max(-0.3, min(0.3, b)) for b in state.face_bias]

    # --- ADAPT tool_success ---
    for tool in tools_used:
        if tool not in state.tool_success:
            state.tool_success[tool] = {"uses": 0, "avg_quality": 0.5}
        ts = state.tool_success[tool]
        ts["avg_quality"] = ts["avg_quality"] * 0.8 + quality * 0.2
        ts["uses"] += 1

    # --- TRACK coherence ---
    coherence_trend = "stable"
    adaptation_note = None
    if len(state.coherence_trajectory) >= 3:
        last3 = state.coherence_trajectory[-3:]
        if all(last3[i] > last3[i + 1] for i in range(len(last3) - 1)):
            coherence_trend = "declining"
            adaptation_note = "Coherence dropping 3+ steps. Consider recalibrating."
            print(f"[track] WARNING: coherence declining: {last3}", file=sys.stderr)
        elif all(last3[i] < last3[i + 1] for i in range(len(last3) - 1)):
            coherence_trend = "improving"
    elif len(state.coherence_trajectory) >= 1:
        coherence_trend = "insufficient_data"

    save_state(state, state_file)

    return {
        "tracked": True,
        "step": state.step,
        "quality": quality,
        "coherence_trend": coherence_trend,
        "face_bias_updated": bias_updated,
        "tools_tracked": tools_used,
        "adaptation_note": adaptation_note,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="JARVIS post-query tracker")
    parser.add_argument("--quality", type=float, required=True, help="Quality signal 0-1")
    parser.add_argument("--query-hash", required=True, help="Hash from route output")
    parser.add_argument("--tools-used", default="", help="Comma-separated tool names")
    parser.add_argument("--state-file", default=STATE_FILE, help="State file path")
    args = parser.parse_args()

    tools = [t.strip() for t in args.tools_used.split(",") if t.strip()]
    result = track(args.quality, args.query_hash, tools, args.state_file)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
