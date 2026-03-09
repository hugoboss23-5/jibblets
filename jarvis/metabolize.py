"""
Session-end knowledge packaging for KD deposit.

Usage:
    python jarvis/metabolize.py [--state-file PATH]

Output: JSON payload for KD:metabolize.
"""

import json
import sys
from pathlib import Path
from collections import Counter

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from jarvis.state import load_state, STATE_FILE
    from jarvis.geometry import FACE_NAMES, NUM_FACES
else:
    from .state import load_state, STATE_FILE
    from .geometry import FACE_NAMES, NUM_FACES


def metabolize(state_file: str = STATE_FILE) -> dict:
    """
    Analyze routing history and package learnings for KD deposit.

    Returns:
        dict with beliefs, facts, chains, and session metadata.
    """
    state = load_state(state_file)
    history = state.routing_history

    if not history:
        return {
            "ready_for_kd": False,
            "session_id": state.session_id,
            "total_steps": 0,
            "reason": "No routing history to metabolize",
        }

    beliefs = []
    facts = []
    chains = []

    # --- Analyze which face patterns led to highest quality ---
    quality_by_dominant = {}  # face_name -> [quality scores]
    for entry in history:
        q = entry.get("quality")
        if q is None:
            continue
        faces = entry.get("faces", {})
        if faces:
            dominant = max(faces, key=faces.get)
            quality_by_dominant.setdefault(dominant, []).append(q)

    for face, qualities in quality_by_dominant.items():
        if len(qualities) >= 2:
            avg_q = sum(qualities) / len(qualities)
            if avg_q >= 0.7:
                beliefs.append({
                    "content": f"{face}-dominant routing produces high quality (avg {avg_q:.2f} over {len(qualities)} queries)",
                    "confidence": round(min(avg_q, 0.95), 2),
                    "domain": "jarvis_routing",
                })
            elif avg_q < 0.4:
                beliefs.append({
                    "content": f"{face}-dominant routing underperforms (avg {avg_q:.2f}) — recalibrate {face} bias",
                    "confidence": round(0.5 + (0.5 - avg_q), 2),
                    "domain": "jarvis_routing",
                })

    # --- Analyze which tool sequences worked best ---
    tool_qualities = {}  # "tool1,tool2" -> [quality]
    for entry in history:
        tools = entry.get("tools_used", [])
        q = entry.get("quality")
        if tools and q is not None:
            key = ",".join(sorted(tools))
            tool_qualities.setdefault(key, []).append(q)

    for tool_seq, qualities in tool_qualities.items():
        if len(qualities) >= 2:
            avg_q = sum(qualities) / len(qualities)
            facts.append({
                "content": f"Tool sequence [{tool_seq}] averaged {avg_q:.2f} quality over {len(qualities)} uses",
                "source": f"jarvis_session_{state.session_id}",
                "domain": "jarvis_routing",
            })

    # --- Individual tool success rates ---
    for tool_name, info in state.tool_success.items():
        uses = info.get("uses", 0)
        avg_q = info.get("avg_quality", 0)
        if uses >= 3:
            facts.append({
                "content": f"Tool '{tool_name}' averaged {avg_q:.2f} quality over {uses} uses",
                "source": f"jarvis_session_{state.session_id}",
                "domain": "jarvis_routing",
            })

    # --- Coherence trajectory analysis ---
    coh = state.coherence_trajectory
    if len(coh) >= 3:
        avg_coh = sum(coh) / len(coh)
        trend = "stable"
        if len(coh) >= 5:
            first_half = sum(coh[:len(coh)//2]) / (len(coh)//2)
            second_half = sum(coh[len(coh)//2:]) / (len(coh) - len(coh)//2)
            if second_half > first_half + 0.05:
                trend = "improving"
            elif second_half < first_half - 0.05:
                trend = "degrading"

        beliefs.append({
            "content": f"Session coherence was {trend} (avg {avg_coh:.3f}, {len(coh)} steps)",
            "confidence": round(min(0.6 + len(coh) * 0.02, 0.95), 2),
            "domain": "jarvis_meta",
        })

    # --- Face bias learnings ---
    significant_biases = [
        (FACE_NAMES[i], state.face_bias[i])
        for i in range(NUM_FACES)
        if abs(state.face_bias[i]) > 0.05
    ]
    if significant_biases:
        bias_desc = ", ".join(f"{name}={bias:+.3f}" for name, bias in significant_biases)
        beliefs.append({
            "content": f"Learned face biases this session: {bias_desc}",
            "confidence": 0.65,
            "domain": "jarvis_routing",
        })

    # --- Build chains from sequential patterns ---
    if len(history) >= 3:
        # Look for quality improvement patterns
        for i in range(len(history) - 2):
            h0 = history[i]
            h1 = history[i + 1]
            h2 = history[i + 2]
            q0 = h0.get("quality")
            q2 = h2.get("quality")
            if q0 is not None and q2 is not None and q2 > q0 + 0.2:
                chains.append({
                    "steps": [
                        f"Step {h0.get('step')}: {h0.get('route', '?')} (q={q0:.1f})",
                        f"Step {h1.get('step')}: adapted routing",
                        f"Step {h2.get('step')}: {h2.get('route', '?')} (q={q2:.1f})",
                    ],
                    "conclusion": f"Quality improved {q0:.1f} → {q2:.1f} over 3 steps via adaptation",
                })

    # --- Overall metrics ---
    avg_coh = sum(coh) / len(coh) if coh else 0.0
    total_steps = len(history)

    return {
        "ready_for_kd": True,
        "session_id": state.session_id,
        "total_steps": total_steps,
        "avg_coherence": round(avg_coh, 3),
        "payload": {
            "beliefs": beliefs,
            "facts": facts,
            "chains": chains,
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="JARVIS session metabolizer")
    parser.add_argument("--state-file", default=STATE_FILE, help="State file path")
    args = parser.parse_args()

    result = metabolize(args.state_file)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
