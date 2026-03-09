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
    # Lowered threshold from 2 to 1 query with quality feedback (Problem 7 fix 1)
    quality_by_dominant = {}
    for entry in history:
        q = entry.get("quality")
        if q is None:
            continue
        faces = entry.get("faces", {})
        if faces:
            dominant = max(faces, key=faces.get)
            quality_by_dominant.setdefault(dominant, []).append(q)

    for face, qualities in quality_by_dominant.items():
        if len(qualities) >= 1:  # Lowered from 2 to 1
            avg_q = sum(qualities) / len(qualities)
            n = len(qualities)
            conf = min(0.5 + n * 0.1, 0.95)  # Scale confidence with sample size
            if avg_q >= 0.7:
                beliefs.append({
                    "content": f"{face}-dominant routing produces high quality (avg {avg_q:.2f} over {n} {'query' if n == 1 else 'queries'})",
                    "confidence": round(min(avg_q * conf, 0.95), 2),
                    "domain": "jarvis_routing",
                })
            elif avg_q < 0.4:
                beliefs.append({
                    "content": f"{face}-dominant routing underperforms (avg {avg_q:.2f}) — recalibrate {face} bias",
                    "confidence": round(0.5 + (0.5 - avg_q), 2),
                    "domain": "jarvis_routing",
                })

    # --- Session-level summary belief (Problem 7 fix 2 — always fires) ---
    all_dominant_faces = []
    for entry in history:
        faces = entry.get("faces", {})
        if faces:
            all_dominant_faces.append(max(faces, key=faces.get))

    if all_dominant_faces:
        face_counts = Counter(all_dominant_faces)
        top_faces = face_counts.most_common(3)
        top_desc = ", ".join(f"{f} ({c}x)" for f, c in top_faces)
        beliefs.append({
            "content": f"Session had {len(history)} queries. Dominant faces: {top_desc}",
            "confidence": 0.85,
            "domain": "jarvis_meta",
        })

    # --- Session fingerprint (Problem 7 fix 5) ---
    if len(history) >= 2:
        face_totals = {f: 0.0 for f in FACE_NAMES}
        for entry in history:
            faces = entry.get("faces", {})
            for f, w in faces.items():
                face_totals[f] = face_totals.get(f, 0) + w
        # Normalize
        n_entries = len(history)
        avg_faces = {f: round(t / n_entries, 3) for f, t in face_totals.items()}
        top_face = max(avg_faces, key=avg_faces.get)

        # Characterize session type
        session_type = "balanced"
        if avg_faces.get("LOAD", 0) > 0.22:
            session_type = "research-heavy"
        elif avg_faces.get("GENERATE", 0) > 0.22:
            session_type = "build-heavy"
        elif avg_faces.get("DIAGNOSE", 0) > 0.25:
            session_type = "diagnostic-heavy"
        elif avg_faces.get("BREAK", 0) > 0.20:
            session_type = "evaluation-heavy"
        elif avg_faces.get("CUT", 0) > 0.20:
            session_type = "decision-heavy"

        facts.append({
            "content": f"Session fingerprint: {session_type} (dominant face avg: {top_face}={avg_faces[top_face]:.3f})",
            "source": f"jarvis_session_{state.session_id}",
            "domain": "jarvis_meta",
        })

    # --- Tool frequency (Problem 7 fix 2) ---
    tool_usage_counts = Counter()
    for entry in history:
        for t in entry.get("tools_used", []):
            tool_usage_counts[t] += 1
    if tool_usage_counts:
        top_tools = tool_usage_counts.most_common(3)
        tool_desc = ", ".join(f"{t} ({c}x)" for t, c in top_tools)
        facts.append({
            "content": f"Most recommended tools: {tool_desc}",
            "source": f"jarvis_session_{state.session_id}",
            "domain": "jarvis_routing",
        })

    # --- Analyze which tool sequences worked best ---
    tool_qualities = {}
    for entry in history:
        tools = entry.get("tools_used", [])
        q = entry.get("quality")
        if tools and q is not None:
            key = ",".join(sorted(tools))
            tool_qualities.setdefault(key, []).append(q)

    for tool_seq, qualities in tool_qualities.items():
        avg_q = sum(qualities) / len(qualities)
        n = len(qualities)
        facts.append({
            "content": f"Tool sequence [{tool_seq}] averaged {avg_q:.2f} quality over {n} {'use' if n == 1 else 'uses'}",
            "source": f"jarvis_session_{state.session_id}",
            "domain": "jarvis_routing",
        })

    # --- Individual tool success rates ---
    for tool_name, info in state.tool_success.items():
        uses = info.get("uses", 0)
        avg_q = info.get("avg_quality", 0)
        if uses >= 2:  # Lowered from 3
            facts.append({
                "content": f"Tool '{tool_name}' averaged {avg_q:.2f} quality over {uses} uses",
                "source": f"jarvis_session_{state.session_id}",
                "domain": "jarvis_routing",
            })

    # --- Coherence trajectory analysis ---
    coh = state.coherence_trajectory
    if coh:
        avg_coh = sum(coh) / len(coh)

        # Coherence delta belief (Problem 7 fix 3)
        if len(coh) >= 2:
            delta = coh[-1] - coh[0]
            if abs(delta) > 0.03:
                direction = "improved" if delta > 0 else "degraded"
                beliefs.append({
                    "content": f"Session coherence {direction} from {coh[0]:.3f} to {coh[-1]:.3f} (delta={delta:+.3f})",
                    "confidence": round(min(0.6 + abs(delta), 0.9), 2),
                    "domain": "jarvis_meta",
                })

        # Overall trend
        trend = "stable"
        if len(coh) >= 5:
            first_half = sum(coh[:len(coh)//2]) / (len(coh)//2)
            second_half = sum(coh[len(coh)//2:]) / (len(coh) - len(coh)//2)
            if second_half > first_half + 0.05:
                trend = "improving"
            elif second_half < first_half - 0.05:
                trend = "degrading"

        if len(coh) >= 3:
            beliefs.append({
                "content": f"Session coherence was {trend} (avg {avg_coh:.3f}, {len(coh)} steps)",
                "confidence": round(min(0.6 + len(coh) * 0.02, 0.95), 2),
                "domain": "jarvis_meta",
            })

    # --- Face bias learnings (lowered threshold to 0.03, Problem 7 fix 4) ---
    significant_biases = [
        (FACE_NAMES[i], state.face_bias[i])
        for i in range(NUM_FACES)
        if abs(state.face_bias[i]) > 0.03
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

    return {
        "ready_for_kd": True,
        "session_id": state.session_id,
        "total_steps": len(history),
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
