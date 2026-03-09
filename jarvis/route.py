"""
Per-query routing — the main loop.

Usage:
    python jarvis/route.py "What is spatial computing?"
    echo "long query..." | python jarvis/route.py --stdin

Output: structured JSON to stdout.
Logs: stderr only.
"""

import hashlib
import json
import sys
from pathlib import Path

# Handle both direct execution and package import
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from jarvis.geometry import (
        project_to_manifold, coherence_score, face_activations,
        adjacency_penalty, dominant_faces, normalize, vec_add, vec_scale,
        FACE_NAMES, FACE_DESC, NUM_FACES,
    )
    from jarvis.classify import classify_query
    from jarvis.state import load_state, save_state, JarvisState, STATE_FILE
else:
    from .geometry import (
        project_to_manifold, coherence_score, face_activations,
        adjacency_penalty, dominant_faces, normalize, vec_add, vec_scale,
        FACE_NAMES, FACE_DESC, NUM_FACES,
    )
    from .classify import classify_query
    from .state import load_state, save_state, JarvisState, STATE_FILE


def _load_tools_registry() -> dict:
    """Load tool registry from tools.yaml. Pure stdlib YAML-subset parser."""
    tools_path = Path(__file__).parent / "tools.yaml"
    if not tools_path.exists():
        print("[route] tools.yaml not found, using empty registry", file=sys.stderr)
        return {}

    # Minimal YAML parser for our specific format (no external deps)
    tools = {}
    current_tool = None
    current_section = None
    text = tools_path.read_text()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level "tools:" marker
        if stripped == "tools:":
            continue

        # Tool name (2-space indent, ends with colon)
        indent = len(line) - len(line.lstrip())
        if indent == 2 and stripped.endswith(":") and not stripped.startswith("-"):
            current_tool = stripped[:-1].strip()
            tools[current_tool] = {"face_affinity": {}, "tags": [], "description": "", "latency": "low", "cost": "none"}
            current_section = None
            continue

        if current_tool is None:
            continue

        # Properties of current tool
        if indent == 4:
            if stripped.startswith("description:"):
                val = stripped.split(":", 1)[1].strip().strip('"\'')
                tools[current_tool]["description"] = val
            elif stripped.startswith("latency:"):
                tools[current_tool]["latency"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("cost:"):
                tools[current_tool]["cost"] = stripped.split(":", 1)[1].strip()
            elif stripped == "face_affinity:":
                current_section = "face_affinity"
            elif stripped.startswith("tags:"):
                # Parse inline array: [tag1, tag2, ...]
                tag_str = stripped.split(":", 1)[1].strip()
                tag_str = tag_str.strip("[]")
                tools[current_tool]["tags"] = [t.strip() for t in tag_str.split(",") if t.strip()]
                current_section = None
            continue

        # Face affinity entries (6-space indent)
        if indent == 6 and current_section == "face_affinity":
            parts = stripped.split(":")
            if len(parts) == 2:
                face = parts[0].strip()
                try:
                    val = float(parts[1].strip())
                    tools[current_tool]["face_affinity"][face] = val
                except ValueError:
                    pass

    return tools


def _score_tools(
    face_weights: list[float],
    tools: dict,
    tool_success: dict,
) -> list[dict]:
    """
    Score and rank tools based on face weights and historical success.

    For each tool with face affinity:
        score = sum(face_weight[i] * affinity[i]) * (1 + success_rate)
    """
    scored = []
    for tool_name, tool_info in tools.items():
        affinity = tool_info.get("face_affinity", {})
        if not affinity:
            continue

        # Base score from face alignment
        score = 0.0
        for face_name, aff_val in affinity.items():
            face_idx = FACE_NAMES.index(face_name) if face_name in FACE_NAMES else -1
            if face_idx >= 0:
                score += face_weights[face_idx] * aff_val

        # Boost from historical success
        success_info = tool_success.get(tool_name, {})
        avg_quality = success_info.get("avg_quality", 0.5)
        uses = success_info.get("uses", 0)
        if uses > 0:
            score *= (1.0 + avg_quality * 0.5)

        # Priority label
        if score > 0.15:
            priority = "high"
        elif score > 0.08:
            priority = "medium"
        else:
            priority = "low"

        # Build reason from dominant face alignment
        top_face = max(affinity, key=affinity.get)
        reason = f"{FACE_DESC.get(top_face, top_face)} ({tool_info.get('description', '')})"

        scored.append({
            "tool": tool_name,
            "score": score,
            "priority": priority,
            "reason": reason,
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _build_constraints(face_weights: list[float]) -> list[str]:
    """Build constraint strings based on low-activation faces."""
    constraints = []
    for i, (name, weight) in enumerate(zip(FACE_NAMES, face_weights)):
        if weight < 0.08:
            constraints.append(f"Low {name} weight — {_LOW_CONSTRAINT_MSG.get(name, 'weight is low')}")
    return constraints


_LOW_CONSTRAINT_MSG = {
    "DIAGNOSE": "situation is clear, don't over-analyze",
    "LOAD": "context is sufficient, don't over-retrieve",
    "CONSTRAIN": "be expansive, don't over-limit the response",
    "GENERATE": "don't create new artifacts unless necessary",
    "BREAK": "don't stress-test, the approach is sound",
    "CUT": "don't compress prematurely, let the analysis breathe",
    "ROUTE": "straightforward dispatch, no complex orchestration needed",
}


def _context_shape(face_weights: list[float]) -> dict:
    """Determine response context shape from face weights."""
    dom = dominant_faces(face_weights, threshold=0.12)

    # Tone
    if "DIAGNOSE" in dom[:2]:
        tone = "analytical"
    elif "GENERATE" in dom[:2]:
        tone = "creative"
    elif "BREAK" in dom[:2]:
        tone = "critical"
    elif "CUT" in dom[:2]:
        tone = "decisive"
    else:
        tone = "balanced"

    # Depth
    diagnose_w = face_weights[0]
    if diagnose_w > 0.25:
        depth = "deep"
    elif diagnose_w > 0.15:
        depth = "moderate"
    else:
        depth = "concise"

    # Format
    if "CUT" in dom[:2]:
        fmt = "bullet_points"
    elif "GENERATE" in dom[:2]:
        fmt = "structured"
    else:
        fmt = "prose"

    return {"tone": tone, "depth": depth, "format": fmt}


def route_query(
    query: str,
    state_file: str = STATE_FILE,
    history: list[dict] | None = None,
) -> dict:
    """
    Route a query through the Chestahedron geometry.

    Returns the routing plan as a dict (JSON-serializable).
    """
    state = load_state(state_file)
    tools = _load_tools_registry()

    # Classify query → raw face weights
    raw_weights = classify_query(query, history or state.routing_history)

    # Apply face_bias from state (learned adaptation)
    adjusted = [raw_weights[i] + state.face_bias[i] for i in range(NUM_FACES)]
    # Re-floor and renormalize
    adjusted = [max(w, 0.02) for w in adjusted]
    total = sum(adjusted)
    adjusted = [w / total for w in adjusted]

    # Project onto Chestahedron manifold
    manifold_weights = project_to_manifold(adjusted)

    # Compute coherence
    coh = coherence_score(adjusted)

    # Face activations (for routing)
    activations = face_activations(adjusted)

    # Adjacency check
    adj_pen = adjacency_penalty(activations)
    adj_ok = adj_pen < 0.05

    # Tool scoring
    tool_sequence_raw = _score_tools(activations, tools, state.tool_success)
    # Filter to tools with activation > 0.1 on any aligned face
    tool_sequence = [t for t in tool_sequence_raw if t["score"] > 0.03][:8]
    # Remove internal score from output
    for t in tool_sequence:
        del t["score"]

    # Constraints
    constraints = _build_constraints(activations)

    # Context shape
    ctx_shape = _context_shape(activations)

    # Query hash
    q_hash = hashlib.md5(query.encode()).hexdigest()[:8]

    # Build operation weights dict
    op_weights = {FACE_NAMES[i]: round(activations[i], 4) for i in range(NUM_FACES)}

    # Dominant operations
    dom_ops = dominant_faces(activations, threshold=0.1)

    # Update state
    state.step += 1
    state.chestahedron_state = list(activations)
    state.coherence_trajectory.append(round(coh, 4))
    state.routing_history.append({
        "step": state.step,
        "query_hash": q_hash,
        "route": dom_ops[0] if dom_ops else "DIAGNOSE",
        "faces": op_weights,
        "coherence": round(coh, 4),
    })
    save_state(state, state_file)

    return {
        "step": state.step,
        "query_hash": q_hash,
        "operation_weights": op_weights,
        "dominant_operations": dom_ops,
        "tool_sequence": tool_sequence,
        "constraints": constraints,
        "context_shape": ctx_shape,
        "coherence": round(coh, 4),
        "adjacency_ok": adj_ok,
        "manifold_distance": round(1.0 - coh, 4),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="JARVIS query router")
    parser.add_argument("query", nargs="?", default=None, help="Query text")
    parser.add_argument("--stdin", action="store_true", help="Read query from stdin")
    parser.add_argument("--state-file", default=STATE_FILE, help="State file path")
    args = parser.parse_args()

    if args.stdin:
        query = sys.stdin.read().strip()
    elif args.query:
        query = args.query
    else:
        print('{"error": "No query provided. Use: route.py \\"query\\" or --stdin"}', flush=True)
        sys.exit(1)

    if not query:
        print('{"error": "Empty query"}', flush=True)
        sys.exit(1)

    result = route_query(query, state_file=args.state_file)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
