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
    from jarvis.classify import classify_query, is_trivial, has_past_reference
    from jarvis.state import load_state, save_state, JarvisState, STATE_FILE
else:
    from .geometry import (
        project_to_manifold, coherence_score, face_activations,
        adjacency_penalty, dominant_faces, normalize, vec_add, vec_scale,
        FACE_NAMES, FACE_DESC, NUM_FACES,
    )
    from .classify import classify_query, is_trivial, has_past_reference
    from .state import load_state, save_state, JarvisState, STATE_FILE


# ---- Tool reason templates (Problem 6) ----
_REASON_TEMPLATES: dict[tuple[str, str], str] = {
    ("web_search", "LOAD"): "Search for current information on this topic",
    ("web_search", "DIAGNOSE"): "Verify facts before analyzing",
    ("web_search", "BREAK"): "Cross-check claims against current sources",
    ("web_fetch", "LOAD"): "Fetch full content from the referenced URL",
    ("web_fetch", "DIAGNOSE"): "Read source material for deeper analysis",
    ("kd_vault_query", "LOAD"): "Check knowledge graph for prior context",
    ("kd_vault_query", "DIAGNOSE"): "Pull beliefs and facts for deeper analysis",
    ("kd_graph_query", "DIAGNOSE"): "Find cross-domain connections in knowledge graph",
    ("kd_graph_query", "LOAD"): "Activate related knowledge nodes",
    ("kd_graph_query", "GENERATE"): "Discover connections to inspire new approaches",
    ("kd_metabolize", "CUT"): "Package key learnings for long-term storage",
    ("kd_metabolize", "ROUTE"): "Deposit routing patterns for future sessions",
    ("kd_graph_build", "GENERATE"): "Build new knowledge connections",
    ("kd_graph_build", "ROUTE"): "Structure relationships between concepts",
    ("conversation_search", "LOAD"): "Search past conversations for relevant context",
    ("conversation_search", "DIAGNOSE"): "Check conversation history for prior analysis",
    ("recent_chats", "LOAD"): "Check recent conversations for continuity",
    ("bash_tool", "GENERATE"): "Execute code or process data",
    ("bash_tool", "BREAK"): "Run tests to verify the approach",
    ("bash_tool", "CUT"): "Process and filter data",
    ("create_file", "GENERATE"): "Create the requested deliverable",
    ("create_file", "CUT"): "Package the output into a file",
    ("create_file", "ROUTE"): "Produce final artifact",
    ("anthropic_api", "GENERATE"): "Spin up sub-agent for parallel analysis",
    ("anthropic_api", "BREAK"): "Use sub-agent to stress-test the approach",
    ("anthropic_api", "DIAGNOSE"): "Deploy sub-agent for deeper reasoning",
    ("view", "LOAD"): "Read files for context",
    ("view", "DIAGNOSE"): "Inspect files to understand the situation",
    ("google_drive_search", "LOAD"): "Search Drive for relevant documents",
    ("google_drive_search", "DIAGNOSE"): "Find related docs for analysis",
    ("google_drive_fetch", "LOAD"): "Fetch document contents from Drive",
    ("message_compose", "GENERATE"): "Draft the requested message",
    ("message_compose", "CUT"): "Compose a concise message",
    ("message_compose", "CONSTRAIN"): "Draft within specified constraints",
    ("image_search", "LOAD"): "Find visual references",
    ("image_search", "GENERATE"): "Source images for the deliverable",
    ("places_search", "LOAD"): "Look up location information",
    ("places_search", "GENERATE"): "Find relevant places and options",
    ("tool_search", "ROUTE"): "Find and load additional tools needed",
    ("tool_search", "LOAD"): "Discover available tools for this task",
}


def _load_tools_registry() -> dict:
    """Load tool registry from tools.yaml. Pure stdlib YAML-subset parser."""
    tools_path = Path(__file__).parent / "tools.yaml"
    if not tools_path.exists():
        print("[route] tools.yaml not found, using empty registry", file=sys.stderr)
        return {}

    tools = {}
    current_tool = None
    current_section = None
    text = tools_path.read_text()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped == "tools:":
            continue

        indent = len(line) - len(line.lstrip())
        if indent == 2 and stripped.endswith(":") and not stripped.startswith("-"):
            current_tool = stripped[:-1].strip()
            tools[current_tool] = {"face_affinity": {}, "tags": [], "description": "", "latency": "low", "cost": "none"}
            current_section = None
            continue

        if current_tool is None:
            continue

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
                tag_str = stripped.split(":", 1)[1].strip()
                tag_str = tag_str.strip("[]")
                tools[current_tool]["tags"] = [t.strip() for t in tag_str.split(",") if t.strip()]
                current_section = None
            continue

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


def _compute_max_tools(raw_weights: list[float], query: str) -> int:
    """
    Determine max tools based on face weight distribution and query complexity.

    Problem 1: Different query types get different tool caps.
    """
    max_w = max(raw_weights)
    spread = max_w - min(raw_weights)
    dominant_idx = raw_weights.index(max_w)
    dominant_face = FACE_NAMES[dominant_idx]

    # Complex multi-step (ROUTE dominant)
    if dominant_face == "ROUTE" and max_w > 0.20:
        return 7

    # Build/create (GENERATE dominant)
    if dominant_face == "GENERATE" and max_w > 0.20:
        return 5

    # Analytical/research (DIAGNOSE or LOAD dominant with decent spread)
    if dominant_face in ("DIAGNOSE", "LOAD") and max_w > 0.20:
        return 4

    # Break/evaluate
    if dominant_face == "BREAK":
        return 3

    # CUT/CONSTRAIN — user wants compression, fewer tools
    if dominant_face in ("CUT", "CONSTRAIN"):
        return 2

    # Uniform-ish distribution (low spread) + short query = simple
    if spread < 0.12 and len(query) < 50:
        return 2

    # Default
    return 3


def _get_tool_reason(tool_name: str, face_name: str, tool_desc: str) -> str:
    """Get a context-specific reason for a tool recommendation."""
    key = (tool_name, face_name)
    if key in _REASON_TEMPLATES:
        return _REASON_TEMPLATES[key]
    # Fallback: use tool description directly (still better than face description)
    return tool_desc


def _score_tools(
    face_weights: list[float],
    tools: dict,
    tool_success: dict,
    query: str = "",
    has_past_ref: bool = False,
) -> list[dict]:
    """
    Score and rank tools based on face weights and historical success.

    Fix Problem 1: Raised thresholds.
    Fix Problem 6: Query-specific reasons.
    Fix C: Past-reference boost for conversation tools.
    """
    scored = []
    dominant_idx = face_weights.index(max(face_weights))
    dominant_face = FACE_NAMES[dominant_idx]

    for tool_name, tool_info in tools.items():
        affinity = tool_info.get("face_affinity", {})
        if not affinity:
            continue

        # Base score from face alignment
        score = 0.0
        best_face = None
        best_face_contribution = 0.0
        for face_name, aff_val in affinity.items():
            face_idx = FACE_NAMES.index(face_name) if face_name in FACE_NAMES else -1
            if face_idx >= 0:
                contribution = face_weights[face_idx] * aff_val
                score += contribution
                if contribution > best_face_contribution:
                    best_face_contribution = contribution
                    best_face = face_name

        # Fix C: conversation_search/recent_chats boost for past references
        if has_past_ref and dominant_face == "LOAD":
            if tool_name in ("conversation_search", "recent_chats"):
                score *= 1.5

        # Boost from historical success
        success_info = tool_success.get(tool_name, {})
        avg_quality = success_info.get("avg_quality", 0.5)
        uses = success_info.get("uses", 0)
        if uses > 0:
            score *= (1.0 + avg_quality * 0.5)

        # Priority labels — recalibrated thresholds (Problem 1)
        if score > 0.20:
            priority = "high"
        elif score > 0.12:
            priority = "medium"
        else:
            priority = "low"

        # Build reason (Problem 6)
        reason = _get_tool_reason(
            tool_name,
            best_face or dominant_face,
            tool_info.get("description", tool_name),
        )

        scored.append({
            "tool": tool_name,
            "score": score,
            "priority": priority,
            "reason": reason,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _build_constraints(raw_weights: list[float]) -> list[str]:
    """
    Build constraint strings based on low-activation faces.

    Fix Problem 5: Use RAW classified weights (pre-softmax) with threshold 0.06.
    """
    constraints = []
    for i, (name, weight) in enumerate(zip(FACE_NAMES, raw_weights)):
        if weight < 0.06:
            constraints.append(
                f"Low {name} weight — {_LOW_CONSTRAINT_MSG.get(name, 'weight is low')}"
            )
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

    diagnose_w = face_weights[0]
    if diagnose_w > 0.25:
        depth = "deep"
    elif diagnose_w > 0.15:
        depth = "moderate"
    else:
        depth = "concise"

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

    # Query hash
    q_hash = hashlib.md5(query.encode()).hexdigest()[:8]

    # Check for trivial query (Problem 1 / Fix B)
    trivial = is_trivial(query)

    # Classify query → raw face weights
    raw_weights = classify_query(query, history or state.routing_history)

    # Apply face_bias from state (learned adaptation)
    adjusted = [raw_weights[i] + state.face_bias[i] for i in range(NUM_FACES)]
    adjusted = [max(w, 0.02) for w in adjusted]
    total = sum(adjusted)
    adjusted = [w / total for w in adjusted]

    # Project onto Chestahedron manifold
    manifold_weights = project_to_manifold(adjusted)

    # Compute coherence
    coh = coherence_score(adjusted)

    # Face activations at temperature=1.0 for routing (sharper discrimination)
    activations = face_activations(adjusted, temperature=1.0)

    # Adjacency check with very sharp activations (temperature=0.5)
    adj_activations = face_activations(adjusted, temperature=0.5)
    adj_pen = adjacency_penalty(adj_activations)
    adj_ok = adj_pen < 0.18  # Recalibrated threshold (Problem 4)

    # Determine max tools (Problem 1)
    max_tools = _compute_max_tools(raw_weights, query)

    # Check for "no tools needed" (Problem 1)
    max_raw = max(raw_weights)
    tools_needed = True
    if trivial:
        tools_needed = False
        max_tools = 0
    elif max_raw < 0.20 and len(query) < 30 and sum(1 for s in raw_weights if s > 0.02) <= 2:
        tools_needed = False
        max_tools = 0

    # Past-reference detection (Fix C)
    past_ref = has_past_reference(query)

    # Tool scoring with raised threshold (Problem 1)
    tool_sequence = []
    if tools_needed:
        tool_sequence_raw = _score_tools(
            activations, tools, state.tool_success,
            query=query, has_past_ref=past_ref,
        )
        # Filter: minimum score 0.08 (Problem 1)
        tool_sequence = [t for t in tool_sequence_raw if t["score"] > 0.08][:max_tools]
        for t in tool_sequence:
            del t["score"]

    # Constraints from RAW weights (Problem 5)
    constraints = _build_constraints(raw_weights)

    # Context shape
    ctx_shape = _context_shape(activations)

    # Build operation weights dict (use routing-temperature activations)
    op_weights = {FACE_NAMES[i]: round(activations[i], 4) for i in range(NUM_FACES)}

    # Dominant operations
    dom_ops = dominant_faces(activations, threshold=0.10)

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

    result = {
        "step": state.step,
        "query_hash": q_hash,
        "operation_weights": op_weights,
        "dominant_operations": dom_ops,
        "tools_needed": tools_needed,
        "tool_sequence": tool_sequence,
        "constraints": constraints,
        "context_shape": ctx_shape,
        "coherence": round(coh, 4),
        "adjacency_ok": adj_ok,
        "manifold_distance": round(1.0 - coh, 4),
    }
    return result


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
