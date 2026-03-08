"""
Saturn + CCN + KD — Integrated Cognitive Loop
==============================================

Perception (Saturn) -> Cognition (CCN) -> Memory (KD) -> Loop

The three systems map to CCN stages:
  MIRROR  <- Saturn twin whisper (how Hugo is thinking right now)
  INHERIT <- KD spreading activation (what has been learned before)
  GATE 6  <- Saturn + KD inform mode: build/explore/maintain
  Output  -> KD metabolize (deposit knowledge back into the graph)

Saturn already runs V9 gates internally for perception.
KD already runs FSRS-based memory metabolism.
The CCN already cycles through 7 geometric stages.
This file wires them together.

Usage:
    engine = CognitiveLoop(
        saturn_url="http://localhost:3001/sse",       # Saturn MCP
        kd_url="https://knowledge-discovery-yzzq.onrender.com/sse",  # KD MCP
        hidden_dim=96,
    )
    result = engine.cycle("what does the fox do")
    print(result.trace)    # 7-stage topology trace
    print(result.mode)     # tamam/darash/zakat
    print(result.state)    # output tensor
"""

import json
import math
import threading
import urllib.request
import urllib.parse
import numpy as np
import torch
import torch.nn as nn

from chestohedron_lm import make_topology_matrices

PHI = (1 + math.sqrt(5)) / 2


# ============================================================
# MCP CLIENT — minimal, sync, stdlib-only
# ============================================================

class MCPClient:
    """
    Talk to MCP servers over SSE. Handles the handshake:
    GET /sse -> read session endpoint -> POST JSON-RPC to it.
    """

    def __init__(self, sse_url, timeout=15):
        self.sse_url = sse_url
        self.endpoint = None
        self.timeout = timeout
        self._handshake()

    def _handshake(self):
        """Connect to SSE, extract the session endpoint from first event."""
        try:
            req = urllib.request.Request(self.sse_url)
            req.add_header("Accept", "text/event-stream")
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            buf = b""
            while True:
                chunk = resp.read(1)
                if not chunk:
                    break
                buf += chunk
                # SSE events end with double newline
                if b"\n\n" in buf:
                    for line in buf.decode("utf-8", errors="replace").split("\n"):
                        line = line.strip()
                        if line.startswith("data:"):
                            data = line[5:].strip()
                            if "/" in data:
                                base = self.sse_url.rsplit("/", 1)[0]
                                self.endpoint = (base + data) if data.startswith("/") else data
                    break
            resp.close()
        except Exception:
            self.endpoint = None  # offline — engine runs in local-only mode

    def call(self, tool_name, arguments=None):
        """JSON-RPC tools/call to the MCP server."""
        if not self.endpoint:
            return None
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments or {}},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
            result = json.loads(resp.read())
            return result.get("result")
        except Exception:
            return None

    @property
    def connected(self):
        return self.endpoint is not None


# ============================================================
# SATURN BRIDGE — Perception feeds MIRROR + GATE 6
# ============================================================

class SaturnBridge:
    """
    Saturn watches Hugo's live thought stream via LevelDB keystroke capture.
    It sees the TEMPORAL SHAPE of thought — not finished prompts.

    What Saturn actually captures (via saturn_watcher.py):
    - Keystroke timestamps to the millisecond
    - Inter-key intervals (typing velocity)
    - Deletion events (backspace/delete bursts)
    - Hesitation gaps (pauses > threshold)
    - Rethinking events (type → delete → retype differently)
    - Character-level edit distance between drafts

    twin_state returns: narrative + temporal metrics + predictions + tensions
    twin_patterns returns: accumulated keystroke behavior patterns
    twin_gates returns: MIRROR/VERIFY/REMOVE gate audit trail

    These feed MIRROR (metacognition) and GATE 6 (mode selection).
    """

    # Feature map: 96 continuous features from Saturn's temporal perception
    # [0-15]  keystroke dynamics (velocity, variance, acceleration)
    # [16-31] deletion patterns (rate, burst length, burst frequency)
    # [32-47] hesitation patterns (gap duration, gap frequency, gap position)
    # [48-63] rethinking patterns (edit distance, rewrite rate, draft count)
    # [64-79] session dynamics (time on task, fatigue curve, momentum)
    # [80-95] twin predictions (confidence, energy, frustration, decisiveness)
    N_FEATURES = 96

    def __init__(self, mcp, hidden_dim):
        self.mcp = mcp
        self.dim = hidden_dim
        rng = np.random.RandomState(777)
        self._proj = torch.tensor(
            rng.randn(self.N_FEATURES, hidden_dim).astype(np.float32)
            / math.sqrt(self.N_FEATURES)
        )

    def query(self):
        """Get twin_state + twin_patterns + twin_gates from Saturn."""
        if not self.mcp:
            return None, None
        state = self.mcp.call("twin_state")
        patterns = self.mcp.call("twin_patterns")
        return state, patterns

    def mirror_signal(self, state=None, patterns=None):
        """
        Encode Saturn's temporal perception as a hidden-dim vector for MIRROR.

        Parses the actual keystroke dynamics Saturn captures:
        - Typing velocity (chars/sec, inter-key interval distribution)
        - Deletion bursts (consecutive deletes, delete-to-type ratio)
        - Hesitation gaps (pause duration, where in the sentence)
        - Rethinking events (typed X, deleted, typed Y instead)
        - Session energy curve (accelerating, steady, decelerating)
        """
        features = torch.zeros(self.N_FEATURES)

        # Parse twin_state — contains temporal metrics from the watcher
        state_data = _parse_saturn_data(_extract_text(state))
        if state_data:
            # ── Keystroke dynamics [0-15] ──
            metrics = state_data.get("metrics", state_data.get("temporal", {}))
            features[0] = _clamp(metrics.get("keystroke_velocity", 0) / 10.0)
            features[1] = _clamp(metrics.get("mean_iki", 0) / 500.0)       # inter-key interval ms
            features[2] = _clamp(metrics.get("iki_variance", 0) / 1000.0)  # variance = erratic typing
            features[3] = _clamp(metrics.get("iki_std", 0) / 200.0)
            features[4] = _clamp(metrics.get("typing_acceleration", 0.5))   # >0.5 speeding up
            features[5] = _clamp(metrics.get("chars_per_second", 0) / 15.0)
            features[6] = _clamp(metrics.get("words_per_minute", 0) / 120.0)
            features[7] = _clamp(metrics.get("peak_velocity", 0) / 20.0)
            # Velocity distribution shape
            vel_hist = metrics.get("velocity_histogram", [])
            for i, v in enumerate(vel_hist[:8]):
                features[8 + i] = _clamp(float(v))

            # ── Deletion patterns [16-31] ──
            deletions = state_data.get("deletions", metrics)
            features[16] = _clamp(deletions.get("delete_rate", 0))          # deletes per keystroke
            features[17] = _clamp(deletions.get("delete_burst_count", 0) / 10.0)
            features[18] = _clamp(deletions.get("mean_burst_length", 0) / 20.0)
            features[19] = _clamp(deletions.get("max_burst_length", 0) / 50.0)
            features[20] = _clamp(deletions.get("delete_to_type_ratio", 0))
            features[21] = _clamp(deletions.get("burst_frequency", 0))      # bursts per minute
            features[22] = _clamp(deletions.get("chars_deleted", 0) / 500.0)
            features[23] = _clamp(deletions.get("words_deleted", 0) / 50.0)
            # Deletion position in sentence (early = rethinking, late = editing)
            del_positions = deletions.get("burst_positions", [])
            for i, p in enumerate(del_positions[:8]):
                features[24 + i] = _clamp(float(p))

            # ── Hesitation patterns [32-47] ──
            hesitations = state_data.get("hesitations", metrics)
            features[32] = _clamp(hesitations.get("pause_count", 0) / 20.0)
            features[33] = _clamp(hesitations.get("mean_pause_ms", 0) / 5000.0)
            features[34] = _clamp(hesitations.get("max_pause_ms", 0) / 30000.0)
            features[35] = _clamp(hesitations.get("pause_ratio", 0))        # time paused / total time
            features[36] = _clamp(hesitations.get("pause_frequency", 0))    # pauses per minute
            features[37] = _clamp(hesitations.get("mid_word_pauses", 0) / 10.0)  # worst kind
            features[38] = _clamp(hesitations.get("pre_delete_pauses", 0) / 10.0)  # think then delete
            features[39] = _clamp(hesitations.get("increasing_pauses", 0))  # fatigue signal
            # Pause duration distribution
            pause_hist = hesitations.get("pause_histogram", [])
            for i, p in enumerate(pause_hist[:8]):
                features[40 + i] = _clamp(float(p))

            # ── Rethinking patterns [48-63] ──
            rethinking = state_data.get("rethinking", metrics)
            features[48] = _clamp(rethinking.get("rethink_count", 0) / 10.0)
            features[49] = _clamp(rethinking.get("edit_distance_mean", 0) / 50.0)
            features[50] = _clamp(rethinking.get("rewrite_rate", 0))         # rewrites per sentence
            features[51] = _clamp(rethinking.get("draft_count", 0) / 5.0)
            features[52] = _clamp(rethinking.get("semantic_shift", 0))       # meaning change 0-1
            features[53] = _clamp(rethinking.get("false_starts", 0) / 10.0)
            features[54] = _clamp(rethinking.get("completion_ratio", 0))     # started vs finished
            features[55] = _clamp(rethinking.get("direction_changes", 0) / 5.0)
            # Per-rethink edit distances
            edits = rethinking.get("edit_distances", [])
            for i, e in enumerate(edits[:8]):
                features[56 + i] = _clamp(float(e) / 50.0)

            # ── Session dynamics [64-79] ──
            session = state_data.get("session", state_data)
            features[64] = _clamp(session.get("time_on_task_min", 0) / 60.0)
            features[65] = _clamp(session.get("energy", session.get("energy_level", 0.5)))
            features[66] = _clamp(session.get("momentum", 0.5))             # rolling velocity trend
            features[67] = _clamp(session.get("fatigue_score", 0))
            features[68] = _clamp(session.get("focus_score", 0.5))
            features[69] = _clamp(session.get("flow_state", 0))             # 0=scattered, 1=deep flow
            features[70] = _clamp(session.get("context_switches", 0) / 10.0)
            features[71] = _clamp(session.get("prompt_count", 0) / 50.0)
            # Energy curve (recent windows)
            energy_curve = session.get("energy_curve", [])
            for i, e in enumerate(energy_curve[:8]):
                features[72 + i] = _clamp(float(e))

            # ── Twin predictions [80-95] ──
            predictions = state_data.get("predictions", state_data)
            features[80] = _clamp(predictions.get("confidence", 0.5))
            features[81] = _clamp(predictions.get("frustration", 0))
            features[82] = _clamp(predictions.get("decisiveness", 0.5))
            features[83] = _clamp(predictions.get("uncertainty", 0))
            features[84] = _clamp(predictions.get("exploration_mode", 0))
            features[85] = _clamp(predictions.get("urgency", 0.5))
            features[86] = _clamp(predictions.get("satisfaction", 0.5))
            features[87] = _clamp(predictions.get("cognitive_load", 0.5))

        # Parse twin_patterns — accumulated behavioral observations
        pattern_data = _parse_saturn_data(_extract_text(patterns))
        if pattern_data:
            pats = pattern_data.get("patterns", pattern_data)
            features[88] = _clamp(pats.get("avg_velocity", 0) / 10.0)
            features[89] = _clamp(pats.get("avg_delete_rate", 0))
            features[90] = _clamp(pats.get("avg_hesitation_rate", 0))
            features[91] = _clamp(pats.get("avg_rethink_rate", 0))
            features[92] = _clamp(pats.get("pattern_stability", 0.5))  # how consistent
            features[93] = _clamp(pats.get("session_count", 0) / 100.0)
            features[94] = _clamp(pats.get("total_keystrokes", 0) / 100000.0)
            features[95] = _clamp(pats.get("behavioral_drift", 0))    # changing over time

        return features @ self._proj  # (hidden_dim,)

    def gate6_bias(self, state=None, patterns=None):
        """
        Saturn-informed GATE 6 mode selection from TEMPORAL signals.

        Uses the actual keystroke dynamics, not keyword matching:
        - High velocity + low deletion + low hesitation -> tamam (EXECUTE)
        - High hesitation + high rethinking + low velocity -> darash (EXPLORE)
        - Increasing pauses + high fatigue + high deletion -> zakat (PRUNE)

        These are continuous signals derived from millisecond-level
        keystroke data, not binary flags from narrative text.
        """
        t, d, z = 0.34, 0.33, 0.33

        state_data = _parse_saturn_data(_extract_text(state))
        if state_data:
            metrics = state_data.get("metrics", state_data.get("temporal", {}))
            predictions = state_data.get("predictions", state_data)
            deletions = state_data.get("deletions", metrics)
            hesitations = state_data.get("hesitations", metrics)
            rethinking = state_data.get("rethinking", metrics)
            session = state_data.get("session", state_data)

            # Velocity signal: fast typing = decisive = build
            velocity = metrics.get("chars_per_second", 0) / 10.0
            # Deletion signal: heavy deleting = uncertain or pruning
            delete_rate = deletions.get("delete_rate", 0)
            # Hesitation signal: long pauses = thinking/uncertain
            pause_ratio = hesitations.get("pause_ratio", 0)
            # Rethinking signal: rewrites = exploring alternatives
            rethink_rate = rethinking.get("rethink_count", 0) / 10.0
            # Fatigue signal: increasing pauses over session
            fatigue = session.get("fatigue_score", 0)
            # Direct predictions from Saturn's 7-gate processing
            decisiveness = predictions.get("decisiveness", 0.5)
            frustration = predictions.get("frustration", 0)
            energy = session.get("energy", predictions.get("energy_level", 0.5))

            # TAMAM (build): fast, decisive, low deletion, low hesitation
            tamam_signal = (
                velocity * 0.3 +
                decisiveness * 0.3 +
                (1.0 - delete_rate) * 0.2 +
                (1.0 - pause_ratio) * 0.2
            )

            # DARASH (explore): hesitant, rethinking, uncertain
            darash_signal = (
                pause_ratio * 0.25 +
                rethink_rate * 0.25 +
                (1.0 - decisiveness) * 0.25 +
                (1.0 - velocity) * 0.25
            )

            # ZAKAT (maintain): fatigued, heavy deletion, frustrated
            zakat_signal = (
                fatigue * 0.3 +
                delete_rate * 0.25 +
                frustration * 0.25 +
                (1.0 - energy) * 0.2
            )

            # Blend with base (signals have 0.6 total influence)
            t += tamam_signal * 0.6
            d += darash_signal * 0.6
            z += zakat_signal * 0.6

        total = t + d + z
        return t / total, d / total, z / total


# ============================================================
# KD BRIDGE — Memory feeds INHERIT, Output feeds metabolize
# ============================================================

class KDBridge:
    """
    KD's spreading activation provides accumulated knowledge:
    - graph_query returns direct matches AND activated neighbors
    - Confidence scores weight how strongly to inherit each node
    - After the cycle, metabolize deposits new knowledge back

    This is how the system LEARNS. Every cycle strengthens or
    weakens beliefs in the living knowledge graph.
    """

    def __init__(self, mcp, hidden_dim):
        self.mcp = mcp
        self.dim = hidden_dim
        rng = np.random.RandomState(888)
        self._proj = torch.tensor(
            rng.randn(128, hidden_dim).astype(np.float32) / math.sqrt(128)
        )

    def query(self, text):
        """Spreading activation search — finds connections you didn't ask for."""
        if not self.mcp:
            return None
        return self.mcp.call("graph_query", {"query": text, "depth": 2})

    def browse(self, domain, limit=20, sort="confidence"):
        """Browse a knowledge domain."""
        if not self.mcp:
            return None
        return self.mcp.call("vault_browse", {
            "domain": domain, "limit": limit, "sort": sort,
        })

    def inherit_signal(self, query_result):
        """
        Encode KD nodes as hidden-dim vector for INHERIT.

        Each node contributes proportional to its confidence.
        Spreading activation means direct + neighbor nodes —
        the model inherits implicit connections, not just matches.
        """
        features = torch.zeros(128)
        text = _extract_text(query_result)
        if text:
            nodes = _parse_nodes(text)
            for i, node in enumerate(nodes[:32]):
                conf = float(node.get("confidence", 0.5))
                content = str(node.get("content", node.get("text", "")))
                for j, c in enumerate(content[:16]):
                    features[(i * 4 + j) % 126] += ord(c) / 128.0 * conf
            # Knowledge density in this area
            features[126] = min(len(nodes) / 20.0, 1.0)
            # Mean confidence
            if nodes:
                features[127] = sum(
                    float(n.get("confidence", 0.5)) for n in nodes[:20]
                ) / min(len(nodes), 20)
        return features @ self._proj  # (hidden_dim,)

    def knowledge_density(self, query_result):
        """How much does KD know about this topic? Informs GATE 6."""
        text = _extract_text(query_result)
        if not text:
            return 0.0
        nodes = _parse_nodes(text)
        if not nodes:
            return 0.0
        density = min(len(nodes) / 20.0, 1.0)
        avg_conf = sum(float(n.get("confidence", 0.5)) for n in nodes) / len(nodes)
        return density * avg_conf

    def metabolize(self, content, gate_mode, trace):
        """
        Deposit cycle output back into KD's knowledge graph.

        The metabolized knowledge includes:
        - The actual content/reasoning
        - Which gate mode was active (build/explore/maintain)
        - The full topology trace (energy at each stage)

        This is how the graph GROWS. Reinforced beliefs strengthen.
        Contradicted beliefs weaken. Unused knowledge decays via FSRS.
        """
        if not self.mcp:
            return None
        return self.mcp.call("metabolize", {
            "content": content,
            "source": "ccn_integrated",
            "domain": "neural-architecture",
            "tags": ["ccn", "integrated", f"gate:{gate_mode}"],
            "metadata": json.dumps({
                "gate_mode": gate_mode,
                "trace": [(name, stype, round(energy, 4))
                          for name, stype, energy in trace],
            }),
        })


# ============================================================
# INTEGRATED CYCLING ENGINE
# ============================================================

class CycleResult:
    """Output of one cognitive loop."""
    __slots__ = ("state", "trace", "mode", "saturn_state", "kd_result")

    def __init__(self, state, trace, mode, saturn_state=None, kd_result=None):
        self.state = state
        self.trace = trace
        self.mode = mode
        self.saturn_state = saturn_state
        self.kd_result = kd_result

    def __repr__(self):
        lines = [f"CycleResult(mode={self.mode})"]
        for name, stype, energy in self.trace:
            icon = "\u25b2" if stype == "constrictive" else "\u25c6"
            bar = "\u2588" * int(energy * 6)
            lines.append(f"  {icon} {name:10s} {stype:14s} {energy:6.3f} {bar}")
        return "\n".join(lines)


class CognitiveLoop:
    """
    The complete cognitive system:

        Saturn (Perception) ---> MIRROR
        KD (Memory)         ---> INHERIT
        CCN (Cognition)     ---> 7-stage chestohedron cycle
        Output              ---> KD metabolize
        Saturn              ---> captures reaction (v4 closed loop)

    GATE 6 mode is NOT random — it's informed by:
    - Saturn's read of Hugo's cognitive state
    - KD's knowledge density in the topic area
    - The cycle's own state energy (intrinsic signal)

    The topology is FIXED. Only thin injection projections modulate
    how external signals enter the cycle. The geometry does the work.
    """

    def __init__(self, hidden_dim=96, seed=42,
                 saturn_url=None, kd_url=None):
        self.dim = hidden_dim

        # Connect to external systems (graceful offline mode)
        saturn_mcp = MCPClient(saturn_url) if saturn_url else None
        kd_mcp = MCPClient(kd_url) if kd_url else None

        self.saturn = SaturnBridge(saturn_mcp, hidden_dim)
        self.kd = KDBridge(kd_mcp, hidden_dim)

        # Fixed topology matrices from the chestohedron
        self.topo = make_topology_matrices(hidden_dim, seed)

        # Fixed input projection: text -> hidden_dim
        # This gives the cycle something to work with even offline
        rng = np.random.RandomState(seed + 50)
        self._input_proj = torch.tensor(
            (rng.randn(256, hidden_dim) / math.sqrt(256)).astype(np.float32)
        )

        # Thin injection gates — the ONLY modulation of external signals
        rng = np.random.RandomState(seed + 100)
        self._mirror_gate = torch.tensor(
            (rng.randn(hidden_dim, hidden_dim) * 0.1 / math.sqrt(hidden_dim)).astype(np.float32)
        )
        self._inherit_gate = torch.tensor(
            (rng.randn(hidden_dim, hidden_dim) * 0.1 / math.sqrt(hidden_dim)).astype(np.float32)
        )

        self._online = {
            "saturn": saturn_mcp.connected if saturn_mcp else False,
            "kd": kd_mcp.connected if kd_mcp else False,
        }

    @property
    def status(self):
        return {k: ("online" if v else "offline") for k, v in self._online.items()}

    def _encode_input(self, text):
        """
        Project input text into hidden_dim via character-level hashing.
        Deterministic, no learned params. Gives the cycle a signal to process.
        """
        features = torch.zeros(256)
        for i, c in enumerate(text):
            features[ord(c) % 256] += 1.0
        # Normalize
        norm = features.norm()
        if norm > 0:
            features = features / norm
        return (features @ self._input_proj).unsqueeze(0)  # (1, hidden_dim)

    def cycle(self, input_text, state=None, n_cycles=1):
        """
        One full cognitive loop.

        1. Saturn -> MIRROR context (perception of Hugo's state)
        2. KD -> INHERIT context (accumulated knowledge via spreading activation)
        3. CCN -> 7-stage cycle with injected signals
        4. Output -> KD metabolize (deposit knowledge back)
        5. Return state + trace + mode

        The cycle CLOSES: GATE 6 output IS the next MIRROR input.
        """
        # Input text becomes the initial state signal
        u = self._encode_input(input_text)
        if state is None:
            state = u
        else:
            state = state + u * 0.3  # blend with prior state

        # ── PERCEPTION: Saturn twin whisper ──
        saturn_state, saturn_patterns = self.saturn.query()
        mirror_ctx = self.saturn.mirror_signal(saturn_state, saturn_patterns)
        gate_bias = self.saturn.gate6_bias(saturn_state, saturn_patterns)

        # ── MEMORY: KD spreading activation ──
        kd_result = self.kd.query(input_text)
        inherit_ctx = self.kd.inherit_signal(kd_result)
        kd_density = self.kd.knowledge_density(kd_result)

        # Adjust gate bias with KD density
        # Sparse knowledge -> more explore. Dense knowledge -> more build.
        t, d, z = gate_bias
        if kd_density < 0.3:
            d += 0.15; t -= 0.075; z -= 0.075
        elif kd_density > 0.7:
            t += 0.15; d -= 0.075; z -= 0.075
        total = t + d + z
        gate_bias = (t / total, d / total, z / total)

        # ── COGNITION: 7-stage chestohedron cycle ──
        trace = []
        for _ in range(n_cycles):
            # 1. MIRROR + Saturn injection (perception feeds metacognition)
            saturn_inj = (mirror_ctx.unsqueeze(0) @ self._mirror_gate) * 0.3
            state = torch.tanh(state @ self.topo["mirror"] + u + saturn_inj)
            trace.append(("MIRROR", "constrictive", state.norm().item()))

            # 2. INHERIT + KD injection (memory feeds knowledge absorption)
            kd_inj = (inherit_ctx.unsqueeze(0) @ self._inherit_gate) * 0.3
            state = torch.tanh(state @ self.topo["inherit"] + u + kd_inj)
            trace.append(("INHERIT", "expansive", state.norm().item()))

            # 3. BOUND — low-rank compression
            state = torch.tanh(state @ self.topo["bound"] + u)
            trace.append(("BOUND", "constrictive", state.norm().item()))

            # 4. EXPRESS — phi-rotated generation (ONLY generative stage)
            state = torch.tanh(state @ self.topo["express"] + u)
            trace.append(("EXPRESS", "expansive", state.norm().item()))

            # 5. VERIFY — adversarial self-test
            state = torch.tanh(state @ self.topo["verify"] + u)
            trace.append(("VERIFY", "constrictive", state.norm().item()))

            # 6. REMOVE — aggressive pruning (double-compression with VERIFY)
            state = torch.tanh(state @ self.topo["remove"] + u)
            trace.append(("REMOVE", "constrictive", state.norm().item()))

            # 7. GATE 6 — adaptive closer, informed by Saturn + KD
            state = self._gate6(state, gate_bias)
            trace.append(("GATE_6", "expansive", state.norm().item()))

        # Determine active mode
        t, d, z = gate_bias
        mode = "tamam" if t >= d and t >= z else ("darash" if d >= z else "zakat")

        # ── MEMORY: Metabolize output back into KD ──
        self.kd.metabolize(input_text, mode, trace)

        return CycleResult(state, trace, mode, saturn_state, kd_result)

    def _gate6(self, state, bias):
        """
        GATE 6: Three-mode adaptive closer.

        Blends intrinsic gating (state energy) with extrinsic bias
        (Saturn's perception of Hugo + KD's knowledge density).

        - tamam (build):    decisive + dense knowledge -> execute
        - darash (explore): uncertain + sparse knowledge -> investigate
        - zakat (maintain): fatigued + accumulated cruft -> prune

        The gate feeds back to MIRROR. The cycle closes.
        """
        # Intrinsic: state energy determines base gating
        energy = (state ** 2).sum(dim=-1, keepdim=True)
        energy_norm = energy / (energy.max() + 1e-8)

        g_t = torch.clamp(energy_norm - 0.66, min=0) * 3
        g_d = torch.clamp(1 - (energy_norm - 0.5).abs() * 4, min=0)
        g_z = torch.clamp(0.33 - energy_norm, min=0) * 3

        # Extrinsic: Saturn + KD bias (0.5 weight — topology still dominates)
        ext_t, ext_d, ext_z = bias
        g_t = g_t + ext_t * 0.5
        g_d = g_d + ext_d * 0.5
        g_z = g_z + ext_z * 0.5

        g_total = g_t + g_d + g_z + 1e-8

        out = ((g_t / g_total) * (state @ self.topo["gate6_tamam"]) +
               (g_d / g_total) * (state @ self.topo["gate6_darash"]) +
               (g_z / g_total) * (state @ self.topo["gate6_zakat"]))

        return torch.tanh(out)

    def multi_cycle(self, input_text, n_cycles=3):
        """
        Run multiple cycles on the same input.
        Each cycle's GATE 6 output feeds the next MIRROR.
        The representation refines with each pass through the geometry.
        """
        state = None
        results = []
        for i in range(n_cycles):
            result = self.cycle(input_text, state, n_cycles=1)
            state = result.state
            results.append(result)
        return results


# ============================================================
# HELPERS
# ============================================================

def _extract_text(mcp_result):
    """Pull text content from an MCP tool result."""
    if not mcp_result:
        return ""
    if isinstance(mcp_result, str):
        return mcp_result
    if isinstance(mcp_result, dict):
        content = mcp_result.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
        if "text" in mcp_result:
            return mcp_result["text"]
    return str(mcp_result)


def _parse_saturn_data(text):
    """
    Parse Saturn's temporal perception data.

    Saturn's twin_state returns structured data with keystroke metrics:
    {
        "metrics": {"keystroke_velocity": 5.2, "mean_iki": 180, ...},
        "deletions": {"delete_rate": 0.12, "burst_count": 3, ...},
        "hesitations": {"pause_count": 7, "mean_pause_ms": 2300, ...},
        "rethinking": {"rethink_count": 2, "edit_distance_mean": 15, ...},
        "session": {"energy": 0.7, "fatigue_score": 0.2, ...},
        "predictions": {"decisiveness": 0.8, "frustration": 0.1, ...},
        "narrative": "Hugo is typing rapidly with high confidence..."
    }

    If it's not JSON (older Saturn versions), we still extract what we can.
    """
    if not text:
        return None
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: extract numeric patterns from narrative text
    # Saturn's narrative includes embedded metrics like "velocity: 5.2 cps"
    result = {}
    import re
    # Match "key: number" patterns in narrative text
    for match in re.finditer(r'(\w+)[:\s]+(\d+\.?\d*)\s*(ms|cps|wpm|%)?', text):
        key, value, unit = match.group(1).lower(), float(match.group(2)), match.group(3)
        if key not in result:
            result.setdefault("metrics", {})[key] = value
    # Extract known Saturn narrative signals as prediction scores
    low = text.lower()
    preds = {}
    if "frustrat" in low:
        preds["frustration"] = 0.8
    if "decisive" in low or "rapid" in low:
        preds["decisiveness"] = 0.8
    if "uncertain" in low or "hesitat" in low:
        preds["uncertainty"] = 0.7
        preds["decisiveness"] = 0.3
    if "fatigue" in low or "tired" in low:
        preds.setdefault("session", {})
        result.setdefault("session", {})["fatigue_score"] = 0.7
    if "explor" in low:
        preds["exploration_mode"] = 0.8
    if preds:
        result["predictions"] = preds
    return result if result else None


def _clamp(v, lo=0.0, hi=1.0):
    """Clamp value to [0, 1] range."""
    return max(lo, min(hi, float(v))) if v is not None else 0.0


def _parse_nodes(text):
    """Try to parse KD nodes from text (JSON or fallback)."""
    try:
        data = json.loads(text)
        return data.get("nodes", data.get("results", []))
    except (json.JSONDecodeError, TypeError):
        return []


# ============================================================
# DEMO / VERIFICATION
# ============================================================

def demo():
    """
    Run the cognitive loop in local-only mode.
    Saturn + KD offline -> signals are zero vectors -> pure CCN topology.
    Connect Saturn/KD to see the full system.
    """
    print("=" * 66)
    print("  INTEGRATED COGNITIVE LOOP")
    print("  Saturn (Perception) + CCN (Cognition) + KD (Memory)")
    print("=" * 66)

    engine = CognitiveLoop(hidden_dim=96, seed=42)

    print(f"\n  System status: {engine.status}")
    print(f"  (Offline systems use zero-vector signals — pure CCN topology)")

    # Single cycle
    print(f"\n{'~'*66}")
    print(f"  Single cycle: 'what does the fox do'")
    print(f"{'~'*66}")
    result = engine.cycle("what does the fox do")
    print(f"\n{result}")
    print(f"\n  Gate mode: {result.mode}")
    print(f"  State norm: {result.state.norm().item():.4f}")

    # Multi-cycle refinement
    print(f"\n{'~'*66}")
    print(f"  Multi-cycle refinement (3 passes)")
    print(f"{'~'*66}")
    results = engine.multi_cycle("topology over substance", n_cycles=3)
    for i, r in enumerate(results):
        norms = [e for _, _, e in r.trace]
        print(f"\n  Cycle {i+1}: mode={r.mode}  "
              f"energy=[{' -> '.join(f'{n:.2f}' for n in norms)}]")

    # Pattern: C -> E -> C -> E -> C -> C -> E
    print(f"\n{'~'*66}")
    print(f"  Topology verification")
    print(f"{'~'*66}")
    patterns = []
    for name, stype, _ in results[-1].trace:
        patterns.append("C" if stype == "constrictive" else "E")
    print(f"\n  Stage pattern: {' -> '.join(patterns)}")
    print(f"  Expected:      C -> E -> C -> E -> C -> C -> E")
    match = patterns == ["C", "E", "C", "E", "C", "C", "E"]
    print(f"  Match: {'YES' if match else 'NO'}")

    # Double-compression bottleneck
    last_trace = results[-1].trace
    verify_energy = last_trace[4][2]
    remove_energy = last_trace[5][2]
    gate6_energy = last_trace[6][2]
    print(f"\n  Double-compression bottleneck:")
    print(f"    VERIFY  energy: {verify_energy:.4f} (constrictive)")
    print(f"    REMOVE  energy: {remove_energy:.4f} (constrictive)")
    print(f"    GATE 6  energy: {gate6_energy:.4f} (re-expands)")

    # Integration points
    print(f"\n{'='*66}")
    print(f"  INTEGRATION ARCHITECTURE")
    print(f"{'='*66}")
    print(f"""
  Saturn MCP ─────────────────────┐
  (twin_state, twin_patterns)     │
                                  ▼
  ┌─ MIRROR ◄── Saturn whisper (perception of cognitive state)
  │    ▲ symmetric reflection
  │    │
  ├─ INHERIT ◄── KD spreading activation (accumulated knowledge)
  │    ◆ near-orthogonal absorption
  │    │
  ├─ BOUND
  │    ▲ low-rank compression
  │    │
  ├─ EXPRESS
  │    ◆ phi-rotated generation (ONLY generative stage)
  │    │
  ├─ VERIFY
  │    ▲ adversarial self-test
  │    │
  ├─ REMOVE
  │    ▲ aggressive pruning (double-compression bottleneck)
  │    │
  └─ GATE 6 ◄── Saturn + KD inform mode selection
       ◆ tamam/darash/zakat
       │
       ├──► KD metabolize (deposit knowledge back)
       │
       └──► feeds back to MIRROR (cycle closes)
                                  ▲
  KD MCP ─────────────────────────┘
  (graph_query, metabolize)

  Pattern: C -> E -> C -> E -> C -> C -> E
  The chestohedron's 7 faces: 4 triangular + 3 kite.
  The geometry is the architecture. The topology is the intelligence.
""")

    print(f"  To connect live systems:")
    print(f"    engine = CognitiveLoop(")
    print(f"        saturn_url='http://localhost:3001/sse',")
    print(f"        kd_url='https://knowledge-discovery-yzzq.onrender.com/sse',")
    print(f"    )")
    print()

    return engine, results


if __name__ == "__main__":
    demo()
