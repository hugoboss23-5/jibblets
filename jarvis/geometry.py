"""
Chestahedron 7D Topology — Pure Python, stdlib only.

The mathematical foundation of JARVIS routing. Implements the 7-face
Chestahedron (4 equilateral triangles + 3 kite quadrilaterals) in 3D,
extends to 7D via phi-scaled basis expansion, and provides all geometric
operations needed for routing: projection, coherence, face activations,
adjacency enforcement.

Zero external dependencies. Uses math module only.
"""

import math
import sys

PHI = (1 + math.sqrt(5)) / 2  # 1.618033988749895
PHI_ANGLE = math.radians(36)  # The phi angle
NUM_FACES = 7

FACE_NAMES = ["DIAGNOSE", "LOAD", "CONSTRAIN", "GENERATE", "BREAK", "CUT", "ROUTE"]

FACE_DESC = {
    "DIAGNOSE": "Read the situation. What is actually being asked?",
    "LOAD": "Pull context. What's already known? Check KD, past chats, memory.",
    "CONSTRAIN": "Set boundaries. What should NOT be done? What's out of scope?",
    "GENERATE": "Produce candidates. Multiple approaches, not just the first.",
    "BREAK": "Stress-test. Where does this fail? What's missing?",
    "CUT": "Select and compress. Pick the best, kill the rest.",
    "ROUTE": "Dispatch. Which tools, what order, what framing.",
}

# --- 12 edges of the Chestahedron face adjacency ---
EDGES = [
    (0, 1), (0, 2), (0, 3),   # DIAGNOSE connects to LOAD, CONSTRAIN, GENERATE
    (1, 4), (1, 5),            # LOAD connects to BREAK, CUT
    (2, 5), (2, 6),            # CONSTRAIN connects to CUT, ROUTE
    (3, 4), (3, 6),            # GENERATE connects to BREAK, ROUTE
    (4, 5), (4, 6),            # BREAK connects to CUT, ROUTE
    (5, 6),                    # CUT connects to ROUTE
]


# ---- Vector math (no numpy) ----

def dot(a: list[float], b: list[float]) -> float:
    """Dot product of two vectors."""
    return sum(ai * bi for ai, bi in zip(a, b))


def norm(v: list[float]) -> float:
    """L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def normalize(v: list[float]) -> list[float]:
    """Normalize to unit vector. Returns zero vector if input is zero."""
    n = norm(v)
    if n < 1e-12:
        return [0.0] * len(v)
    return [x / n for x in v]


def vec_add(a: list[float], b: list[float]) -> list[float]:
    return [ai + bi for ai, bi in zip(a, b)]


def vec_scale(v: list[float], s: float) -> list[float]:
    return [x * s for x in v]


def softmax(v: list[float], temperature: float = PHI) -> list[float]:
    """Softmax with phi-scaled temperature."""
    # Subtract max for numerical stability
    max_v = max(v)
    exps = [math.exp((x - max_v) / temperature) for x in v]
    total = sum(exps)
    if total < 1e-12:
        return [1.0 / len(v)] * len(v)
    return [e / total for e in exps]


def mat_vec(matrix: list[list[float]], vec: list[float]) -> list[float]:
    """Matrix-vector multiply. matrix[i][j] * vec[j] -> result[i]."""
    return [dot(row, vec) for row in matrix]


# ---- Chestahedron Construction ----

def build_vertices_3d() -> list[list[float]]:
    """
    Build 7 Chestahedron vertices in 3D, normalized to unit sphere.

    The Chestahedron sits at the phi angle (36°) in a cube:
    - Vertex 0: apex (top)
    - Vertices 1-3: upper triangle, rotated by phi angle
    - Vertices 4-6: lower triangle, at phi-scaled distance
    """
    vertices = []

    # Apex
    vertices.append([0.0, 1.0, 0.0])

    # Upper triangle — rotated by phi angle around Y axis
    upper_r = math.sin(PHI_ANGLE)
    upper_y = math.cos(PHI_ANGLE)
    for k in range(3):
        angle = 2 * math.pi * k / 3 + PHI_ANGLE  # Offset by phi angle
        x = upper_r * math.cos(angle)
        z = upper_r * math.sin(angle)
        vertices.append([x, upper_y, z])

    # Lower triangle — inverted, scaled by 1/phi, rotated by pi/3
    lower_r = math.sin(PHI_ANGLE) / PHI
    lower_y = -math.cos(PHI_ANGLE)
    for k in range(3):
        angle = 2 * math.pi * k / 3 + PHI_ANGLE + math.pi / 3
        x = lower_r * math.cos(angle)
        z = lower_r * math.sin(angle)
        vertices.append([x, lower_y, z])

    # Normalize each to unit sphere
    return [normalize(v) for v in vertices]


def build_vertices_7d() -> list[list[float]]:
    """
    Extend 3D Chestahedron vertices to 7D.

    Strategy: each vertex is primarily a one-hot basis vector in R^7
    (ensuring high orthogonality), with phi-scaled 3D coordinate bleed
    into neighboring dimensions for geometric coupling.

    The one-hot dominance ensures each vertex is distinct enough that
    projection onto the manifold is faithful. The 3D coupling ensures
    the Chestahedron topology is encoded in the inter-vertex relationships.

    Result: 7 vectors in R^7, normalized to unit norm.
    """
    v3d = build_vertices_3d()
    vertices_7d = []

    for i in range(NUM_FACES):
        v = [0.0] * 7

        # Primary: strong one-hot in dimension i (each vertex owns one dim)
        v[i] = PHI  # Phi-scaled for dominance

        # Secondary: 3D geometry bleeds into adjacent dimensions
        # This encodes the actual Chestahedron topology
        for d in range(3):
            target_dim = (i + d + 1) % 7
            v[target_dim] += v3d[i][d] / (PHI * PHI)

        # Tertiary: adjacency-informed coupling
        # Adjacent faces get a tiny phi-inverse bleed
        for a, b in EDGES:
            if a == i:
                v[b] += 1.0 / (PHI * PHI * PHI)
            elif b == i:
                v[a] += 1.0 / (PHI * PHI * PHI)

        vertices_7d.append(normalize(v))

    return vertices_7d


def build_adjacency() -> list[list[float]]:
    """
    Build 7x7 adjacency matrix for Chestahedron face connectivity.
    adj[i][j] = 1.0 if faces i and j share an edge, else 0.0.
    Symmetric. Diagonal is 0.
    """
    adj = [[0.0] * NUM_FACES for _ in range(NUM_FACES)]
    for i, j in EDGES:
        adj[i][j] = 1.0
        adj[j][i] = 1.0
    return adj


# ---- Precomputed geometry (computed once at import) ----

VERTICES_7D = build_vertices_7d()
ADJACENCY = build_adjacency()


# ---- Projection and Coherence ----

def project_to_manifold(x: list[float]) -> list[float]:
    """
    Project a 7D vector onto the Chestahedron manifold.

    1. Compute similarity (dot product) with each vertex
    2. Sharp softmax (1/PHI temperature) over similarities -> weights
    3. Weighted combination of vertices
    4. Normalize result to unit norm

    Uses a sharp temperature (1/PHI) so vertices project faithfully
    back to themselves. The routing face_activations function uses the
    warmer PHI temperature for smoother blending.
    """
    x_norm = normalize(x)

    # Similarities to each vertex
    sims = [dot(x_norm, v) for v in VERTICES_7D]

    # Sharp softmax for faithful projection (1/PHI^2 temperature)
    weights = softmax(sims, temperature=1.0 / (PHI * PHI))

    # Weighted combination
    result = [0.0] * 7
    for w, vertex in zip(weights, VERTICES_7D):
        for d in range(7):
            result[d] += w * vertex[d]

    return normalize(result)


def coherence_score(x: list[float]) -> float:
    """
    How close x is to the Chestahedron manifold [0, 1].
    = cosine similarity between normalized x and its manifold projection.
    """
    x_norm = normalize(x)
    proj = project_to_manifold(x)
    cos_sim = dot(x_norm, proj)
    # Clamp to [0, 1]
    return max(0.0, min(1.0, cos_sim))


def face_activations(x: list[float], temperature: float = PHI) -> list[float]:
    """
    Activation strength of each face for state x.
    = softmax of similarities to each vertex at given temperature.

    Temperature guide:
        PHI (1.618)  — smooth blending for manifold projection
        1.0          — sharper discrimination for routing decisions
        0.5          — very sharp, only dominant faces matter (adjacency checks)
    """
    x_norm = normalize(x)
    sims = [dot(x_norm, v) for v in VERTICES_7D]
    return softmax(sims, temperature=temperature)


def adjacency_penalty(face_weights: list[float]) -> float:
    """
    Penalize co-activation of non-adjacent faces.
    = sum of (w_i * w_j) for all non-adjacent pairs (i, j).
    """
    penalty = 0.0
    for i in range(NUM_FACES):
        for j in range(i + 1, NUM_FACES):
            if ADJACENCY[i][j] < 0.5:  # Not adjacent
                penalty += face_weights[i] * face_weights[j]
    return penalty


def dominant_faces(face_weights: list[float], threshold: float = 0.1) -> list[str]:
    """Return face names with activation above threshold, sorted by weight."""
    indexed = [(w, FACE_NAMES[i]) for i, w in enumerate(face_weights)]
    indexed.sort(reverse=True)
    return [name for w, name in indexed if w > threshold]


# ---- Self-Test ----

def self_test() -> bool:
    """
    Verify geometry correctness. Returns True if all pass.
    Prints results to stderr.
    """
    log = lambda msg: print(msg, file=sys.stderr)
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            log(f"  PASS: {name}")
            passed += 1
        else:
            log(f"  FAIL: {name} — {detail}")
            failed += 1

    log("=== Chestahedron Geometry Self-Test ===")

    # 1. All 7 vertices have unit norm
    for i, v in enumerate(VERTICES_7D):
        n = norm(v)
        check(f"vertex[{i}] unit norm", abs(n - 1.0) < 1e-6, f"norm={n:.6f}")

    # 2. Adjacency is symmetric
    adj = ADJACENCY
    sym = all(adj[i][j] == adj[j][i] for i in range(7) for j in range(7))
    check("adjacency symmetric", sym)

    # 3. Correct number of edges (12 edges = 24 entries)
    edge_count = sum(adj[i][j] for i in range(7) for j in range(7))
    check("12 edges (24 entries)", abs(edge_count - 24.0) < 0.1, f"got {edge_count}")

    # 4. Projecting a vertex onto manifold returns ~itself
    for i, v in enumerate(VERTICES_7D):
        proj = project_to_manifold(v)
        cos_sim = dot(normalize(v), normalize(proj))
        check(f"vertex[{i}] self-projection", cos_sim > 0.85,
              f"cos_sim={cos_sim:.4f}")

    # 5. Coherence of a vertex is high
    for i, v in enumerate(VERTICES_7D):
        c = coherence_score(v)
        check(f"vertex[{i}] coherence high", c > 0.80, f"coherence={c:.4f}")

    # 6. Coherence of random vectors is lower
    import random
    random.seed(42)
    random_coherences = []
    for _ in range(50):
        rv = [random.gauss(0, 1) for _ in range(7)]
        random_coherences.append(coherence_score(rv))
    avg_random = sum(random_coherences) / len(random_coherences)
    avg_vertex = sum(coherence_score(v) for v in VERTICES_7D) / 7
    check("random coherence < vertex coherence",
          avg_random < avg_vertex,
          f"random={avg_random:.4f} vertex={avg_vertex:.4f}")

    # 7. Face activations sum to 1
    rv = [random.gauss(0, 1) for _ in range(7)]
    fa = face_activations(rv)
    fa_sum = sum(fa)
    check("face activations sum to 1", abs(fa_sum - 1.0) < 1e-6, f"sum={fa_sum}")

    # 8. Each vertex dominates its own face
    for i, v in enumerate(VERTICES_7D):
        fa = face_activations(v)
        dominant_idx = fa.index(max(fa))
        check(f"vertex[{i}] dominates face {i}", dominant_idx == i,
              f"dominant={dominant_idx}")

    # 9. Adjacency penalty is 0 for adjacent-only activations
    adj_only = [0.0] * 7
    adj_only[0] = 0.5  # DIAGNOSE
    adj_only[1] = 0.5  # LOAD (adjacent to DIAGNOSE)
    adj_pen = adjacency_penalty(adj_only)
    check("adjacent-only penalty is 0", adj_pen < 1e-6, f"penalty={adj_pen}")

    # 10. Adjacency penalty is >0 for non-adjacent co-activation
    non_adj = [0.0] * 7
    non_adj[0] = 0.5  # DIAGNOSE
    non_adj[4] = 0.5  # BREAK (not adjacent to DIAGNOSE)
    non_adj_pen = adjacency_penalty(non_adj)
    check("non-adjacent penalty > 0", non_adj_pen > 0, f"penalty={non_adj_pen}")

    log(f"\n=== Results: {passed} passed, {failed} failed ===")
    return failed == 0


if __name__ == "__main__":
    success = self_test()
    sys.exit(0 if success else 1)
