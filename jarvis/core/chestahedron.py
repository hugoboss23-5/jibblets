"""
Chestahedron 7D geometric topology and coherence metric.

The Chestahedron is a 7-faced polyhedron (4 equilateral triangles + 3 kite quadrilaterals)
that bridges rational (3,4) and irrational (5/phi) geometry. The router operates
in a 7D embedding space where each dimension maps to a cognitive operation.

Cognitive faces:
    0: DIAGNOSE — read the situation
    1: LOAD    — pull relevant context
    2: CONSTRAIN — set boundaries
    3: GENERATE — produce candidates
    4: BREAK   — stress-test
    5: CUT     — select/compress
    6: ROUTE   — dispatch to organs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
PHI_ANGLE_RAD = math.radians(36)  # Phi angle in radians

FACE_NAMES = [
    "DIAGNOSE", "LOAD", "CONSTRAIN", "GENERATE", "BREAK", "CUT", "ROUTE"
]

NUM_FACES = 7


def _build_chestahedron_vertices() -> torch.Tensor:
    """
    Construct the 7 vertices of the Chestahedron in 3D, then project into 7D
    via a geometric mapping where each vertex anchors one cognitive dimension.

    The Chestahedron sits at 36° (phi angle) inside a cube. We use the known
    vertex coordinates normalized to unit sphere, then extend to 7D by assigning
    each vertex a one-hot basis direction scaled by its 3D coordinates' geometric
    properties.
    """
    # Chestahedron vertices in 3D (normalized form)
    # Apex at top, base triangle below, with kite faces connecting
    h = math.sqrt(2 / 3)
    r = math.sqrt(1 / 3)

    # Top apex
    v0 = [0.0, h * PHI, 0.0]
    # Upper triangle vertices (rotated by phi angle)
    v1 = [r * math.cos(PHI_ANGLE_RAD), h, r * math.sin(PHI_ANGLE_RAD)]
    v2 = [r * math.cos(PHI_ANGLE_RAD + 2 * math.pi / 3), h,
          r * math.sin(PHI_ANGLE_RAD + 2 * math.pi / 3)]
    v3 = [r * math.cos(PHI_ANGLE_RAD + 4 * math.pi / 3), h,
          r * math.sin(PHI_ANGLE_RAD + 4 * math.pi / 3)]
    # Lower triangle vertices
    v4 = [r * PHI * math.cos(0), -h * 0.5, r * PHI * math.sin(0)]
    v5 = [r * PHI * math.cos(2 * math.pi / 3), -h * 0.5,
          r * PHI * math.sin(2 * math.pi / 3)]
    v6 = [r * PHI * math.cos(4 * math.pi / 3), -h * 0.5,
          r * PHI * math.sin(4 * math.pi / 3)]

    verts_3d = torch.tensor([v0, v1, v2, v3, v4, v5, v6], dtype=torch.float32)

    # Normalize each vertex to unit sphere
    norms = verts_3d.norm(dim=1, keepdim=True).clamp(min=1e-8)
    verts_3d = verts_3d / norms

    # Extend to 7D: each vertex gets a one-hot basis component + scaled 3D coords
    # This creates a 7D manifold anchored by the Chestahedron's 3D geometry
    verts_7d = torch.zeros(7, 7, dtype=torch.float32)
    for i in range(7):
        # Primary dimension: unit basis vector (cognitive face identity)
        verts_7d[i, i] = 1.0
        # Modulate by 3D geometric position (spread across all dims)
        for j in range(3):
            # Cross-couple 3D coords into neighboring dimensions
            target_dim = (i + j + 1) % 7
            verts_7d[i, target_dim] += verts_3d[i, j].item() * (1.0 / PHI)

    # Normalize rows to unit norm for consistent manifold distance
    row_norms = verts_7d.norm(dim=1, keepdim=True).clamp(min=1e-8)
    verts_7d = verts_7d / row_norms

    return verts_7d


# Pre-computed Chestahedron vertices in 7D (registered as buffer in modules that use it)
CHESTA_VERTICES_7D = _build_chestahedron_vertices()


def _build_adjacency() -> torch.Tensor:
    """
    Adjacency matrix for Chestahedron faces.
    Faces sharing an edge are adjacent. This encodes the topology.

    Face layout:
        Triangles: 0,1,2,3 (DIAGNOSE, LOAD, CONSTRAIN, GENERATE)
        Kites: 4,5,6 (BREAK, CUT, ROUTE)
    Each kite shares edges with 2 triangles and 1 other kite.
    Each triangle shares edges with 2 kites and 1 other triangle.
    """
    adj = torch.zeros(7, 7, dtype=torch.float32)
    edges = [
        (0, 1), (0, 2), (0, 3),   # Apex triangle connects to upper triangles
        (1, 4), (1, 5),           # LOAD connects to BREAK, CUT
        (2, 5), (2, 6),           # CONSTRAIN connects to CUT, ROUTE
        (3, 4), (3, 6),           # GENERATE connects to BREAK, ROUTE
        (4, 5), (5, 6), (4, 6),  # Kites interconnect
    ]
    for i, j in edges:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


CHESTA_ADJACENCY = _build_adjacency()


class ChestahedronProjector(nn.Module):
    """
    Projects arbitrary vectors onto the Chestahedron manifold in 7D space.
    Used to constrain the router's internal state to geometrically valid configurations.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("vertices", CHESTA_VERTICES_7D)
        self.register_buffer("adjacency", CHESTA_ADJACENCY)

    def project_to_manifold(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project a batch of 7D vectors onto the Chestahedron manifold.

        Uses soft assignment to vertices (convex combination) weighted by
        proximity, constrained by adjacency so only nearby faces interact.

        Args:
            x: (batch, 7) tensor of 7D points

        Returns:
            (batch, 7) tensor projected onto the manifold
        """
        # Compute similarity to each vertex
        # x: (B, 7), vertices: (7, 7) -> similarities: (B, 7)
        sims = torch.matmul(x, self.vertices.t())

        # Temperature-scaled softmax for soft vertex assignment
        weights = F.softmax(sims * PHI, dim=-1)  # phi-scaled temperature

        # Reconstruct as convex combination of vertices
        # weights: (B, 7), vertices: (7, 7) -> projected: (B, 7)
        projected = torch.matmul(weights, self.vertices)

        return projected

    def coherence_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute how close a 7D state is to the Chestahedron manifold.

        Returns a scalar in [0, 1] where 1 = perfectly on-manifold.
        This is the geometric coherence metric used in the router's loss.

        Args:
            x: (batch, 7) tensor

        Returns:
            (batch,) coherence scores
        """
        # Normalize input
        x_norm = F.normalize(x, dim=-1)

        # Project onto manifold
        projected = self.project_to_manifold(x_norm)
        proj_norm = F.normalize(projected, dim=-1)

        # Coherence = cosine similarity between original and projected
        coherence = (x_norm * proj_norm).sum(dim=-1)

        return coherence

    def face_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute activation strength of each cognitive face for a 7D state.

        Args:
            x: (batch, 7) tensor

        Returns:
            (batch, 7) face activation weights (sum to 1)
        """
        x_norm = F.normalize(x, dim=-1)
        sims = torch.matmul(x_norm, self.vertices.t())
        return F.softmax(sims * PHI, dim=-1)

    def adjacency_loss(self, face_weights: torch.Tensor) -> torch.Tensor:
        """
        Penalize activation patterns that violate adjacency topology.
        Non-adjacent faces should not be strongly co-activated.

        Args:
            face_weights: (batch, 7) face activations

        Returns:
            scalar loss
        """
        # Compute pairwise co-activation: (B, 7, 1) * (B, 1, 7) -> (B, 7, 7)
        coactivation = face_weights.unsqueeze(-1) * face_weights.unsqueeze(-2)

        # Mask: penalize co-activation of non-adjacent faces
        non_adj = 1.0 - self.adjacency
        # Zero out diagonal (self-co-activation is fine)
        non_adj = non_adj - torch.diag(torch.diag(non_adj))

        # Loss = average non-adjacent co-activation
        penalty = (coactivation * non_adj.unsqueeze(0)).sum(dim=(-1, -2))
        return penalty.mean()


class ChestahedronEmbedding(nn.Module):
    """
    Learnable embedding layer that maps input features into the 7D Chestahedron space.
    The embedding is constrained to produce outputs near the manifold.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.projector = ChestahedronProjector()
        self.linear = nn.Linear(input_dim, NUM_FACES)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map input features to 7D Chestahedron-constrained embedding.

        Args:
            x: (batch, input_dim) input features

        Returns:
            (batch, 7) embedding on/near the Chestahedron manifold
        """
        raw = self.linear(x) * self.scale
        # Soft projection: blend between raw output and manifold projection
        projected = self.projector.project_to_manifold(raw)
        # Learnable interpolation (starts closer to manifold)
        alpha = torch.sigmoid(self.scale)
        return alpha * projected + (1 - alpha) * F.normalize(raw, dim=-1)
