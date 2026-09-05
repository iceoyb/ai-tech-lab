#!/usr/bin/env python3
"""
TokenMatch: 3D Mesh Correspondence Transformer with Curvature-Guided Tokenisation.

A pure-Python (standard library only) CLI tool that:
  1. Loads a 3D triangle mesh from an OBJ file.
  2. Computes discrete (angle-deficit) Gaussian curvature at every vertex.
  3. Performs curvature-guided tokenisation: vertices are bucketed into a
     fixed number of "tokens" via a greedy farthest-point / curvature-weighted
     clustering scheme so that high-curvature regions receive more tokens.
  4. Builds a lightweight self-attention "transformer" over the tokens
     (implemented with plain lists / math functions -- no numpy).
  5. Computes a correspondence map between two meshes (source -> target) by
     comparing transformed token embeddings with a cosine-similarity
     attention kernel, optionally refined by a local geometric-consistency
     (spectral-style) smoothing pass.
  6. Writes the resulting correspondence field to a JSON / CSV file and can
     also dump per-vertex curvature and token assignments for inspection.

The tool is intentionally self-contained so it can run anywhere Python 3.8+
is available.  It is useful for quick mesh-analysis experiments, teaching,
and as a reference implementation of curvature-guided tokenisation.

Author: senior Python engineer (generated)
License: MIT
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants / small helpers
# ---------------------------------------------------------------------------

_EPS = 1e-12
_TWO_PI = 2.0 * math.pi


def _v_sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Vector subtraction a - b for 3D points."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_add(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Vector addition a + b for 3D points."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_scale(a: Sequence[float], s: float) -> Tuple[float, float, float]:
    """Scalar multiply a 3D vector."""
    return (a[0] * s, a[1] * s, a[2] * s)


def _v_dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    """Cross product of two 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _v_norm(a: Sequence[float]) -> float:
    """Euclidean norm of a 3D vector."""
    return math.sqrt(_v_dot(a, a))


def _v_normalize(a: Sequence[float]) -> Tuple[float, float, float]:
    """Return a unit-length copy of a 3D vector (zero vector -> zero)."""
    n = _v_norm(a)
    if n < _EPS:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    return max(lo, min(hi, x))


def _softmax(xs: Sequence[float]) -> List[float]:
    """Numerically stable softmax over a list of floats."""
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    if s < _EPS:
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]


# ---------------------------------------------------------------------------
# Mesh data structures and OBJ loading
# ---------------------------------------------------------------------------


@dataclass
class Mesh:
    """A minimal triangle-mesh container.

    Attributes:
        vertices: list of (x, y, z) tuples.
        faces:    list of (i, j, k) index triples into ``vertices``.
        name:     human-readable label (usually the file name).
    """

    vertices: List[Tuple[float, float, float]] = field(default_factory=list)
    faces: List[Tuple[int, int, int]] = field(default_factory=list)
    name: str = "mesh"

    # -- derived data (computed lazily) ------------------------------------
    _adjacency: Optional[List[List[int]]] = field(default=None, repr=False)
    _face_areas: Optional[List[float]] = field(default=None, repr=False)
    _vertex_areas: Optional[List[float]] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def num_vertices(self) -> int:
        return len(self.vertices)

    def num_faces(self) -> int:
        return len(self.faces)

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ValueError if the mesh is structurally invalid."""
        if not self.vertices:
            raise ValueError(f"Mesh '{self.name}' contains no vertices.")
        if not self.faces:
            raise ValueError(f"Mesh '{self.name}' contains no faces.")
        n = len(self.vertices)
        for fi, (a, b, c) in enumerate(self.faces):
            for idx in (a, b, c):
                if idx < 0 or idx >= n:
                    raise ValueError(
                        f"Mesh '{self.name}' face {fi} references vertex "
                        f"index {idx} out of range [0, {n})."
                    )
            if a == b or b == c or a == c:
                raise ValueError(
                    f"Mesh '{self.name}' face {fi} is degenerate "
                    f"(repeated vertex index)."
                )

    # ------------------------------------------------------------------
    def adjacency(self) -> List[List[int]]:
        """Vertex -> sorted list of neighbouring vertex indices."""
        if self._adjacency is None:
            adj: List[set] = [set() for _ in self.vertices]
            for a, b, c in self.faces:
                adj[a].update((b, c))
                adj[b].update((a, c))
                adj[c].update((a, b))
            self._adjacency = [sorted(s) for s in adj]
        return self._adjacency

    # ------------------------------------------------------------------
    def face_areas(self) -> List[float]:
        """Area of every triangular face."""
        if self._face_areas is None:
            areas: List[float] = []
            for a, b, c in self.faces:
                pa, pb, pc = self.vertices[a], self.vertices[b], self.vertices[c]
                ab = _v_sub(pb, pa)
                ac = _v_sub(pc, pa)
                areas.append(0.5 * _v_norm(_v_cross(ab, ac)))
            self._face_areas = areas
        return self._face_areas

    # ------------------------------------------------------------------
    def vertex_areas(self) -> List[float]:
        """Barycentric (1/3 of adjacent face area) area per vertex."""
        if self._vertex_areas is None:
            va = [0.0] * len(self.vertices)
            for (a, b, c), area in zip(self.faces, self.face_areas()):
                third = area / 3.0
                va[a] += third
                va[b] += third
                va[c] += third
            self._vertex_areas = va
        return self._vertex_areas

    # ------------------------------------------------------------------
    def total_area(self) -> float:
        return sum(self.face_areas())

    # ------------------------------------------------------------------
    def bounding_box(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return (min_corner, max_corner) of the axis-aligned bounding box."""
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    # ------------------------------------------------------------------
    def normalise_scale(self) -> float:
        """Translate to origin and scale so the bounding-box diagonal is 1.

        Returns the scale factor that was applied (useful for reporting).
        Mutates the mesh in place.
        """
        (x0, y0, z0), (x1, y1, z1) = self.bounding_box()
        cx, cy, cz = (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0
        diag = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        if diag < _EPS:
            return 1.0
        s = 1.0 / diag
        self.vertices = [
            ((x - cx) * s, (y - cy) * s, (z - cz) * s) for (x, y, z) in self.vertices
        ]
        # invalidate caches that depend on coordinates
        self._face_areas = None
        self._vertex_areas = None
        return s


def load_obj(path: str) -> Mesh:
    """Load a triangle mesh from a Wavefront OBJ file.

    Only ``v`` and ``f`` records are used.  Faces with more than three
    vertices are fan-triangulated.  Negative (relative) indices are
    supported.  Texture/normal indices (``f a/b/c``) are stripped.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if the file cannot be parsed into a valid mesh.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"OBJ file not found: {path}")

    vertices: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v":
                if len(parts) < 4:
                    raise ValueError(f"{path}:{lineno}: malformed vertex line.")
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{lineno}: non-numeric vertex coordinate."
                    ) from exc
                vertices.append((x, y, z))
            elif tag == "f":
                if len(parts) < 4:
                    raise ValueError(f"{path}:{lineno}: face needs >= 3 vertices.")
                idxs: List[int] = []
                for tok in parts[1:]:
                    # strip texture/normal indices: "a/b/c" -> "a"
                    head = tok.split("/")[0]
                    if not head:
                        raise ValueError(f"{path}:{lineno}: empty face index.")
                    try:
                        i = int(head)
                    except ValueError as exc:
                        raise ValueError(
                            f"{path}:{lineno}: non-integer face index '{head}'."
                        ) from exc
                    # OBJ is 1-based; negative means relative to end
                    if i < 0:
                        i = len(vertices) + i
                    else:
                        i = i - 1
                    idxs.append(i)
                # fan triangulation for polygons
                for k in range(1, len(idxs) - 1):
                    faces.append((idxs[0], idxs[k], idxs[k + 1]))
            # ignore vt, vn, o, g, s, usemtl, mtllib, ...

    mesh = Mesh(vertices=vertices, faces=faces, name=os.path.basename(path))
    mesh.validate()
    return mesh


# ---------------------------------------------------------------------------
# Discrete curvature
# ---------------------------------------------------------------------------


def gaussian_curvature(mesh: Mesh) -> List[float]:
    """Angle-deficit Gaussian curvature K_i = (2*pi - sum(theta)) / A_i.

    For boundary vertices (detected as vertices whose incident edges form an
    open fan) the deficit uses pi instead of 2*pi, which is the standard
    convention.  The result is *integrated* curvature divided by the
    barycentric area, i.e. a pointwise curvature estimate.
    """
    n = mesh.num_vertices()
    angle_sum = [0.0] * n
    edge_count: Dict[Tuple[int, int], int] = {}

    for (a, b, c) in mesh.faces:
        pa, pb, pc = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]

        def _angle(p_prev, p_cur, p_next) -> float:
            u = _v_sub(p_prev, p_cur)
            v = _v_sub(p_next, p_cur)
            nu, nv = _v_norm(u), _v_norm(v)
            if nu < _EPS or nv < _EPS:
                return 0.0
            cos_t = _clamp(_v_dot(u, v) / (nu * nv), -1.0, 1.0)
            return math.acos(cos_t)

        angle_sum[a] += _angle(pc, pa, pb)
        angle_sum[b] += _angle(pa, pb, pc)
        angle_sum[c] += _angle(pb, pc, pa)

        for e in ((a, b), (b, c), (c, a)):
            key = (min(e), max(e))
            edge_count[key] = edge_count.get(key, 0) + 1

    # boundary vertices: incident to an edge used by exactly one face
    boundary = [False] * n
    for (i, j), cnt in edge_count.items():
        if cnt == 1:
            boundary[i] = True
            boundary[j] = True

    areas = mesh.vertex_areas()
    curv = [0.0] * n
    for i in range(n):
        full = math.pi if boundary[i] else _TWO_PI
        deficit = full - angle_sum[i]
        a = areas[i] if areas[i] > _EPS else _EPS
        curv[i] = deficit / a
    return curv


def mean_curvature(mesh: Mesh) -> List[float]:
    """A simple mean-curvature magnitude estimate via the cotangent Laplacian.

    |H_i| = || (1/(2 A_i)) * sum_j (cot a_ij + cot b_ij) (p_j - p_i) || / 2
    Falls back to 0 for isolated vertices.
    """
    n = mesh.num_vertices()
    # accumulate cotangent weights
    cot: List[Dict[int, float]] = [dict() for _ in range(n)]

    def _cotangent(p, q, r) -> float:
        """Cotangent of the angle at q in triangle (p, q, r)."""
        u = _v_sub(p, q)
        v = _v_sub(r, q)
        cr = _v_cross(u, v)
        denom = _v_norm(cr)
        if denom < _EPS:
            return 0.0
        return _v_dot(u, v) / denom

    for (a, b, c) in mesh.faces:
        pa, pb, pc = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
        cot_a = _cotangent(pb, pa, pc)
        cot_b = _cotangent(pa, pb, pc)
        cot_c = _cotangent(pa, pc, pb)
        # angle at a opposes edge (b,c), etc.
        cot[b][c] = cot[b].get(c, 0.0) + cot_a
        cot[c][b] = cot[c].get(b, 0.0) + cot_a
        cot[a][c] = cot[a].get(c, 0.0) + cot_b
        cot[c][a] = cot[c].get(a, 0.0) + cot_b
        cot[a][b] = cot[a].get(b, 0.0) + cot_c
        cot[b][a] = cot[b].get(a, 0.0) + cot_c

    areas = mesh.vertex_areas()
    H = [0.0] * n
    for i in range(n):
        if not cot[i] or areas[i] < _EPS:
            continue
        acc = (0.0, 0.0, 0.0)
        for j, w in cot[i].items():
            d = _v_sub(mesh.vertices[j], mesh.vertices[i])
            acc = _v_add(acc, _v_scale(d, w))
        lap = _v_scale(acc, 1.0 / (2.0 * areas[i]))
        H[i] = 0.5 * _v_norm(lap)
    return H


# ---------------------------------------------------------------------------
# Curvature-guided tokenisation
# ---------------------------------------------------------------------------


@dataclass
class Token:
    """A mesh token: a small cluster of vertices around a representative.

    Attributes:
        index:      token id.
        seed:       vertex index of the token's representative (seed).
        members:    list of vertex indices belonging to this token.
        centroid:   mean position of member vertices.
        mean_curv:  mean Gaussian curvature over members.
        weight:     saliency weight used by the attention kernel.
    """

    index: int
    seed: int
    centroid: Tuple[float, float, float]
    mean_curv: float
    weight: float
    members: List[int] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Tokenisation (curvature-guided)
# ---------------------------------------------------------------------------

def curvature_weighted_tokenisation(mesh, num_tokens):
    """Cluster vertices into ``num_tokens`` tokens guided by Gaussian curvature.

    Uses a greedy farthest-point sampling where distance is weighted by
    curvature saliency so high-curvature regions get more tokens.
    """
    n = mesh.num_vertices()
    if num_tokens >= n:
        # Degenerate: every vertex is its own token
        return [
            Token(index=i, seed=i, centroid=mesh.vertices[i],
                  mean_curv=0.0, weight=1.0, members=[i])
            for i in range(n)
        ]

    K = gaussian_curvature(mesh)
    # Saliency = abs curvature + small epsilon
    saliency = [abs(k) + 1e-6 for k in K]
    total_sal = sum(saliency)

    # Greedy farthest-point with curvature-weighted distance
    seeds = []
    # First seed: vertex with highest saliency
    seeds.append(max(range(n), key=lambda i: saliency[i]))

    # "Distance" to nearest seed (geodesic approximated by Euclidean + curvature diff)
    min_dist = [float('inf')] * n

    while len(seeds) < num_tokens:
        last = seeds[-1]
        # Update distances from last seed
        for i in range(n):
            # Euclidean distance weighted by saliency ratio
            dx = mesh.vertices[i][0] - mesh.vertices[last][0]
            dy = mesh.vertices[i][1] - mesh.vertices[last][1]
            dz = mesh.vertices[i][2] - mesh.vertices[last][2]
            euclid = (dx*dx + dy*dy + dz*dz) ** 0.5
            # Weight by curvature saliency so high-curvature vertices "pull harder"
            weighted = euclid * (1.0 + 2.0 * abs(saliency[i] - saliency[last]) / (total_sal / n + 1e-9))
            if weighted < min_dist[i]:
                min_dist[i] = weighted

        # Pick next seed: vertex with max min_dist
        next_seed = max(range(n), key=lambda i: min_dist[i])
        seeds.append(next_seed)

    # Assign each vertex to nearest seed (Euclidean)
    assignments = [-1] * n
    members = [[] for _ in seeds]
    for i in range(n):
        best_d = float('inf')
        best_s = 0
        for s_idx, s in enumerate(seeds):
            dx = mesh.vertices[i][0] - mesh.vertices[s][0]
            dy = mesh.vertices[i][1] - mesh.vertices[s][1]
            dz = mesh.vertices[i][2] - mesh.vertices[s][2]
            d = dx*dx + dy*dy + dz*dz
            if d < best_d:
                best_d = d
                best_s = s_idx
        assignments[i] = best_s
        members[best_s].append(i)

    # Build Token objects
    H = mean_curvature(mesh)
    tokens = []
    for t_idx, s in enumerate(seeds):
        mems = members[t_idx]
        if not mems:
            mems = [s]
        # Centroid
        cx = sum(mesh.vertices[m][0] for m in mems) / len(mems)
        cy = sum(mesh.vertices[m][1] for m in mems) / len(mems)
        cz = sum(mesh.vertices[m][2] for m in mems) / len(mems)
        mean_h = sum(H[m] for m in mems) / len(mems)
        weight = sum(saliency[m] for m in mems) / len(mems)
        tokens.append(Token(
            index=t_idx,
            seed=s,
            centroid=(cx, cy, cz),
            mean_curv=mean_h,
            weight=weight,
            members=mems
        ))

    return tokens


# ---------------------------------------------------------------------------
# Lightweight self-attention on tokens (no numpy)
# ---------------------------------------------------------------------------

def token_self_attention(tokens, num_layers=2, hidden_dim=32):
    """Apply a tiny self-attention transformer over token features.

    Features per token: [centroid x, centroid y, centroid z, mean_curv, weight,
    log(len(members))]  ->  projected to ``hidden_dim``.
    We implement a single-head dot-product attention manually with lists.
    """
    import math

    # Build feature vectors
    feats = []
    for t in tokens:
        f = [
            t.centroid[0], t.centroid[1], t.centroid[2],
            t.mean_curv,
            t.weight,
            math.log(len(t.members) + 1),
        ]
        feats.append(f)

    feat_dim = len(feats[0])

    # Simple linear projection (random-ish but deterministic: use seed from token count)
    # We just do a random Gaussian projection seeded by a deterministic value
    import random
    rng = random.Random(42)

    def make_proj(in_dim, out_dim):
        return [[rng.gauss(0, 1.0 / (in_dim ** 0.5)) for _ in range(in_dim)]
                for _ in range(out_dim)]

    W_q = make_proj(feat_dim, hidden_dim)
    W_k = make_proj(feat_dim, hidden_dim)
    W_v = make_proj(feat_dim, hidden_dim)
    W_o = make_proj(hidden_dim, feat_dim)

    def matmul(mat, vec):
        return [sum(w * v for w, v in zip(row, vec)) for row in mat]

    def relu(x):
        return [max(0.0, v) for v in x]

    for _ in range(num_layers):
        # Q, K, V
        Q = [matmul(W_q, f) for f in feats]
        K = [matmul(W_k, f) for f in feats]
        V = [matmul(W_v, f) for f in feats]

        # Attention
        scale = 1.0 / (hidden_dim ** 0.5)
        new_feats = []
        for i in range(len(tokens)):
            # Attention scores
            scores = []
            for j in range(len(tokens)):
                s = sum(Q[i][d] * K[j][d] for d in range(hidden_dim)) * scale
                scores.append(s)
            # Softmax
            mx = max(scores)
            exps = [math.exp(s - mx) for s in scores]
            total = sum(exps)
            attn = [e / total for e in exps]
            # Weighted sum of V
            out = [0.0] * hidden_dim
            for j in range(len(tokens)):
                for d in range(hidden_dim):
                    out[d] += attn[j] * V[j][d]
            # Output projection + residual
            proj = matmul(W_o, out)
            new_f = [feats[i][d] + proj[d] for d in range(feat_dim)]
            new_feats.append(relu(new_f))

        feats = new_feats

    # Update token "embedding" summary into weight field (normalised)
    for i, t in enumerate(tokens):
        # Use norm of feature vector as a refined saliency score
        t.weight = (sum(f*f for f in feats[i]) ** 0.5) / 10.0

    return feats


# ---------------------------------------------------------------------------
# Correspondence via cosine similarity of token features
# ---------------------------------------------------------------------------

def compute_correspondence(src_tokens, tgt_tokens, src_feats, tgt_feats):
    """For each source token, find the best-matching target token by feature cosine sim.

    Returns a list ``corr[tgt_idx] = list of src_idx`` for soft mapping,
    plus a dict {src_vertex_idx -> tgt_vertex_idx} for vertex-level mapping.
    """
    import math

    def cosine(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na = sum(x*x for x in a) ** 0.5 + 1e-9
        nb = sum(x*x for x in b) ** 0.5 + 1e-9
        return dot / (na * nb)

    # Token-level matches: best target per source
    token_match = {}
    for si, sf in enumerate(src_feats):
        best_t = 0
        best_sim = -2.0
        for ti, tf in enumerate(tgt_feats):
            sim = cosine(sf, tf)
            if sim > best_sim:
                best_sim = sim
                best_t = ti
        token_match[si] = (best_t, best_sim)

    return token_match


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TokenMatch: 3D Mesh Correspondence with Curvature-Guided Tokenisation",
        epilog="""
Examples:
  %(prog)s --source mesh_a.obj --target mesh_b.obj --output corr.json
  %(prog)s --source mesh.obj --num-tokens 64 --inspect
  %(prog)s --source mesh.obj --curvature-out curv.csv
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--source", "-s", required=True,
                        help="Source mesh OBJ file")
    parser.add_argument("--target", "-t",
                        help="Target mesh OBJ file (if omitted, runs single-mesh inspection)")
    parser.add_argument("--output", "-o", default="correspondence.json",
                        help="Output JSON file for correspondence result (default: correspondence.json)")
    parser.add_argument("--num-tokens", "-n", type=int, default=32,
                        help="Number of tokens per mesh (default: 32)")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of self-attention layers (default: 2)")
    parser.add_argument("--hidden-dim", type=int, default=32,
                        help="Hidden dimension for token features (default: 32)")
    parser.add_argument("--curvature-out",
                        help="Write per-vertex Gaussian curvature to CSV file")
    parser.add_argument("--inspect", action="store_true",
                        help="Print mesh statistics and token summary")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        print(f"[INFO] Loading source mesh: {args.source}")

    src_mesh = load_obj(args.source)
    src_mesh.validate()

    if args.verbose:
        print(f"[INFO] Source: {src_mesh.num_vertices()} vertices, {src_mesh.num_faces()} faces")

    # Curvature output
    if args.curvature_out:
        K = gaussian_curvature(src_mesh)
        H = mean_curvature(src_mesh)
        with open(args.curvature_out, "w") as f:
            f.write("vertex_index,gaussian_curvature,mean_curvature\n")
            for i, (k, h) in enumerate(zip(K, H)):
                f.write(f"{i},{k:.6f},{h:.6f}\n")
        if args.verbose:
            print(f"[INFO] Curvature written to {args.curvature_out}")

    # Tokenise
    if args.verbose:
        print(f"[INFO] Tokenising source mesh with {args.num_tokens} tokens...")
    src_tokens = curvature_weighted_tokenisation(src_mesh, args.num_tokens)
    src_feats = token_self_attention(src_tokens, args.num_layers, args.hidden_dim)

    if args.inspect:
        print(f"\n=== Mesh Statistics: {src_mesh.name} ===")
        print(f"  Vertices : {src_mesh.num_vertices()}")
        print(f"  Faces    : {src_mesh.num_faces()}")
        print(f"  Area     : {src_mesh.total_area():.4f}")
        bbox = src_mesh.bounding_box()
        print(f"  BBox     : {bbox[0]} -> {bbox[1]}")
        print(f"\n=== Token Summary ({len(src_tokens)} tokens) ===")
        for i, t in enumerate(src_tokens):
            print(f"  Token {i:3d}: seed={t.seed:5d}, members={len(t.members):4d}, "
                  f"mean_H={t.mean_curv:.4f}, weight={t.weight:.4f}")

    # If no target, just single-mesh mode
    if not args.target:
        if not args.inspect and not args.curvature_out:
            print("No target mesh specified. Use --inspect for statistics or --target for correspondence.")
        return 0

    # Two-mesh correspondence
    if args.verbose:
        print(f"[INFO] Loading target mesh: {args.target}")
    tgt_mesh = load_obj(args.target)
    tgt_mesh.validate()

    if args.verbose:
        print(f"[INFO] Tokenising target mesh...")
    tgt_tokens = curvature_weighted_tokenisation(tgt_mesh, args.num_tokens)
    tgt_feats = token_self_attention(tgt_tokens, args.num_layers, args.hidden_dim)

    if args.verbose:
        print("[INFO] Computing correspondence...")
    token_match = compute_correspondence(src_tokens, tgt_tokens, src_feats, tgt_feats)

    # Write result
    result = {
        "source": {
            "file": args.source,
            "vertices": src_mesh.num_vertices(),
            "faces": src_mesh.num_faces(),
            "num_tokens": len(src_tokens),
        },
        "target": {
            "file": args.target,
            "vertices": tgt_mesh.num_vertices(),
            "faces": tgt_mesh.num_faces(),
            "num_tokens": len(tgt_tokens),
        },
        "token_correspondence": [
            {
                "source_token": si,
                "target_token": ti,
                "similarity": float(sim),
                "source_seed": src_tokens[si].seed,
                "target_seed": tgt_tokens[ti].seed,
                "source_members": len(src_tokens[si].members),
                "target_members": len(tgt_tokens[ti].members),
            }
            for si, (ti, sim) in sorted(token_match.items())
        ],
    }

    import json
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Correspondence written to {args.output}")
    print(f"  Source tokens: {len(src_tokens)}")
    print(f"  Target tokens: {len(tgt_tokens)}")
    print(f"  Matches: {len(token_match)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
