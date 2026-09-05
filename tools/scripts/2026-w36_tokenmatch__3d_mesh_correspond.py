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
    members: List[int] = field(default_factory=list)
    centroid: Tuple[float, float, float]