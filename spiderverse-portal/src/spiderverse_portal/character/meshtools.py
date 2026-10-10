"""Mesh utilities for building the prototype character (offline, NumPy only)."""

from __future__ import annotations

from typing import Callable

import numpy as np


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    v = np.asarray(vertices, np.float64)
    t = np.asarray(triangles, np.int64)
    fn = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])  # area-weighted
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, t[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def adjacency(n_vertices: int, triangles: np.ndarray) -> list[np.ndarray]:
    nbrs: list[set[int]] = [set() for _ in range(n_vertices)]
    for a, b, c in np.asarray(triangles, np.int64):
        nbrs[a].update((b, c))
        nbrs[b].update((a, c))
        nbrs[c].update((a, b))
    return [np.fromiter(s, np.int64) for s in nbrs]


def umbrella(values: np.ndarray, nbrs: list[np.ndarray]) -> np.ndarray:
    """Uniform Laplacian: mean of neighbours minus the value."""
    out = np.zeros_like(values)
    for i, nb in enumerate(nbrs):
        if len(nb):
            out[i] = values[nb].mean(axis=0) - values[i]
    return out


def taubin_smooth(vertices: np.ndarray, triangles: np.ndarray, iterations: int = 30, lam: float = 0.5,
                  mu: float = -0.53, fixed: np.ndarray | None = None) -> np.ndarray:
    """Low-pass smoothing without the shrinkage of plain Laplacian smoothing (Taubin 1995)."""
    v = np.asarray(vertices, np.float64).copy()
    nbrs = adjacency(len(v), triangles)
    keep = np.zeros(len(v), bool) if fixed is None else np.asarray(fixed, bool)
    for _ in range(iterations):
        for f in (lam, mu):
            d = umbrella(v, nbrs)
            d[keep] = 0.0
            v += f * d
    return v


def diffuse(values: np.ndarray, triangles: np.ndarray, iterations: int = 5, rate: float = 0.5) -> np.ndarray:
    nbrs = adjacency(len(values), triangles)
    v = np.asarray(values, np.float64).copy()
    for _ in range(iterations):
        v += rate * umbrella(v, nbrs)
    return v


def ray_triangles(origins: np.ndarray, dirs: np.ndarray, vertices: np.ndarray, triangles: np.ndarray,
                  chunk: int = 1024, both_sides: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Nearest hit (Moller-Trumbore) of each ray. Returns (t, triangle index); t = inf / index -1 on a miss."""
    o_all = np.asarray(origins, np.float64)
    d_all = np.asarray(dirs, np.float64)
    v0 = vertices[triangles[:, 0]]
    e1 = vertices[triangles[:, 1]] - v0
    e2 = vertices[triangles[:, 2]] - v0
    t_out = np.full(len(o_all), np.inf)
    i_out = np.full(len(o_all), -1, np.int64)
    for s in range(0, len(o_all), chunk):
        o = o_all[s:s + chunk, None, :]
        d = d_all[s:s + chunk, None, :]
        p = np.cross(d, e2[None])
        det = np.einsum("rtk,tk->rt", p, e1)
        ok = np.abs(det) > 1e-12 if both_sides else det > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - v0[None]
        u = np.einsum("rtk,rtk->rt", tv, p) * inv
        q = np.cross(tv, e1[None])
        v = np.einsum("rtk,rtk->rt", d, q) * inv
        t = np.einsum("tk,rtk->rt", e2, q) * inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
        t = np.where(hit, t, np.inf)
        idx = np.argmin(t, axis=1)
        t_out[s:s + chunk] = t[np.arange(len(idx)), idx]
        i_out[s:s + chunk] = np.where(np.isfinite(t_out[s:s + chunk]), idx, -1)
    return t_out, i_out


def surface_nets(sdf: Callable[[np.ndarray], np.ndarray], lo: np.ndarray, hi: np.ndarray, step: float,
                 refine: int = 3, blur: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Polygonise the zero level set of `sdf` (negative inside) with naive surface nets.

    Returns (vertices, triangles) with counter-clockwise outward-facing triangles.
    Vertices are pulled onto the surface with a few Newton steps along the gradient.
    """
    lo = np.asarray(lo, np.float64)
    hi = np.asarray(hi, np.float64)
    n = np.ceil((hi - lo) / step).astype(int) + 1
    xs, ys, zs = (lo[i] + step * np.arange(n[i]) for i in range(3))
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    f = sdf(grid.reshape(-1, 3)).reshape(n)
    for _ in range(blur):  # [1 2 1] blur per axis: removes sub-voxel slivers that break manifoldness
        for ax in range(3):
            f = 0.25 * (np.roll(f, 1, ax) + 2 * f + np.roll(f, -1, ax))
    inside = f < 0
    nx, ny, nz = n
    cell = (slice(0, nx - 1), slice(0, ny - 1), slice(0, nz - 1))
    offs = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]

    def corner(a, b, c, arr):
        return arr[a:nx - 1 + a, b:ny - 1 + b, c:nz - 1 + c]

    ins = [corner(*o, inside) for o in offs]
    any_in = np.logical_or.reduce(ins)
    all_in = np.logical_and.reduce(ins)
    active = any_in & ~all_in
    cell_id = -np.ones(active.shape, np.int64)
    ai = np.argwhere(active)
    cell_id[tuple(ai.T)] = np.arange(len(ai))

    # Vertex = mean of the edge crossings of the cell.
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if sum(abs(x - y) for x, y in zip(offs[a], offs[b])) == 1]
    acc = np.zeros((len(ai), 3))
    cnt = np.zeros(len(ai))
    base = lo + ai * step
    fv = [f[ai[:, 0] + o[0], ai[:, 1] + o[1], ai[:, 2] + o[2]] for o in offs]
    for a, b in edges:
        va, vb = fv[a], fv[b]
        cross = (va < 0) != (vb < 0)
        t = np.where(cross, va / np.where(cross, va - vb, 1.0), 0.0)
        pa = base + np.asarray(offs[a]) * step
        pb = base + np.asarray(offs[b]) * step
        p = pa + t[:, None] * (pb - pa)
        acc[cross] += p[cross]
        cnt[cross] += 1
    verts = acc / np.maximum(cnt, 1)[:, None]

    # Quads: every grid edge with a sign change is shared by 4 active cells.
    quads = []
    for axis in range(3):
        a1, a2 = [x for x in range(3) if x != axis]
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis] = slice(0, n[axis] - 1)
        sl1[axis] = slice(1, n[axis])
        f0, f1 = inside[tuple(sl0)], inside[tuple(sl1)]
        change = f0 != f1
        idx = np.argwhere(change)
        # The 4 cells around the edge: offsets 0/-1 along both other axes.
        good = (idx[:, a1] >= 1) & (idx[:, a2] >= 1) & (idx[:, a1] < n[a1] - 1) & (idx[:, a2] < n[a2] - 1)
        idx = idx[good]
        cells = []
        for d1, d2 in ((0, 0), (1, 0), (1, 1), (0, 1)):  # a cycle around the edge
            c = idx.copy()
            c[:, a1] -= d1
            c[:, a2] -= d2
            cells.append(cell_id[tuple(c.T)])
        quads.append(np.stack(cells, axis=1))  # orientation is fixed per triangle below
    quads = np.concatenate(quads)
    quads = quads[(quads >= 0).all(axis=1)]

    # Newton projection onto the surface.
    for _ in range(refine):
        g = sdf_gradient(sdf, verts, step * 0.25)
        fval = sdf(verts)
        verts = verts - (fval / np.maximum((g * g).sum(1), 1e-12))[:, None] * g

    # Split quads along the shorter diagonal.
    p = verts[quads]
    d02 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    d13 = np.linalg.norm(p[:, 1] - p[:, 3], axis=1)
    use02 = d02 <= d13
    t1 = np.where(use02[:, None], quads[:, [0, 1, 2]], quads[:, [0, 1, 3]])
    t2 = np.where(use02[:, None], quads[:, [0, 2, 3]], quads[:, [1, 2, 3]])
    tris = np.concatenate([t1, t2])
    # Make every triangle face along the SDF gradient (outward).
    v = verts
    fn = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    centre = v[tris].mean(axis=1)
    g = sdf_gradient(sdf, centre, step * 0.25)
    wrong = (fn * g).sum(1) < 0
    tris[wrong] = tris[wrong][:, ::-1]
    return verts, tris


def sdf_gradient(sdf: Callable[[np.ndarray], np.ndarray], p: np.ndarray, h: float) -> np.ndarray:
    p = np.asarray(p, np.float64)
    g = np.zeros_like(p)
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        g[:, k] = (sdf(p + e) - sdf(p - e)) / (2 * h)
    return g


def is_closed(triangles: np.ndarray) -> bool:
    """Every edge is shared by exactly two triangles with opposite orientation."""
    t = np.asarray(triangles, np.int64)
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    key = e[:, 0] * (int(e.max()) + 1) + e[:, 1]
    rev = e[:, 1] * (int(e.max()) + 1) + e[:, 0]
    if len(np.unique(key)) != len(key):
        return False
    return bool(np.isin(rev, key).all())
