"""Endless 3D parts on a substitution surface map (named parts_penrose.py until 2026-11-23). First Penrose's
(2026-11-16: "do the 3d parts for penrose and icosahedron tilings"): parts.py's
pieces on the levels of the rhomb tiling surface_substitution.py draws, a vertex piece on each corner of its rhombs and an
edge piece along each of their edges, from the mesh's own faces (level 0) down forever, each level phi times finer
than the one above. Since the same day's flights they're on every other level only (level_step), phi^2 apart: on every
level they were dense. The open frames (posts, beams and horseshoe rings) stand on the same lines the strapwork's levels
are drawn on, and a coarser level's corners are corners of every finer one, so the posts nest as the square maps' do.
Since 2026-11-22 any rule's (one SubstitutionLattice each): on a girih mesh (Darb-i Imam), the girih tiles' corners and
edges, on every level, LAMBDA = 8.5 times finer (pieces_near(), Atlas).

What parts.py asks of the lattice (parts.SphereLattice, surface.MeshLattice) is answered here, from the tiles the
strapwork already keeps near the drone (surface_substitution.State's levels, shared: the parts ask for the windows they need
and the strapwork's levels are cut wider if need be):
- A level's corners and edges near a point (lattice_near()): each half's corners A, B, C, its legs AB and AC (the
  rhombs' edges; the base BC is a rhomb's inside). A corner inside a face is named by its ring coefficients from its
  face's root, exactly; one on a face's side, which both faces at that side have, by the mesh's edge and how far along
  it it is, exactly, as P + Q phi in whole numbers (the matching rule puts it at the same point from both faces), and
  one at a mesh corner by the mesh's vertex. Which side a corner is on is decided from its ring coefficients too: the
  ideal rhomb's sides are lines whose points have a whole-number property (Im(zeta^a z) fixed, in Z[phi]).
- Each vertex piece's x axis is toward its first neighbor, picked exactly (the edge with the least of the ten
  directions in its vertex's home face, the lowest numbered face it's on), so a level built again round another point
  turns no post. Normals and slopes are the faces', averaged at a side or corner, as surface.py's keys have them.
- Places: from the map's anchor, the vertex the pieces are around rounded once from its exact point, the others in
  float64 from the tiles' roots (surface_substitution.Level.map_points()); a level's own units are the mesh's times phi^d.
- The giant levels' pieces in the window: their tiles near the drone (kept for the window by the strapwork), the
  pieces within reach in float64, and those worked out exactly: their points are in Q(sqrt 5) over powers of 2 (the
  mesh's float32 corners and the ideal rhomb's coordinates), the camera dyadic, so their differences are exact, and
  with sqrt 5 to as many bits as the depth needs, surface.exact_piece_at() cuts them to the window in Fractions.
"""

import math
import time
from fractions import Fraction

import numpy as np

from . import parts
from . import penrose
from . import schedule
from . import substitution
from . import surface
from . import surface_substitution as SS
from .cyclotomic import (RING10 as RING, im_pair, q_add, q_float, q_fraction, q_mul, q_scale,
                         q_sub, re2_pair, ring_im, ring_re2)

# A rhomb mesh's tiles (rhombs_near()) are Penrose's
RULE = substitution.RULES["penrose"]
EDGE, VERTEX = surface.EDGE, surface.VERTEX

# Each kind, turn and mirroring's corners A, B, C as ring coefficients (kind, rot, mirror, corner, 4), and its legs AB
# and AC's directions as powers of zeta (kind, rot, mirror, leg)
CORNERS = np.zeros((2, RING.n, 2, 3, 4), dtype=np.int64)
LEGS = np.zeros((2, RING.n, 2, 2), dtype=np.int64)
for _kind in penrose.KINDS:
    for _r in range(RING.n):
        for _m in (0, 1):
            _c = [penrose.orient(_r, bool(_m), v) for v in penrose.PROTOTILE[_kind]]
            CORNERS[_kind, _r, _m] = _c
            for _leg, _i in enumerate((1, 2)):
                LEGS[_kind, _r, _m, _leg] = RING.unit_index(RING.sub(_c[_i], _c[0]))
# zeta^a times a ring element, in coefficients: z zeta^a = z @ TURN[a % 10]
TURN = np.array([[RING.power(a + j) for j in range(4)] for a in range(RING.n)], dtype=np.int64)
# A rhomb's A' direction: zeta^4 for a thin one (144 degrees), zeta^2 for a thick one (72)
E2 = {penrose.THIN: 4, penrose.THICK: 2}
# A level's faces are re-rooted (Level.reroot()) before a build at a tile among theirs once any is this far from its
# root, in ring coefficients
REROOT = 64
# A giant level with no piece within the window round the camera (add_levels()) has none within this share of the
# nearer of the window and its nearest piece round that camera, for as long as that holds (its tiles are only kept
# out to the window; an edge piece's frame isn't a rotation, so its distances are only near the true ones). Every
# rebuild of the giant window looked at every giant level again, nearly all with nothing near: in a fast dive a
# rebuild each loop, ~10 ms a level, 0.24 s a tick 48 loops down (2026-11-19, 5 to 9 frames a second)
NO_PIECES_SHARE = 0.5
# Substitution levels with frames are at least this many times finer than the next coarser with them
FRAME_SPACING = 2.5
# The level 0 a parts map shows whole while its corners are this few (and the map shallow: surface_substitution.FLOAT_N)
WHOLE_POINTS = 20000


def turned(z, a):
    return RING.rot(z, a)


class FaceSides:
    """A face's sides at level d, for corners given from its root (ring coefficients): for each side (from its corner
    s to s + 1, the loop being B, A, C, A'), the turn a and target: a corner z = root + rel is on side s when
    Im(zeta^a rel) is the target (p, q); and how far along the side such a corner is: 2 t = sign 2 Re(zeta^a rel) +
    base (side_t())"""

    def __init__(self, faces, f, root, d):
        self.k2 = k2 = E2[int(faces.kind[f])]
        self.length = L = RULE.power(d)                   # phi^d: a side's length in level units
        self.root = root
        self.turns = turns = (0, -k2, 0, -k2)
        constants = (RING.zero, turned(L, -k2), turned(L, k2), RING.zero)
        self.targets = []
        for a, K in zip(turns, constants):
            kp, kq = ring_im(K)
            rp, rq = ring_im(turned(root, a))
            self.targets.append((int(kp - rp), int(kq - rq)))
        self.length2 = L2 = tuple(int(v) for v in ring_re2(L))
        r0 = [int(v) for v in ring_re2(root)]
        rk = [int(v) for v in ring_re2(turned(root, -k2))]
        far2 = [int(v) for v in ring_re2(RING.add(L, turned(L, k2)))]
        # Side 1 starts at A = L, which turned by zeta^-k2 is L zeta^-k2 (not L: a side's start is where 2 t is 0)
        a2 = [int(v) for v in ring_re2(turned(L, -k2))]
        self.signs = (1, 1, -1, -1)
        self.bases = [tuple(r0), (rk[0] - a2[0], rk[1] - a2[1]), (far2[0] - r0[0], far2[1] - r0[1]),
                      (L2[0] - rk[0], L2[1] - rk[1])]

    def side_t(self, s, rel):
        """2 t, t how far a corner (rel from the root, on side s) is from the side's start, as (p, q) Python ints"""
        p, q = ring_re2(turned(tuple(int(c) for c in rel), self.turns[s]))
        base = self.bases[s]
        return int(self.signs[s] * p + base[0]), int(self.signs[s] * q + base[1])


def level_sides(level, faces, f):
    """Face f's FaceSides at level, kept on the level by its root (a re-rooted face gets new ones)"""
    kept = level.__dict__.setdefault("_sides", {})
    root = level.roots[f][0]
    found = kept.get(f)
    if found is None or found.root != root:
        found = kept[f] = FaceSides(faces, f, root, level.d)
    return found


def corner_keys(top, faces, f, sides, s, rel):
    """The key of a corner on side s of face f (rel from the face's root): (VERTEX, v) at a mesh corner, else (EDGE,
    the mesh edge, 2 t from its smaller vertex as P, Q)"""
    t = sides.side_t(s, rel)
    e = top._side_edge[f][s]
    forward = top._side_forward[f][s]
    L = sides.length2
    if not forward:
        t = (L[0] - t[0], L[1] - t[1])
    a, b = top.edges[e]
    if t == (0, 0):
        return (VERTEX, a, 0, 0)
    if t == L:
        return (VERTEX, b, 0, 0)
    return (EDGE, e, t[0], t[1])


def topology_arrays(top):
    """A rhomb mesh's sides as arrays, kept on its Topology: each face's sides' edges and whether each runs from its
    edge's smaller vertex (F, 4), each edge's vertices (E, 2), and the lowest numbered face on each edge and vertex"""
    found = getattr(top, "_substitution_arrays", None)
    if found is None:
        side_edge = np.array(top._side_edge, dtype=np.int64)
        side_forward = np.array(top._side_forward, dtype=bool)
        edges = np.array(top.edges, dtype=np.int64)
        edge_home = np.array([min(g for g, _s, _f in sides) for sides in top.edge_sides], dtype=np.int64)
        vertex_home = np.array([min((g for g, _c in vf), default=-1) for vf in top.vertex_faces], dtype=np.int64)
        found = top._substitution_arrays = (side_edge, side_forward, edges, edge_home, vertex_home)
    return found


unique_rows = SS.unique_rows

def fits(*values):
    return all(abs(v) < FIT for v in values)


# Whole numbers that a level's ring places from its faces' roots, and their sides' targets and bases, are worked with
# in int64 while they're under this (deep levels' are far bigger: their corners on a face's side go one at a time)
FIT = 1 << 60


class Near:
    """A level's corners and edges near a point (lattice_near()): each corner's map place (this world's), its normal and
    slope, its home face, and how to place it exactly (a face it's on and its ring point there, from the face's root);
    each one's neighbors (rows, -1 for none, the first exact: see the module's docstring) and the edges (pairs of rows,
    each once)"""


def lattice_near(world, faces, level, center, reach):
    """Level's corners and edges whose tiles come within reach of center (map units), as a Near: reach in the level's
    ideal units (its tiles' edges), as far in map units as the face nearest center stretches them at most, and a tenth
    more. A rhomb mesh's tiles are its faces' halves; a girih mesh's are cut to its faces (pieces_near())."""
    rule = level.rule
    turn = rule.orient[level.rot, level.mirror.astype(np.int64)]
    cents = level.rel + np.einsum("tij,tj->ti", turn, rule.middles[level.kind])
    rows = np.arange(len(level.face))
    if len(rows) and math.isfinite(reach):
        with np.errstate(over="ignore", invalid="ignore"):
            places = level.map_points(world, faces, rows, cents)
            size = level.scale * faces.sigma_hi[level.face]
            away = np.linalg.norm(places - center, axis=1)
            stretch = level.scale * float(faces.sigma_hi[level.face[int(np.argmin(away))]])
            near = away - rule.radii[level.kind] * size <= 1.1 * reach * stretch
        rows = rows[near]
    if faces.clip:
        return pieces_near(world, faces, level, rows)
    return rhombs_near(world, faces, level, rows)


def copy_normals(world, faces, level, rows, ideal):
    """The surface's normal at each of ideal's points (t, k, 2: level units from the faces' roots of tiles rows), as
    its tile's face has it: (t k, 3)"""
    top = world.topology
    face = level.face[rows]
    k = ideal.shape[1]
    _place, ur, vr = level.face_params(world, faces, rows) if len(rows) else (None, np.zeros(0), np.zeros(0))
    small = level.rule.shrink_float(level.d)
    sin, cos = faces.ev_y[face], faces.ev_x[face]
    dv = ideal[..., 1] / sin[:, None]
    du = ideal[..., 0] - dv * cos[:, None]
    u = np.clip(ur[:, None] + small * du, 0.0, 1.0)
    v = np.clip(vr[:, None] + small * dv, 0.0, 1.0)
    fu = np.repeat(face, k)
    ju = top.eu[fu] + v.reshape(-1)[:, None] * top.w[fu]
    jv = top.ev[fu] + u.reshape(-1)[:, None] * top.w[fu]
    normals = np.cross(ju, jv)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-300)
    return normals


def finish_near(out, count, vertex_of, copy_face, normals, legs, directions, leg_faces, group=None):
    """A Near's normals, slopes, edges and neighbors from its corners' copies (vertex_of: each copy's corner, -1 for
    none; copy_face, normals: its face and the normal there) and its tiles' edges (legs: pairs of copies' corners, each
    one's direction from its first, a power of zeta in its face's frame, leg_faces). Normals and slopes: each face a
    corner is on once, as surface.Topology.key_normal() and key_slope() have them. A corner's first neighbor is along
    the least of its edges' directions among its home face's (or, with group, among the faces in its home's group,
    which share one frame)."""
    held = vertex_of >= 0
    first_pair, _inverse = unique_rows(np.stack([vertex_of[held], copy_face[held]], axis=1))
    pair_vertex = vertex_of[held][first_pair]
    pair_normals = normals[held][first_pair]
    total = np.zeros((count, 3))
    np.add.at(total, pair_vertex, pair_normals)
    faces_on = np.bincount(pair_vertex, minlength=count)
    length = np.linalg.norm(total, axis=1)
    out.normals = total / np.maximum(length, 1e-300)[:, None]
    cancel = ~(length > surface.NORMALS_CANCEL * faces_on)
    for vtx in np.flatnonzero(cancel).tolist():
        out.normals[vtx] = pair_normals[np.flatnonzero(pair_vertex == vtx)[0]]
    c = np.sum(out.normals[pair_vertex] * pair_normals, axis=1)
    tilt = np.sqrt(np.maximum(1.0 - c * c, 0.0)) / np.maximum(c, 0.2)
    out.slopes = np.zeros(count)
    np.maximum.at(out.slopes, pair_vertex, np.where(faces_on[pair_vertex] > 1, tilt, 0.0))
    # Edges, each once
    sorted_legs = np.sort(legs, axis=1)
    first_edge, _inverse = unique_rows(sorted_legs)
    edges = sorted_legs[first_edge] if len(legs) else np.zeros((0, 2), np.int64)
    out.edges = edges
    # Each corner's first neighbor: along the least of the directions (zeta^k, from it) of its edges in its home face
    tails = np.concatenate([legs[:, 0], legs[:, 1]])
    heads = np.concatenate([legs[:, 1], legs[:, 0]])
    ks = np.concatenate([directions, (directions + RING.n // 2) % RING.n])
    both_faces = np.concatenate([leg_faces, leg_faces])
    if group is None:
        in_home = both_faces == out.home[tails]
    else:
        in_home = group[both_faces] == group[out.home[tails]]
    order = np.lexsort((ks, ~in_home, tails))
    tails_sorted = tails[order]
    starts = np.flatnonzero(np.r_[True, tails_sorted[1:] != tails_sorted[:-1]]) if len(tails_sorted) else []
    first_neighbor = np.full(count, -1, dtype=np.int64)
    first_neighbor[tails_sorted[starts]] = heads[order][starts]
    # Every neighbor from the edges, the first one first: each one's place in its corner's row
    ends = np.concatenate([edges, edges[:, ::-1]])
    is_first = ends[:, 1] == first_neighbor[ends[:, 0]]
    ends = ends[np.lexsort((ends[:, 1], ~is_first, ends[:, 0]))]
    group_start = np.r_[0, np.flatnonzero(ends[1:, 0] != ends[:-1, 0]) + 1] if len(ends) else np.zeros(0, np.int64)
    rank = np.arange(len(ends)) - np.repeat(group_start, np.diff(np.r_[group_start, len(ends)]))
    width = int(max(rank.max() + 1 if len(rank) else 1, 1))
    neighbors = np.full((count, width), -1, dtype=np.int64)
    neighbors[ends[:, 0], rank] = ends[:, 1]
    out.neighbors = neighbors


def rhombs_near(world, faces, level, rows):
    """lattice_near() of a rhomb mesh (Penrose): each half's corners A, B, C and its legs AB and AC, the rhombs' edges
    (its base BC is the rhomb's inside); corners on a face's side keyed by its mesh edge (see the module's docstring)"""
    top = world.topology
    side_edge, side_forward, mesh_edges, edge_home, vertex_home = topology_arrays(top)
    out = Near()
    face, kind = level.face[rows], level.kind[rows].astype(np.int64)
    rot, mirror = level.rot[rows].astype(np.int64), level.mirror[rows].astype(np.int64)
    rel = level.coef[rows][:, None, :] + CORNERS[kind, rot, mirror]                    # (t, 3, 4)
    T = len(rows)
    # Ideal places from the faces' roots, then map places and normals of every corner as its tile has it
    ideal = np.stack([rel @ RULE.basis_x, rel @ RULE.basis_y], axis=-1)                # (t, 3, 2)
    maps = level.map_points(world, faces, rows, ideal) if T else np.zeros((0, 3, 3))
    normals = copy_normals(world, faces, level, rows, ideal)
    fu = np.repeat(face, 3)
    # Which corners are on a face's side, exactly: Im(zeta^a rel) at the side's target, all corners at once
    flat_rel = rel.reshape(-1, 4)
    fs, face_index = np.unique(fu, return_inverse=True)
    sides = [level_sides(level, faces, int(f)) for f in fs]
    k2 = np.array([s.k2 for s in sides], dtype=np.int64)[face_index]
    targets = np.zeros((len(fs), 4, 2), dtype=np.int64)
    bases = np.zeros((len(fs), 4, 2), dtype=np.int64)
    valid = np.zeros((len(fs), 4), dtype=bool)
    quick = np.zeros((len(fs), 4), dtype=bool)
    for i, fsides in enumerate(sides):
        for s in range(4):
            if fits(*fsides.targets[s]):
                targets[i, s] = fsides.targets[s]
                valid[i, s] = True
                if fits(*fsides.bases[s], *fsides.length2):
                    bases[i, s] = fsides.bases[s]
                    quick[i, s] = True
    turned_rel = {a: flat_rel @ TURN[a % RING.n] for a in (0, -2, -4)}
    im = {a: np.stack(im_pair(c), axis=-1) for a, c in turned_rel.items()}
    re2 = {a: np.stack(re2_pair(c), axis=-1) for a, c in turned_rel.items()}
    by_k2 = lambda table: np.where((k2 == 4)[:, None], table[-4], table[-2])
    side_of = np.full(T * 3, -1, dtype=np.int64)
    for s in range(4):
        lhs = im[0] if s in (0, 2) else by_k2(im)
        on = valid[face_index, s] & np.all(lhs == targets[face_index, s], axis=1) & (side_of < 0)
        side_of[on] = s
    boundary = np.flatnonzero(side_of >= 0)
    # Corners inside their faces, by their face and ring coefficients; the ones on sides by their keys
    vertex_of = np.full(T * 3, -1, dtype=np.int64)
    inside = np.flatnonzero(side_of < 0)
    first_inside, inverse = unique_rows(np.concatenate([fu[inside, None], flat_rel[inside]], axis=1))
    vertex_of[inside] = inverse
    count = len(first_inside)
    first_copy = list(inside[first_inside])
    homes = list(fu[inside[first_inside]])
    keys = [None] * count
    if len(boundary):
        bs, bf, bi = side_of[boundary], fu[boundary], face_index[boundary]
        fast = quick[bi, bs]
        # Along their sides in int64, where the numbers fit: 2 t = sign 2 Re(zeta^a rel) + base, from the side's start,
        # then from its mesh edge's smaller vertex
        lhs = np.where(np.isin(bs, (0, 2))[:, None], re2[0][boundary], by_k2(re2)[boundary])
        signs = np.array([1, 1, -1, -1], dtype=np.int64)[bs]
        t = signs[:, None] * lhs + bases[bi, bs]
        L2 = np.array([sides[0].length2], dtype=np.int64) if fits(*sides[0].length2) else np.zeros((1, 2), np.int64)
        forward = side_forward[bf, bs]
        t = np.where(forward[:, None], t, L2 - t)
        e = side_edge[bf, bs]
        at_start = np.all(t == 0, axis=1)
        at_end = np.all(t == L2, axis=1)
        codes = np.stack([np.where(at_start | at_end, VERTEX, EDGE),
                          np.where(at_start, mesh_edges[e, 0], np.where(at_end, mesh_edges[e, 1], e)),
                          np.where(at_start | at_end, 0, t[:, 0]), np.where(at_start | at_end, 0, t[:, 1])], axis=1)
        by_key = {}
        quick_rows = np.flatnonzero(fast)
        first_b, inverse_b = unique_rows(codes[quick_rows])
        ids = np.arange(count, count + len(first_b), dtype=np.int64)
        for row in first_b.tolist():
            key = tuple(int(c) for c in codes[quick_rows[row]])
            by_key[key] = count
            first_copy.append(boundary[quick_rows[row]])
            homes.append(int(edge_home[key[1]] if key[0] == EDGE else vertex_home[key[1]]))
            keys.append(key)
            count += 1
        vertex_of[boundary[quick_rows]] = ids[inverse_b]
        # The rest one at a time, in Python's whole numbers (a deep level's, near a mesh edge)
        for i in boundary[~fast].tolist():
            f = int(fu[i])
            key = corner_keys(top, faces, f, sides[int(face_index[i])], int(side_of[i]), flat_rel[i])
            row = by_key.get(key)
            if row is None:
                row = by_key[key] = count
                count += 1
                first_copy.append(i)
                homes.append(int(edge_home[key[1]] if key[0] == EDGE else vertex_home[key[1]]))
                keys.append(key)
            vertex_of[i] = row
    first_copy = np.array(first_copy, dtype=np.int64)
    out.places = maps.reshape(-1, 3)[first_copy] if count else np.zeros((0, 3))
    out.copy_face = fu[first_copy] if count else np.zeros(0, np.int64)
    out.copy_rel = flat_rel[first_copy] if count else np.zeros((0, 4), np.int64)
    out.keys = keys
    out.home = np.array(homes, dtype=np.int64)
    # Edges: each half's legs AB and AC, with their directions in their tiles' faces
    corner_v = vertex_of.reshape(T, 3)
    legs = np.concatenate([corner_v[:, [0, 1]], corner_v[:, [0, 2]]])
    directions = np.concatenate([LEGS[kind, rot, mirror, 0], LEGS[kind, rot, mirror, 1]])
    finish_near(out, count, vertex_of, fu, normals, legs, directions, np.concatenate([face, face]))
    out.level, out.world_text = level, level.world_text
    return out



# ----- a girih mesh's corners: faces in one frame, and the gluings between frames -----

class Atlas:
    """A girih mesh's faces (Faces.labels) as frames of the tiling's plane: faces whose labels agree at the corners
    they share are one group, labeled in one frame. Across a mesh edge where they don't (the Darb sphere's rim, between
    the pillow's two copies), the gluing is a turn and a shift of the plane, z -> zeta^k z + c, read off the edge's two
    corners on either side: crossing[(f, s)] = (g, k, c), face f's side s to face g. Every tile edge is a mirror line
    of everything drawn across it (surface_substitution.py), so each level's tiles carry across a gluing whole."""

    def __init__(self, top, faces):
        count = len(faces.labels)
        parent = list(range(count))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        label_at = [dict(zip(top._corners[f], faces.labels[f])) for f in range(count)]
        glued = []
        for f in range(count):
            corners = top._corners[f]
            n = len(corners)
            for s in range(n):
                g = top.across(f, s)
                if g is None:
                    continue
                a, b = corners[s], corners[(s + 1) % n]
                fa, fb, ga, gb = label_at[f][a], label_at[f][b], label_at[g][a], label_at[g][b]
                if (fa, fb) == (ga, gb):
                    parent[find(f)] = find(g)
                    continue
                along_f, along_g = RING.sub(fb, fa), RING.sub(gb, ga)
                k = next((k for k in range(RING.n) if RING.rot(along_f, k) == along_g), None)
                if k is None:
                    raise ValueError(f"parts_substitution: faces {f} and {g} meet along sides of different lengths")
                glued.append((f, s, g, k, RING.sub(ga, RING.rot(fa, k))))
        roots = [find(f) for f in range(count)]
        index = {r: i for i, r in enumerate(sorted(set(roots)))}
        self.group = np.array([index[r] for r in roots], dtype=np.int64)
        self.crossing = {(f, s): (g, k, c) for f, s, g, k, c in glued}
        self.glued = np.zeros(count, dtype=bool)          # faces with a side on a gluing
        self.glued[[f for f, _s, _g, _k, _c in glued]] = True


def atlas_of(top, faces):
    found = getattr(faces, "_atlas", None)
    if found is None or found[0] is not top:
        found = faces._atlas = (top, Atlas(top, faces))
    return found[1]


# A corner within this many of its tiles' edges of its face counts as on it (exactly, the corners on a side are on
# it; the rest are a tile's algebraic distance from it at least, far more than this)
ON_SIDE = 1e-6


class Prototiles:
    """A rule's tiles as ring coefficients, for a level's corners at once: each kind, turn and mirroring's corners
    (kind, rot, mirror, corner, 4, padded with copies of the last), each kind's corner count, and each edge's direction
    (corner i to i + 1, a power of zeta)"""

    def __init__(self, rule):
        m = rule.module
        width = max(len(m.PROTOTILE[kind]) for kind in rule.kinds)
        self.corners = np.zeros((len(rule.kinds), RING.n, 2, width, 4), dtype=np.int64)
        self.directions = np.zeros((len(rule.kinds), RING.n, 2, width), dtype=np.int64)
        self.count = np.array([len(m.PROTOTILE[kind]) for kind in rule.kinds], dtype=np.int64)
        for kind in rule.kinds:
            for r in range(RING.n):
                for mi in (0, 1):
                    tile = (kind, r, bool(mi), RING.zero)
                    c = m.polygon(tile) if hasattr(m, "polygon") else [penrose.place(tile, v) for v in m.PROTOTILE[kind]]
                    c = list(c) + [c[-1]] * (width - len(c))
                    self.corners[kind, r, mi] = c
                    n = self.count[kind]
                    for i in range(n):
                        self.directions[kind, r, mi, i] = RING.unit_index(RING.sub(c[(i + 1) % n], c[i]))


_prototiles = {}


def prototiles_of(rule):
    found = _prototiles.get(rule.name)
    if found is None:
        found = _prototiles[rule.name] = Prototiles(rule)
    return found


def pieces_near(world, faces, level, rows):
    """lattice_near() of a girih mesh: each tile's corners and edges, the tiles cut to their faces. A tile is listed
    by every face it reaches into, each in its own frame, so a corner has copies: they're one corner by their ring
    point in their group's frame (Atlas), placed from the copy whose face holds it, and a corner past its group's edge
    is the next group's, through the gluing (its own tiles have it too, if they're kept). A corner on a gluing is the
    lowest group's."""
    top = world.topology
    rule = level.rule
    atlas = atlas_of(top, faces)
    tiles = prototiles_of(rule)
    out = Near()
    face, kind = level.face[rows], level.kind[rows].astype(np.int64)
    rot, mirror = level.rot[rows].astype(np.int64), level.mirror[rows].astype(np.int64)
    width = tiles.corners.shape[3]
    T = len(rows)
    rel = level.coef[rows][:, None, :] + tiles.corners[kind, rot, mirror]               # (t, k, 4)
    ideal = np.stack([rel @ rule.basis_x, rel @ rule.basis_y], axis=-1)                # (t, k, 2)
    real = (np.arange(width)[None, :] < tiles.count[kind][:, None]).reshape(-1)
    # How far each copy is past each of its face's sides (level units; a triangle's third side twice)
    side_normals, offsets = level.face_sides(faces, rows) if T else (np.zeros((0, 4, 2)), np.zeros((0, 4)))
    past = (np.einsum("tsj,tkj->tks", side_normals, ideal) - offsets[:, None, :]).reshape(-1, 4)
    outside = past.max(axis=1)
    fu = np.repeat(face, width)
    flat_rel = rel.reshape(-1, 4)
    # Each copy in its group's frame, from a point of that frame near the drone (the first face's root): level d's
    # point Z of face f's frame is zeta^turn Z + labels[0] LAMBDA^d there
    grow = rule.power(level.d)
    frame_of = {}
    level_faces, face_index = np.unique(fu, return_inverse=True)
    shifts = np.zeros((len(level_faces), 4), dtype=np.int64)
    for i, f in enumerate(level_faces.tolist()):
        g = int(atlas.group[f])
        z = RING.add(RING.rot(level.roots[f][0], int(faces.turn[f])), RING.mul(faces.labels[f][0], grow))
        frame_of.setdefault(g, z)
        shift = RING.sub(z, frame_of[g])
        if not fits(*shift):
            raise ValueError(f"parts_substitution: face {f}'s root is too far from its group's ({level.d})")
        shifts[i] = shift
    keys = np.einsum("cj,cjl->cl", flat_rel, TURN[faces.turn[fu]]) + shifts[face_index.reshape(-1)]
    groups = atlas.group[fu]
    copies = np.flatnonzero(real)
    first, inverse = unique_rows(np.concatenate([groups[copies, None], keys[copies]], axis=1))
    count = len(first)
    corner_of = np.full(T * width, -1, dtype=np.int64)
    corner_of[copies] = inverse
    # Each corner's best copy: the one most inside its face
    order = np.lexsort((outside[copies], inverse))
    starts = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1] if len(order) else np.zeros(0, np.int64)
    best = copies[order[starts]]
    target = np.arange(count)
    keep = np.ones(count, dtype=bool)
    beyond = outside[best] > ON_SIDE
    if atlas.crossing:
        # Corners past a gluing, or on one to a lower group, are the other group's (if it has them here): through the
        # gluing of a side any copy of theirs is on, or past (the nearest past first)
        by_key = {(int(groups[c]), *keys[c].tolist()): i for i, c in enumerate(best.tolist())}
        by_corner = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[by_corner], np.arange(count + 1))
        on_side = np.abs(outside[best]) <= ON_SIDE
        glued = np.zeros(count, dtype=bool)
        glued[inverse[atlas.glued[fu[copies]]]] = True
        keep &= ~(beyond & ~glued)
        for i in np.flatnonzero((beyond | on_side) & glued).tolist():
            own = int(groups[best[i]])
            candidates = []
            for c in copies[by_corner[bounds[i]:bounds[i + 1]]].tolist():
                f = int(fu[c])
                for s in range(int(top.sides[f])):
                    found = atlas.crossing.get((f, s))
                    p = past[c, s]
                    if found is not None and (p > -ON_SIDE if beyond[i] else abs(p) <= ON_SIDE):
                        candidates.append((p, found))
            candidates.sort(key=lambda item: item[0])
            moved = False
            for _p, (g, k, glue) in candidates:
                group = int(atlas.group[g])
                if group not in frame_of or (not beyond[i] and group > own):
                    continue
                # zeta^k (key + frame) + glue LAMBDA^d - the other frame
                constant = RING.sub(RING.add(RING.rot(frame_of[own], k), RING.mul(glue, grow)), frame_of[group])
                other = RING.add(RING.rot(tuple(int(v) for v in keys[best[i]]), k), constant)
                j = by_key.get((group, *other))
                if j is not None and not beyond[j]:
                    target[i] = j
                    moved = True
                    break
            if beyond[i] and not moved:
                keep[i] = False
    else:
        keep &= ~beyond
    # Merged and dropped corners renumbered (a corner where three groups meet may be merged twice over)
    final = target.copy()
    while np.any(final[final] != final):
        final = final[final]
    alive = keep & (final == np.arange(count))           # the corners kept, each standing for itself
    number = np.cumsum(alive) - 1
    stands_for = final[np.maximum(corner_of, 0)]
    vertex_of = np.where((corner_of >= 0) & alive[stands_for], number[stands_for], -1)
    count = int(alive.sum())
    # Places from each corner's best copy over all its copies; homes: the lowest face holding it
    held = vertex_of >= 0
    holds = held & (outside <= ON_SIDE)
    order = np.lexsort((outside, vertex_of))
    order = order[held[order]]
    starts = np.r_[0, np.flatnonzero(np.diff(vertex_of[order])) + 1] if len(order) else np.zeros(0, np.int64)
    first_copy = order[starts]
    maps = level.map_points(world, faces, rows, ideal).reshape(-1, 3) if T else np.zeros((0, 3))
    out.places = maps[first_copy] if count else np.zeros((0, 3))
    out.copy_face = fu[first_copy] if count else np.zeros(0, np.int64)
    out.copy_rel = flat_rel[first_copy] if count else np.zeros((0, 4), np.int64)
    home = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(home, vertex_of[holds], fu[holds])
    out.home = np.where(home == np.iinfo(np.int64).max, out.copy_face, home)
    out.keys = None
    normals = copy_normals(world, faces, level, rows, ideal)
    # Edges: each tile's sides, corner i to i + 1, their directions in their group's frame
    corner_v = vertex_of.reshape(T, width)
    nexts = (np.arange(width)[None, :] + 1) % tiles.count[kind][:, None]                # (t, k)
    legs = np.stack([corner_v, np.take_along_axis(corner_v, nexts, axis=1)], axis=-1).reshape(-1, 2)
    directions = ((tiles.directions[kind, rot, mirror] + faces.turn[face][:, None]) % RING.n).reshape(-1)
    leg_faces = np.repeat(face, width)
    whole = real & np.all(legs >= 0, axis=1)
    finish_near(out, count, np.where(holds, vertex_of, -1), fu, normals, legs[whole], directions[whole],
                leg_faces[whole], atlas.group)
    out.level, out.world_text = level, level.world_text
    return out


# ----- exactly: a corner's point in the mesh's units, in Q(sqrt 5) -----

def exact_point(world, faces, level, f, rel):
    """A corner's point (face f, ring coefficients rel from the face's root at level) in the mesh's units, exactly: three
    (a, b) pairs of Fractions, a + b sqrt 5"""
    z = RING.add(level.roots[f][0], tuple(int(c) for c in rel))
    u, v = faces.uv_exact(f, z, level.d)
    top = world.topology
    p0, eu, ev, tw = top.exact[f]
    uv = q_mul(u, v)
    scale = Fraction(1, 1 << top.E)
    return [q_scale(q_add(q_add((Fraction(p0[k]), Fraction(0)), q_scale(u, eu[k])),
                                q_add(q_scale(v, ev[k]), q_scale(uv, tw[k]))), scale) for k in range(3)]


def camera_exact(world, camera):
    """The camera's exact point (map units, float64) in the mesh's units, as three Fractions"""
    return surface.d_fractions(world.camera_point(np.asarray(camera, dtype=float) / world.rho))


# ----- a level's pieces round a point -----

class SubstitutionPieces(parts.LaidOut):
    """A level's pieces round the corner nearest center (map units) out to radius of its cells (the mean distance from
    it to its neighbors), from level (a surface_substitution.Level of this world), as parts.LaidOut lays them out, in the
    level's own units (the mesh's times phi^d) from that corner, origin() in the map; or with an infinite radius,
    every corner and edge level has"""

    def __init__(self, world, faces, level, center, radius, settings, shape):
        self.world, self.faces, self.level, self.d = world, faces, level, level.d
        top = world.topology
        # Each face's tiles from a root among them (a kept level's can be 2^40 of its tiles from it, and float64 then
        # places a corner a ten-thousandth of a tile off, a different way for each root: a level built again turned
        # pieces by that), so every corner is a few tiles from its root, placed to ~1e-14 of a tile
        if math.isfinite(radius):
            level.reroot(REROOT)
        near = self.near = lattice_near(world, faces, level, center, radius + 2.0)
        if not len(near.places):
            # What it was given, for finding out why (seen once on the worker thread, 2026-11-17, not reproduced): the
            # build is then made again in the tick (parts.MapState.failed)
            raise ValueError(f"parts_substitution: no corners near the drone (level {level.d}, {len(level.face)} tiles on "
                             f"faces {sorted(set(level.face.tolist()))[:8]}, kept for {level.window:.4g} round a point "
                             f"{float(np.linalg.norm(np.asarray(center) - level.center)):.4g} from this one, radius "
                             f"{radius:g} cells of {level.scale:.4g}, n {world.n}, world {world.text() == level.world_text})")
        middle = int(np.argmin(np.linalg.norm(near.places - center, axis=1)))
        self.middle_place = near.places[middle]
        points = (near.places - self.middle_place) / level.scale
        # Within radius of the mean step from the middle corner to its neighbors
        nb = near.neighbors[middle]
        nb = nb[nb >= 0]
        step = float(np.mean(np.linalg.norm(points[nb], axis=1))) if len(nb) else 1.0
        inner = np.linalg.norm(points, axis=1) <= radius * step
        inner[middle] = True
        edges = near.edges[inner[near.edges[:, 0]] & inner[near.edges[:, 1]]]
        self.lay_out(inner, near.neighbors, edges, middle, points, near.normals, settings, shape, near.slopes)
        self.middle_row = middle
        self._point = None

    def scale(self, rho=None):
        return self.level.scale

    def point(self, row):
        """Corner row's exact point, the mesh's units (Q(sqrt 5) pairs)"""
        near = self.near
        return exact_point(self.world, self.faces, self.level, int(near.copy_face[row]), near.copy_rel[row])

    def origin(self, placer=None):
        """The middle corner in the map (map units from its origin), as its tiles place it (Level.map_points(): from its
        face's root, rounded once from its exact place, a few tiles off), so that every piece's place is its corners',
        whichever corner a build is round"""
        return self.middle_place

    def camera(self, camera, placer=None):
        """A camera (map units, float64) in the pieces' units from the middle corner, from both exact points: (that,
        0.0)"""
        if self._point is None:
            self._point = self.point(self.middle_row)
        grow = self.level.rule.q_grow(self.d)
        cam = camera_exact(self.world, camera)
        return np.array([q_float(q_mul(q_sub((c, Fraction(0)), p), grow)) for c, p in zip(cam, self._point)]), 0.0


def exact_piece(world, pieces, row, camera_point, window, shape, settings, phases, h, local=None):
    """A giant level's piece (row of pieces) worked out exactly for the window (surface.exact_piece_at()): its frame
    and size from its corners' exact differences, rounded once, its normals, stretch and scales as laid out, and the
    camera's exact offset from its first corner, sqrt 5 taken to as many bits as the depth needs. With local (the
    camera in the pieces' units from their middle corner, pieces.camera()), it's all from the corners as laid out, in
    float64."""
    kind = int(pieces.kinds[row])
    rows = pieces.piece_rows(row)
    normals = [tuple(float(x) for x in pieces.normals[r]) for r in rows]
    stretch = None
    scales = None
    if kind == 0:
        stretch = parts.foot_stretch(shape, float(pieces.near.slopes[rows[0]]))
    else:
        scales = tuple(float(s) for s in pieces.edge_scales[row - pieces.vertex_count])
    rule = pieces.level.rule
    phase = parts.level_phase(pieces.d, phases, lattice_for(rule).level_step)
    if local is not None:
        # The corners as laid out, from the middle corner (to ~1e-14 of a tile, as add_levels()'s tolerance has it)
        points = pieces.points[rows]
        offsets = [tuple(float(x) for x in point - points[0]) for point in points]
        relative = [float(c) for c in np.asarray(local, dtype=float) - points[0]]
        return surface.exact_piece_at(world, pieces.d, kind, offsets, normals, stretch, scales, relative, None,
                                      window, shape, settings, phase, scale=pieces.scale())
    exact = [pieces.point(r) for r in rows]
    P = exact[0]
    grow = rule.q_grow(pieces.d)
    offsets = [tuple(q_float(q_mul(q_sub(p, q), grow)) for p, q in zip(point, P)) for point in exact]
    # Enough bits of sqrt 5: a point's a + b sqrt 5 has a and b about LAMBDA^2d (its ring point's coefficients, times
    # LAMBDA^-d's), cancelling to the mesh's size, and the drone is 2^-n h of the mesh's units (h map units, rho about 1)
    grown = int(2 * pieces.d * rule.log_loop / math.log(2.0))
    bits = grown + world.n + max(0, int(-math.log2(max(h, 1e-300)))) + 128
    relative = [q_fraction(q_sub((c, Fraction(0)), p), bits) for c, p in zip(camera_point, P)]
    unit = q_fraction(rule.q_grow(-pieces.d), bits)
    return surface.exact_piece_at(world, pieces.d, kind, offsets, normals, stretch, scales, relative, unit, window,
                                  shape, settings, phase)


def built_pieces(world, faces, tiles, center, radius, settings, shape):
    """A level's pieces (SubstitutionPieces) built on the worker thread, as parts.write_level() takes them"""
    pieces = SubstitutionPieces(world, faces, tiles, center, radius, settings, shape)
    return world, pieces, pieces.origin(), radius


def tiles_pieces(world, faces, st, level, center, window, radius, settings, shape):
    """built_pieces() of level's tiles covering window round center (map units), as the strapwork keeps them or cut
    here (SS.tiles(): kept for the strapwork's builds too), on the worker thread (SubstitutionLattice.pieces_job())"""
    tiles = SS.tiles(st, world, faces, level, center, window)
    built = built_pieces(world, faces, tiles, center, radius, settings, shape)
    SS.keep_copy(st, tiles)
    return built


# ----- parts.py's view of a Penrose map's lattice -----

class SubstitutionLattice(surface.MeshLattice):
    """parts.py's view of a substitution surface map's lattice, one per rule (surface.MeshLattice is the square and
    triangle maps')"""

    def __init__(self, rule):
        super().__init__(rule.loop)
        self.rule = rule
        # Frames on every level_step-th level, at least FRAME_SPACING apart: Penrose's every other level, phi^2 (on
        # every level they were dense, the user, 2026-11-16), Darb-i Imam's every level, LAMBDA = 8.5
        self.level_step = max(1, math.ceil(math.log(FRAME_SPACING) / rule.log_loop - 1e-9))

    def world(self, obj):
        world = surface.world_of(obj)
        top = world.topology
        world.substitution = (SS.faces_of(obj, top), SS.state_of(obj, top))
        return world

    def unit_level(self, world):
        """The level whose tiles are about a map unit at the anchor"""
        edge = self.edge(world)
        return max(0, math.floor((math.log(world.rho * edge) + world.n * math.log(2.0)) / self.rule.log_loop))

    @staticmethod
    def edge(world):
        top = world.topology
        f = world.face
        return math.sqrt(float(np.linalg.norm(top.eu[f])) * float(np.linalg.norm(top.ev[f])))

    def center(self, world, below, level):
        return world.rho * below.foot

    def first_radius(self, world, level, build, wanted):
        """The coarsest level whole while its corners are few and the map shallow (it's then never built again as the
        drone moves); else out to the next power of 2 of the cells wanted, at most surface.FIRST_REACH"""
        faces, _st = world.substitution
        if world.n < SS.FLOAT_N and len(faces.kind) * self.rule.loop ** (2 * level) <= WHOLE_POINTS:
            return math.inf
        if not wanted > build:
            return build
        return float(min(2 ** math.ceil(math.log2(min(wanted, 2.0 ** 30))), max(surface.FIRST_REACH, build)))

    def level_for(self, world, level, center, window):
        """Level d's tiles covering window round center (map units), a Level of its own: the strapwork's kept level,
        or cut (SS.tiles(), kept for the strapwork too); or the whole level"""
        faces, st = world.substitution
        if math.isinf(window):
            whole = SS.level_zero(world, faces, None, center, math.inf)
            for _ in range(level):
                whole = SS.level_below(world, whole, center)
            return whole
        SS.want_tiles(st, "parts", level, world, center, window, time.perf_counter())
        return SS.tiles(st, world, faces, level, center, window)

    def tiles_for(self, world, level, center, radius):
        """The tiles a build out to radius cells needs (tiles_window())"""
        return self.level_for(world, level, center, math.inf if math.isinf(radius) else
                              self.tiles_window(world, level, radius))

    def pieces(self, world, level, center, radius, settings, shape):
        faces, _st = world.substitution
        tiles = self.tiles_for(world, level, center, radius)
        return SubstitutionPieces(world, faces, tiles, center, radius, settings, shape)

    def pieces_job(self, world, level, center, radius, settings, shape):
        """pieces() for the worker thread (parts.SphereLattice.pieces_job()): the corners found and the pieces laid out
        there (5 to 15 ms a level), on its own world, from the level's tiles as the strapwork keeps them, or cut there
        (tiles_pieces()), and kept for the strapwork too: a Penrose map's levels are cut once for both
        (surface_substitution.tiles(), 2026-11-19). Cut in the tick round a center ahead of the drone (parts.settle()), cuts
        took up to 50 ms, and the strapwork then cut its own levels again for its own windows (2026-11-17); cut in each
        parts build, they were some 40% of its time in a fast dive (2026-11-19)."""
        faces, st = world.substitution
        own = surface.World.from_text(world.text(), world.rho, world.topology)
        center = np.array(center, dtype=float)
        if math.isinf(radius):
            return (built_pieces, own, faces, self.tiles_for(world, level, center, radius), center, radius, settings,
                    shape)
        window = self.tiles_window(world, level, radius)
        SS.want_tiles(st, "parts", level, world, center, window, time.perf_counter())
        return tiles_pieces, own, faces, st, level, center, window, radius, settings, shape

    @staticmethod
    def giant_tiles(world, faces, st, d, below, foot, window):
        """A giant level's tiles for add_levels(), this world's, its own (the pieces re-root it): the kept level's, cut
        down to the window round the foot if it's much wider, and kept (st.giant_tiles) while the kept level has the
        same tiles (SS.origin_of()) and the window holds. The kept levels near the boundary are as wide as the 3D
        parts' instanced builds of them want, some 30,000 tiles, and every giant level's pieces went through all of
        them at every window build: 400 ms a second of a climb's ticks (2026-11-19)."""
        text = world.text()
        kept = st.levels[d]
        if kept.world_text != text:
            kept = st.levels[d] = SS.rebased(world, faces, kept)
        found = st.giant_tiles.get(d)
        if found is not None and found[0] is SS.origin_of(kept) and SS.covers(world, text, found[1], foot, window):
            tiles = found[1]
            if tiles.world_text != text:
                tiles = SS.rebased(world, faces, tiles)
                st.giant_tiles[d] = (found[0], tiles)
            return tiles
        tiles = kept.copy()
        keep = window * SS.SLACK
        if len(tiles.face) > SS.TRIM_MIN and tiles.window > schedule.OVERBUILT * keep:
            centers = SS.Centers(world, faces, below, foot, keep) if world.n >= SS.FLOAT_N else None
            tiles.keep_near(world, faces, centers, keep, foot)
        st.giant_tiles[d] = (SS.origin_of(kept), tiles)
        return tiles

    def tiles_window(self, world, level, radius):
        """How far round its center (map units) a build out to radius cells needs a level's tiles: with room for faces
        stretched more than the anchor's (kept_levels() keeps half as much again, and two tiles)"""
        return (radius + 4.0) * self.cell_sizes(world, [level])[level] * 1.2

    def cell_sizes(self, world, levels):
        """Each level's tile edge at the anchor's face (map units): rho 2^n LAMBDA^-d times its edge"""
        edge = self.edge(world)
        return {d: world.rho * edge * self.rule.shrink_float(d, world.n) for d in levels}

    def add_levels(self, world, camera, q, h, window, levels, settings, shape, phases, placer):
        """The giant levels' pieces within window of the camera (map units), as parts.ExactPieces, finest first: each
        level's pieces round the corner nearest the point below, in float64 against the camera, and those in reach
        worked out for the window (exact_piece()): in float64 where its rounding is well under the drone's height,
        exactly past that"""
        found = []
        faces, st = world.substitution
        text = world.text()
        # Levels known to have nothing round the camera aren't looked at again (NO_PIECES_SHARE)
        camera = np.asarray(camera, dtype=float)
        levels = [d for d in levels if not (d in st.no_pieces and st.no_pieces[d][0] is shape
                                            and SS.covers(world, text, st.no_pieces[d][1], camera, window))]
        if not levels:
            return found
        below = surface.Below(world, np.asarray(q, dtype=float))
        foot = world.rho * below.foot
        camera_point = camera_exact(world, camera)
        reach_cells = max(shape.plan[0], shape.plan[1], shape.past, float(shape.reach.max()))
        SS.kept_levels(world, faces, below, st, max(levels), foot, lambda d: window)
        for d in reversed(levels):
            tiles = self.giant_tiles(world, faces, st, d, below, foot, window)
            cell = self.cell_sizes(world, [d])[d]
            radius = math.ceil(1.5 + 1.25 * window / cell + reach_cells)
            try:
                pieces = SubstitutionPieces(world, faces, tiles, foot, radius, settings, shape)
            except ValueError:
                continue
            local, rounding = pieces.camera(camera)
            scale = pieces.scale()
            rows, distances, tolerance = parts.window_rows(shape, pieces, local, rounding, scale, window)
            if not len(rows):
                least = float(np.abs(distances).min()) * scale if len(distances) else math.inf
                st.no_pieces[d] = (shape, SS.Drawn(text, world.n, camera.copy(),
                                                   NO_PIECES_SHARE * min(least, window), 1.0))
            else:
                st.no_pieces.pop(d, None)
            for row in rows:
                # float64 is enough while its rounding is well under the drone's height (parts.add_levels())
                floats = tolerance[row] * scale < parts.EXACT_SHARE * h
                piece = exact_piece(world, pieces, int(row), camera_point, window, shape, settings, phases, h,
                                    local if floats else None)
                if piece is not None:
                    found.append(piece)
        return found


_lattices = {}


def lattice_for(rule):
    found = _lattices.get(rule.name)
    if found is None:
        found = _lattices[rule.name] = SubstitutionLattice(rule)
    return found


def lattice_of(obj):
    """The map's lattice, for its rule"""
    return lattice_for(SS.rule_of(obj))


def forget():
    pass
