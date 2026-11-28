"""A smooth chart's float points by the core (Rust: eternal_return_core's SmoothChart) against the Python's own
(subdivision.SubdivisionChart._evaluate() with the core turned off), on meshes of every kind a chart takes and on
recorded maps' own: points inside faces, right at their corners, along their sides, past them (the surface beyond),
and nearer and nearer their extraordinary corners, with first and second derivatives, at the depths the add-on asks.

    <Blender's python> -I tools/checks/chart_points.py [--core <folder>] [--tolerance <t>] [<recordings folder> ...]

The meshes made here: a cube (every corner of three quads), a torus of quads (all regular), a tetrahedron, an
octahedron and an icosahedron (Loop's, corners of 3, 4 and 5), a capped cylinder (quads and triangles: Stam and
Loop's borders) and an open box and an open sheet (a rim: borders too). A recordings folder (tools/probes/
record_builds.py's or record_calls.py's) gives its maps' shared Topologies.

The two differ in the last bits (the Python's trees are numpy's matrix products, which add in an order of their own),
and a derivative multiplies that by its piece's size's inverse, once for each order: a piece by an extraordinary
corner is as small as the point is near it (down to 2^-40 of the face), and by a border as small as the depth asked
makes it. So a value passes within tolerance (1e-9) of its size, or within NOISE (2e-15) of the mesh's size times
that inverse to the derivative's order: points to 14 digits everywhere, derivatives to what floats hold of them
there.

Then exactly: the same faces' points at fractions (halves, quarters, ... down to 2^-300 of a face from its corners,
and where the chart has no borders, thirds and sevenths), right at corners, past sides, and at points of Q(sqrt 5)
(a Penrose map's), by _exact_point() and point_q5(): the core's fractions must be the Python's, to the last digit, and
where the Python can't answer (a point off the mesh's rim, a point by a border that no split reaches) the core
mustn't either. And a lattice's steps from its points (_offsets(): the points a few steps of 2^-26 to 2^-300 of the
face, or of 3^-20, from lattice points inside irregular faces, at their corners and beside them), which are floats
again: within tolerance of the biggest of each call's. And tiles' steps (step(): a few offsets from points inside
faces, by their corners and right at them, with steps of 2^-3 of a face down to 2^-60), the same way.

PASS or FAIL at the end, and how long each took.
"""

import importlib
import math
import os
import pickle
import sys
import time
import types
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PACKAGE = "eternal_return"
TOLERANCE = 1e-9
NOISE = 2e-15
ORDERS = (((0, 0),), ((0, 0), (1, 0), (0, 1)), ((2, 0), (1, 1), (0, 2)))
DEPTHS = (10, 14, 24)


def cube():
    positions = [(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return positions, faces


def torus(around=6, along=8):
    positions, faces = [], []
    for i in range(around):
        for j in range(along):
            a, b = 2.0 * math.pi * i / around, 2.0 * math.pi * j / along
            r = 2.0 + 0.7 * math.cos(a)
            positions.append((r * math.cos(b), r * math.sin(b), 0.7 * math.sin(a)))
            faces.append((i * along + j, ((i + 1) % around) * along + j, ((i + 1) % around) * along + (j + 1) % along,
                          i * along + (j + 1) % along))
    return positions, faces


def tetrahedron():
    return [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)], [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]


def octahedron():
    positions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    return positions, faces


def icosahedron():
    p = (1.0 + math.sqrt(5.0)) / 2.0
    positions = [(-1, p, 0), (1, p, 0), (-1, -p, 0), (1, -p, 0), (0, -1, p), (0, 1, p), (0, -1, -p), (0, 1, -p),
                 (p, 0, -1), (p, 0, 1), (-p, 0, -1), (-p, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
             (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]
    return positions, faces


def cylinder(around=6, rings=3):
    """Quads round its side, a fan of triangles on each end"""
    positions, faces = [], []
    for k in range(rings):
        for j in range(around):
            a = 2.0 * math.pi * j / around
            positions.append((math.cos(a), math.sin(a), float(k)))
    for k in range(rings - 1):
        for j in range(around):
            faces.append((k * around + j, k * around + (j + 1) % around, (k + 1) * around + (j + 1) % around,
                          (k + 1) * around + j))
    bottom, top = len(positions), len(positions) + 1
    positions += [(0.0, 0.0, -0.6), (0.0, 0.0, rings - 0.4)]
    for j in range(around):
        faces.append((bottom, (j + 1) % around, j))
        faces.append((top, (rings - 1) * around + j, (rings - 1) * around + (j + 1) % around))
    return positions, faces


def open_box():
    positions, faces = cube()
    return positions, faces[:-1]


def sheet(n=4):
    positions = [(float(i), float(j), 0.3 * math.sin(i) * math.cos(j)) for i in range(n + 1) for j in range(n + 1)]
    faces = [(i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1)
             for i in range(n) for j in range(n)]
    return positions, faces


MESHES = (("cube", cube, "catmull_clark"), ("torus of quads", torus, "catmull_clark"), ("tetrahedron", tetrahedron, "loop"),
          ("octahedron", octahedron, "loop"), ("icosahedron", icosahedron, "loop"), ("capped cylinder", cylinder, "stam_loop"),
          ("open box", open_box, "catmull_clark"), ("open sheet", sheet, "catmull_clark"))


def points_of(chart, rng):
    """Faces and (u, v) to ask a chart for: (name, f, u, v) lists"""
    F = len(chart.sides)
    faces = rng.permutation(F)[:min(F, 40)]
    tri = lambda f, u, v: np.where((chart.sides[f] == 3) & (u + v > 1.0), 1.0 - u, u)
    out = []
    # Inside
    f = rng.choice(faces, 400)
    u, v = rng.random(400), rng.random(400)
    u, v = tri(f, u, v), np.where((chart.sides[f] == 3) & (u + v > 1.0), 1.0 - v, v)
    out.append(("inside", f, u, v))
    # Right at the corners, and along the sides
    cf, cu, cv = [], [], []
    for g in faces.tolist():
        for a, b in chart.schemes[g].corners:
            cf.append(g), cu.append(float(a)), cv.append(float(b))
    out.append(("at corners", np.array(cf), np.array(cu), np.array(cv)))
    f = rng.choice(faces, 200)
    t = rng.random(200)
    side = rng.integers(0, 3, 200)
    u = np.where(side == 0, t, np.where(side == 1, 0.0, np.where(chart.sides[f] == 3, t, 1.0)))
    v = np.where(side == 0, 0.0, np.where(side == 1, t, np.where(chart.sides[f] == 3, 1.0 - t, t)))
    out.append(("on sides", f, u, v))
    # Past the sides: the surface beyond
    f = rng.choice(faces, 300)
    out.append(("past sides", f, rng.uniform(-0.4, 1.4, 300), rng.uniform(-0.4, 1.4, 300)))
    # Nearer and nearer the corners
    nf, nu, nv = [], [], []
    for g in faces[:12].tolist():
        corners = chart.schemes[g].corners
        mu, mv = chart.schemes[g].middle
        for a, b in corners:
            for k in range(1, 50, 3):
                reach = 2.0 ** -k * rng.uniform(0.3, 1.0)
                nf.append(g), nu.append(a + reach * (mu - a) * rng.uniform(0.2, 1.8)), nv.append(b + reach * (mv - b) * rng.uniform(0.2, 1.8))
    out.append(("by corners", np.array(nf), np.array(nu), np.array(nv)))
    # A few points only (the Python walks those one at a time)
    out.append(("a few", faces[:3], rng.random(3) * 0.5, rng.random(3) * 0.5))
    return out


def exact_points(chart, rng):
    """Faces and exact (u, v) to ask a chart for: (f, u, v) with u, v Fractions"""
    F = len(chart.sides)
    faces = rng.permutation(F)[:min(F, 8)].tolist()
    out = []
    for f in faces:
        triangle = chart.sides[f] == 3
        corners = chart.schemes[f].corners
        mu, mv = (Fraction(1, 3), Fraction(1, 3)) if triangle else (Fraction(1, 2), Fraction(1, 2))
        # Inside, at halves and quarters and finer; and at thirds and sevenths where every point has an answer
        for den in (2, 4, 8, 64, 1 << 20) + (() if chart.border else (3, 7, 3 ** 5)):
            a, b = int(rng.integers(0, den + 1)), int(rng.integers(0, den + 1))
            if triangle and a + b > den:
                a, b = den - a, den - b
            out.append((f, Fraction(a, den), Fraction(b, den)))
        # Right at the corners, and nearer and nearer them
        for cu, cv in corners:
            out.append((f, Fraction(cu), Fraction(cv)))
            for k in (3, 11, 40, 300):
                out.append((f, cu + (mu - cu) / 2 ** k, cv + (mv - cv) / 2 ** k))
        # Past the sides
        out.append((f, Fraction(-1, 8), Fraction(3, 8)))
        out.append((f, Fraction(5, 8), Fraction(5, 8) if triangle else Fraction(9, 8)))
    return out


def exactly(own, cored, subdivision, native, rng):
    """The exact questions, the Python's answers against the core's: (how many asked, how many differ, a few lines
    about those, the seconds each took)"""
    def ask(chart, core_on, question):
        subdivision._native = native if core_on else None
        start = time.perf_counter()
        try:
            found = question(chart)
        except subdivision.ExtraordinaryError:
            found = "no answer"
        return found, time.perf_counter() - start
    asked, differ, lines, seconds = 0, 0, [], [0.0, 0.0]
    questions = []
    for f, u, v in exact_points(own, rng):
        questions.append((f"_exact_point({f}, {u}, {v})", lambda chart, f=f, u=u, v=v: chart._exact_point(f, u, v)))
        # (a + b sqrt 5 of the same point's neighborhood: b a small fraction, the point still on or just off the face)
        q = lambda x, b: (x - b * Fraction(2), b * Fraction(8, 9))
        if not own.border:
            questions.append((f"point_q5({f}, {u} ~, {v} ~)",
                              lambda chart, f=f, u=u, v=v: chart.point_q5(f, q(u, Fraction(1, 64)), q(v, Fraction(-1, 128)))))
        questions.append((f"point_q5({f}, {u}, {v})", lambda chart, f=f, u=u, v=v: chart.point_q5(f, u, v)))
    # A lattice's steps: from points of irregular faces, a few steps round each
    for f in [f for f in sorted({f for f, _u, _v in exact_points(own, np.random.default_rng(7))}) if not own.regular[f]][:5]:
        triangle = own.sides[f] == 3
        for M in (1 << 26, 1 << 40, 1 << 300) + (() if own.border else (3 ** 20,)):
            spots = [(M // 3, M // 5), (M // 2, M // 2 - 1 if triangle else M // 2), (M // 7, M // 64)]
            spots += [(0, 0), (M, 0), (0, M), (1, 0), (5, 2), (M - 3, 1), (2, M - 7)]
            for I, J in spots:
                gi, gj = rng.uniform(-3.0, 3.0, 9), rng.uniform(-3.0, 3.0, 9)
                gi[:3], gj[:3] = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
                # (the points kept on the face: a step past its rim has no answer either way, and a corner's steps go in)
                gi, gj = np.clip(gi, -I, M - I), np.clip(gj, -J, M - J - (I if triangle else 0))
                if triangle:
                    over = np.maximum(I + gi + J + gj - M, 0.0)
                    gi, gj = gi - over / 2.0, gj - over / 2.0
                questions.append((f"_offsets({f}, {I}/{M if M < 1 << 60 else '2^' + str(M.bit_length() - 1)}, {J}/...)",
                                  lambda chart, f=f, I=I, J=J, M=M, gi=gi, gj=gj: chart._offsets(f, Fraction(I, M), Fraction(J, M), Fraction(1, M), gi, gj)))
    # Tiles' steps: from points inside faces, by their corners and right at them, coarse and deep
    for small in (2.0 ** -3, 2.0 ** -10, 2.0 ** -24, 2.0 ** -30, 2.0 ** -60):
        faces = rng.permutation(len(own.sides))[:min(len(own.sides), 10)]
        f = np.repeat(faces, 3)
        triangle = own.sides[f] == 3
        ur, vr = rng.uniform(0.05, 0.45, len(f)), rng.uniform(0.05, 0.45, len(f))
        # (every third tile by a corner of its face, every ninth right at it)
        corner = np.array([own.schemes[g].corners[k % own.sides[g]] for k, g in enumerate(f.tolist())], dtype=float)
        middle = np.array([own.schemes[g].middle for g in f.tolist()])
        by = np.arange(len(f)) % 3 == 1
        at = np.arange(len(f)) % 9 == 2
        ur = np.where(by, corner[:, 0] + (middle[:, 0] - corner[:, 0]) * 2.0 ** -9, np.where(at, corner[:, 0], ur))
        vr = np.where(by, corner[:, 1] + (middle[:, 1] - corner[:, 1]) * 2.0 ** -9, np.where(at, corner[:, 1], vr))
        du, dv = rng.uniform(-2.0, 2.0, (len(f), 5)), rng.uniform(-2.0, 2.0, (len(f), 5))
        # (a tile right at a corner steps into its face: out of it there may be no mesh)
        inward = np.sign(middle - corner)
        du = np.where(at[:, None], np.abs(du) * inward[:, :1] * 0.25, du)
        dv = np.where(at[:, None], np.abs(dv) * inward[:, 1:] * 0.25, dv)
        # (a coarse step is two float points' difference over it: their last bits, that much bigger)
        questions.append((f"step(30 tiles, small 2^{int(np.log2(small))})",
                          lambda chart, f=f, ur=ur, vr=vr, du=du, dv=dv, small=small: chart.step(f, ur, vr, du, dv, small),
                          max(TOLERANCE, 64.0 * NOISE / small)))
    for name, question, *allowed in questions:
        a, took_a = ask(own, False, question)
        b, took_b = ask(cored, True, question)
        seconds[0] += took_a
        seconds[1] += took_b
        asked += 1
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            within = allowed[0] if allowed else TOLERANCE
            same = a.shape == b.shape and bool(np.all(np.abs(a - b) <= within * max(float(np.abs(a).max()), 1e-300)))
        else:
            same = type(a) is type(b) and a == b
        if not same:
            differ += 1
            if len(lines) < 4:
                lines.append(f"    {name}: {str(b)[:110]} for {str(a)[:110]}")
    subdivision._native = native
    return asked, differ, lines, seconds


def main():
    global TOLERANCE
    arguments = sys.argv[1:]
    while arguments[:1] and arguments[0] in ("--core", "--tolerance"):
        if arguments[0] == "--core":
            sys.path.insert(1, arguments[1])
        else:
            TOLERANCE = float(arguments[1])
        arguments = arguments[2:]
    package = types.ModuleType(PACKAGE)
    package.__path__ = [os.path.join(REPO, PACKAGE)]
    sys.modules[PACKAGE] = package
    surface = importlib.import_module(PACKAGE + ".surface")
    subdivision = importlib.import_module(PACKAGE + ".subdivision")
    native = subdivision._native
    if getattr(native, "SmoothChart", None) is None:
        print("FAIL: no core with a smooth chart (--core <folder>)")
        sys.exit(1)
    meshes = [(name, (np.array(positions, dtype=np.float32).astype(np.float64), faces, chart, None))
              for name, make, chart in MESHES for positions, faces in [make()]]
    for folder in arguments:
        for name in sorted(os.listdir(folder)):
            if name.startswith("shared_"):
                with open(os.path.join(folder, name), "rb") as f:
                    _key, _module, cls, data = pickle.load(f)
                if cls == "Topology" and data[2] != "bilinear":
                    meshes.append((os.path.basename(os.path.normpath(folder)), data))
    failed = 0
    for name, data in meshes:
        # The same mesh's chart twice: the Python's own, and the core's
        subdivision._native = None
        start = time.perf_counter()
        own = surface.Topology.from_shared(data).chart
        made_own = time.perf_counter() - start
        subdivision._native = native
        start = time.perf_counter()
        cored = surface.Topology.from_shared(data).chart
        made_cored = time.perf_counter() - start
        if cored.core() is None:
            print(f"FAIL {name}: the core took no chart of it")
            failed += 1
            continue
        rng = np.random.default_rng(7)
        worst, asked, seconds = 0.0, 0, [0.0, 0.0]
        scale = float(np.abs(data[0]).max())
        lines = []
        for kind, f, u, v in points_of(own, rng):
            for orders in ORDERS:
                for depth in DEPTHS if own.border else DEPTHS[:1]:
                    subdivision._native = None
                    start = time.perf_counter()
                    wanted = own._evaluate(f, u, v, orders=orders, depth=depth)
                    seconds[0] += time.perf_counter() - start
                    subdivision._native = native
                    start = time.perf_counter()
                    found = cored._evaluate(f, u, v, orders=orders, depth=depth)
                    seconds[1] += time.perf_counter() - start
                    asked += len(f) * len(orders)
                    # How small a point's piece may be, as its size's inverse: by a border what the depth makes it,
                    # elsewhere what its nearness to a corner of its face does
                    corners = [np.array(own.schemes[g].corners, dtype=float) for g in np.asarray(f).tolist()]
                    near = np.array([np.abs(c - (x, y)).max(axis=1).min() for c, x, y in zip(corners, u, v)])
                    inverse = np.full(len(near), 2.0 ** (depth + 2)) if own.border else np.minimum(8.0 / np.maximum(near, 1e-300), 2.0 ** 42)
                    for order, a, b in zip(orders, wanted, found):
                        allowed = TOLERANCE * np.abs(a) + (NOISE * scale * inverse ** sum(order))[:, None]
                        with np.errstate(invalid="ignore"):
                            off = np.abs(a - b) / allowed * TOLERANCE
                        off = np.where((a == b) | (np.isnan(a) & np.isnan(b)), 0.0, off)
                        off = np.where(np.isnan(off), np.inf, off)
                        worst = max(worst, float(off.max()) if off.size else 0.0)
                        if off.size and off.max() > TOLERANCE and len(lines) < 4:
                            row = int(np.argmax(off.max(axis=-1)))
                            lines.append(f"    {kind}, d{order}, depth {depth}: face {int(f[row])} at ({u[row]!r}, {v[row]!r}): "
                                         f"{b[row].tolist()} for {a[row].tolist()}")
        exact_asked, exact_differ, exact_lines, exact_seconds = exactly(own, cored, subdivision, native, rng)
        lines += exact_lines
        ok = worst <= TOLERANCE and not exact_differ
        failed += not ok
        shapes = len(cored._core_shapes)
        print(f"{'same' if ok else 'DIFFERENT'} {name} ({data[2]}, {len(own.sides)} faces, {int((~own.regular).sum())} irregular"
              f"{', borders' if own.border else ''}): {asked} values, the furthest apart {worst / TOLERANCE:.2g} of what's allowed | Python {seconds[0]:.2f} s, "
              f"core {seconds[1]:.2f} s ({shapes} shapes) | charts made in {made_own:.2f} and {made_cored:.2f} s | exactly: "
              f"{exact_asked - exact_differ} of {exact_asked} the same, Python {exact_seconds[0]:.2f} s, core {exact_seconds[1]:.2f} s")
        for line in lines:
            print(line)
    print("FAIL" if failed else "PASS", f"({len(meshes)} meshes; floats within {TOLERANCE:g})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
