"""The subdivision charts' checks (subdivision.py): run in a background Blender, from the repo,

    Blender -b --factory-startup --python tools/checks/subdivision_check.py [-- name ...]

Each mesh is made from Blender's primitives (nothing of ~/blender_tests is needed), its chart built by surface.Topology,
and checked against what the chart is meant to be: PASS or FAIL a line, with the numbers; the exit status is the number
of failures. Names (all by default):

- cube: Catmull-Clark, corners of 3 faces (every face irregular)
- torus: Catmull-Clark, every corner of 4 (every face regular)
- cylinder: Catmull-Clark, open at both ends (the rim's B-spline)
- ico, tetra, octa: Loop (Warren's weights), corners of 5, 3 and 4 triangles
- trigrid: Loop, an open grid of triangles, bent
- uvsphere: Stam-Loop, quads with triangle fans at the poles (borders between quads and triangles)
- monkey: Stam-Loop, Suzanne: open eye holes and a corner of two quads on two edges (her chin)

What's checked on each:
- split: the mesh split three times exactly (its own scheme's subdivide()), every vertex's limit point against the
  chart's exact point there, and its float point
- edges: two faces' exact points along the side they share, the same
- step: step() (offsets from a point, the motif drawer's) against exact differences, coarse (2^-2) and deep (2^-12)
- across: just past a face's side and just inside it, the same surface (relocated()): position and first derivatives
  (by a border, where float derivatives come from rough pieces, to a hundredth)
- opensubdiv (quads only): the chart's points against Blender's own subdivision surface's limit (OpenSubdiv), level 5
"""

import importlib
import os
import random
import sys
import time
import types
from fractions import Fraction

import bpy
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "eternal_return")
package = types.ModuleType("er_checks")
package.__path__ = [REPO]
sys.modules["er_checks"] = package
SU = importlib.import_module("er_checks.surface")
SD = importlib.import_module("er_checks.subdivision")

MESHES = ("cube", "torus", "cylinder", "ico", "tetra", "octa", "trigrid", "uvsphere", "monkey")
CHARTS = {"cube": "catmull_clark", "torus": "catmull_clark", "cylinder": "catmull_clark", "ico": "loop",
          "tetra": "loop", "octa": "loop", "trigrid": "loop", "uvsphere": "stam_loop", "monkey": "stam_loop"}
failures = []


def report(name, check, ok, text):
    print(f"{'PASS' if ok else 'FAIL'} {name} {check}: {text}")
    sys.stdout.flush()
    if not ok:
        failures.append((name, check))


def made(name):
    """The mesh's Blender object, about 20 units across"""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    if name == "cube":
        bpy.ops.mesh.primitive_cube_add(size=20.0)
    elif name == "torus":
        bpy.ops.mesh.primitive_torus_add(major_segments=16, minor_segments=8, major_radius=10.0, minor_radius=4.0)
    elif name == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=10.0, depth=20.0, end_fill_type='NOTHING')
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.subdivide(number_cuts=2)
        bpy.ops.object.mode_set(mode='OBJECT')
    elif name == "ico":
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=10.0)
    elif name in ("tetra", "octa"):
        if name == "tetra":
            verts = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
            faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
        else:
            verts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata([tuple(10.0 * c for c in v) for v in verts], [], faces)
        mesh.update()
        bpy.context.scene.collection.objects.link(bpy.data.objects.new(name, mesh))
        bpy.context.view_layer.objects.active = bpy.data.objects[name]
    elif name == "trigrid":
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=7, y_subdivisions=7, size=20.0)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.quads_convert_to_tris()
        bpy.ops.object.mode_set(mode='OBJECT')
        for v in bpy.context.object.data.vertices:
            v.co.z = 0.02 * (v.co.x ** 2 - 0.5 * v.co.y ** 2)
    elif name == "uvsphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=10.0)
    elif name == "monkey":
        bpy.ops.mesh.primitive_monkey_add(size=20.0)
    return bpy.context.view_layer.objects.active


def check_split(name, top):
    ch = top.chart
    loops = [tuple(loop) for loop in top._corners]
    count = len(ch.ints)
    mesh = ch.mesh_class({x: ch.ints[x] for x in range(count)}, {x: ch._valence[x] for x in range(count)}, loops,
                         1 << ch.E, ch._boundary)
    place = [(f, Fraction(0), Fraction(0), Fraction(1), 1) for f in range(len(loops))]
    for _ in range(3):
        finer, children = mesh.subdivide()
        new = [None] * len(finer.loops)
        for t, row in enumerate(children):
            f, u0, v0, size, sign = place[t]
            for c, child in enumerate(row):
                if child is not None:
                    (ou, ov), turn = ch.schemes[f].children[c]
                    new[child] = (f, u0 + sign * size * Fraction(ou, 2), v0 + sign * size * Fraction(ov, 2),
                                  size / 2, sign * turn)
        mesh, place = finer, new
    random.seed(2)
    bad = checked = 0
    worst = 0.0
    seen = set()
    for t in random.sample(range(len(mesh.loops)), min(2000, len(mesh.loops))):
        f, u0, v0, size, sign = place[t]
        for k, y in enumerate(mesh.loops[t]):
            if (y, f) in seen or not mesh.complete(y):
                continue
            seen.add((y, f))
            cu, cv = ch.schemes[f].corners[k]
            u, v = u0 + sign * size * cu, v0 + sign * size * cv
            want = mesh.limit(y)
            checked += 1
            bad += tuple(ch._exact_point(f, u, v)) != tuple(want)
            got = ch.points(np.array([f]), np.array([float(u)]), np.array([float(v)]))[0]
            worst = max(worst, float(np.abs(got - np.array([float(c) for c in want])).max()))
    report(name, "split", bad == 0 and worst < 1e-9, f"{checked} vertices of the mesh split 3 times, {bad} exact "
                                                     f"mismatches, floats worst {worst:.3g}")


def check_edges(name, top):
    ch = top.chart
    loops = [tuple(loop) for loop in top._corners]
    bad = checked = 0
    for f in range(len(loops)):
        for s in range(len(loops[f])):
            g = top.across(f, s)
            if g is None or g < f:
                continue
            a, b = loops[f][s], loops[f][(s + 1) % len(loops[f])]
            ka, kb = loops[g].index(a), loops[g].index(b)
            p0, p1 = ch.schemes[f].corners[s], ch.schemes[f].corners[(s + 1) % len(loops[f])]
            q0, q1 = ch.schemes[g].corners[ka], ch.schemes[g].corners[kb]
            for i in range(9):
                w = Fraction(i, 8)
                pf = ch._exact_point(f, p0[0] + (p1[0] - p0[0]) * w, p0[1] + (p1[1] - p0[1]) * w)
                pg = ch._exact_point(g, q0[0] + (q1[0] - q0[0]) * w, q0[1] + (q1[1] - q0[1]) * w)
                checked += 1
                bad += pf != pg
    report(name, "edges", bad == 0, f"{checked} points on shared sides, {bad} mismatches")


def check_step(name, top):
    """step() against exact differences at dyadic points (exact points by a border are only there)"""
    ch = top.chart
    rng = random.Random(5)
    q = lambda x, bits: round(x * (1 << bits)) / (1 << bits)
    worst = {}
    for k in (2, 12):
        for _ in range(60):
            f = rng.randrange(len(ch.regular))
            tri = top.sides[f] == 3
            ur, vr = q(rng.uniform(0.05, 0.45), k + 2), q(rng.uniform(0.05, 0.45), k + 2)
            du = np.array([[q(rng.uniform(0, 1), 10) for _ in range(4)]])
            dv = np.array([[q(rng.uniform(0, 1), 10) for _ in range(4)]])
            if tri:
                du, dv = du * 0.5, dv * 0.5
            small = 2.0 ** -k
            got = ch.step(np.array([f]), np.array([ur]), np.array([vr]), du, dv, small)[0]
            h, U, V = Fraction(small), Fraction(ur), Fraction(vr)
            base = ch._exact_point(f, U, V)
            for j in range(du.shape[1]):
                p = ch._exact_point(f, U + h * Fraction(float(du[0, j])), V + h * Fraction(float(dv[0, j])))
                want = np.array([float((p[c] - base[c]) / h) for c in range(3)])
                worst[k] = max(worst.get(k, 0.0), float(np.abs(got[j] - want).max() / np.abs(want).max()))
    report(name, "step", max(worst.values()) < 1e-5,
           "worst relative error " + ", ".join(f"{v:.2g} at steps of 2^-{k}" for k, v in sorted(worst.items())))


def check_across(name, top):
    ch = top.chart
    rng = random.Random(3)
    eps = 1e-6
    worst_p = worst_j = 0.0
    checked = 0
    for _ in range(300):
        f = rng.randrange(len(ch.regular))
        scheme = ch.schemes[f]
        m = scheme.sides
        k = rng.randrange(m)
        if ch._beyond[f, k] < 0:
            continue                      # an open side: nothing beyond it
        a, b = np.array(scheme.corners[k], float), np.array(scheme.corners[(k + 1) % m], float)
        e = a + rng.uniform(0.15, 0.85) * (b - a)
        normal = ch._side_normal[f, k] / np.linalg.norm(ch._side_normal[f, k])
        inside, past = e - eps * normal, e + eps * normal
        P, ju, jv = ch._evaluate(np.array([f, f]), np.array([inside[0], past[0]]), np.array([inside[1], past[1]]),
                                 orders=((0, 0), (1, 0), (0, 1)))
        J_in, J_past = np.stack([ju[0], jv[0]], 1), np.stack([ju[1], jv[1]], 1)
        scale = np.abs(J_in).max()
        worst_p = max(worst_p, float(np.linalg.norm(P[1] - P[0]) / (2 * eps * scale)))
        worst_j = max(worst_j, float(np.abs(J_past - J_in).max() / scale))
        checked += 1
    # (By a border, between quads and triangles or along an open rim, float derivatives come from rough pieces some
    # COARSE_DEPTH levels down, good to a thousandth or so, not to a float's rounding)
    report(name, "across", worst_p < 3.0 and worst_j < (1e-2 if ch.border else 1e-3),
           f"{checked} sides: a step of 2e-6 across moves the point {worst_p:.2g} times its length at most, the "
           f"first derivatives jump {worst_j:.2g} at most")


def check_opensubdiv(name, obj, top):
    ch = top.chart
    modifier = obj.modifiers.new("subdivision", 'SUBSURF')
    modifier.levels, modifier.use_limit_surface, modifier.quality = 5, True, 6
    modifier.boundary_smooth = 'ALL'
    V = np.array([v.co for v in obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.vertices])
    obj.modifiers.remove(modifier)
    size = float(np.ptp(V, axis=0).max())
    cell = size / 200
    grid = {}
    for i, p in enumerate(V):
        grid.setdefault(tuple(np.floor(p / cell).astype(int)), []).append(i)

    def nearest(p):
        c = np.floor(p / cell).astype(int)
        return min((np.linalg.norm(V[i] - p) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                    for i in grid.get((c[0] + dx, c[1] + dy, c[2] + dz), [])), default=np.inf)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        f = int(rng.integers(len(top._corners)))
        I, J = int(rng.integers(33)), int(rng.integers(33))
        worst = max(worst, nearest(ch.points(np.array([f]), np.array([I / 32]), np.array([J / 32]))[0]))
    # (OpenSubdiv's points are float32: some 1e-7 of the size)
    report(name, "opensubdiv", worst < 1e-5 * size, f"worst {worst:.3g} off its level-5 limit (size {size:.3g})")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    names = argv or MESHES
    for name in names:
        start = time.perf_counter()
        obj = made(name)
        top = SU.Topology(obj.data, (0.1, 0.1, 0.1, 1.0), chart=CHARTS[name])
        print(f"---- {name}: {len(top._corners)} faces, {type(top.chart).__name__}, "
              f"{len(top.chart._boundary)} open edges, {int(top.chart.regular.sum())} regular")
        check_split(name, top)
        check_edges(name, top)
        check_step(name, top)
        check_across(name, top)
        if CHARTS[name] == "catmull_clark":
            check_opensubdiv(name, obj, top)
        print(f"     ({time.perf_counter() - start:.1f} s)")
    print(f"{len(failures)} failed" + (": " + ", ".join(f"{n} {c}" for n, c in failures) if failures else ""))
    sys.stdout.flush()
    os._exit(len(failures))


main()
