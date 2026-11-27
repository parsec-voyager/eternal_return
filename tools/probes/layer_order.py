"""Flight harness probe (tools/flight_sim.py --probe): does a coarser layer of a surface map show over a finer one?
Every ORDER_EVERY frames, a grid of rays from the drone camera (ORDER_GRID across) is cast into each of the map's motif
levels and its floor, as Blender shows them, pushes included, ordered by their push (the finest on top, the floor under
all). A ray whose nearest hit is a layer coarser than one it also hits just behind, on the same sheet of the surface
(their gap across the surface under SHEET of the distance), shows that coarser layer over the finer one: a layer order
fault, what the pilot sees as a coarse bar standing up through the finer levels (the knot's slabs).

ORDER lines: the faults' share of the rays that hit anything, by the layer shown over; WHO lines name a layer behind
more than 1% of them (its level, the world it was built for and now, coarse, its rebuild's state); the end: the worst.
Judge layering changes by these, not by the pieces' sag alone: most of a sag's faults at a horizon are slivers. Slow
(BVH ray casts in Python, some 25 s a sample at 100 across): fly --timing, and --realtime for what a pilot sees."""

import importlib
import os
import time

import numpy as np

EVERY = int(os.environ.get("ORDER_EVERY", "180"))
GRID = int(os.environ.get("ORDER_GRID", "100"))
SHEET = 0.01
FROM = 1.0
WHO_SHARE = 0.01
ROWS = []


def setup(m):
    pass


def frame(t, number, m, scene):
    if number % EVERY or t < FROM:
        return
    import bpy
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    package = m.__name__.rpartition(".")[0]
    surface = importlib.import_module(package + ".surface")
    ss = importlib.import_module(package + ".surface_substitution")
    obj = next((o for o in scene.objects if surface.is_surface(o)), None)
    if obj is None:
        return
    layers = [o for o in obj.children if o.type == 'MESH' and not o.hide_get() and len(o.data.polygons)
              and (o.get(ss.LEVEL_KEY) == obj.name or o.get(surface.FLOOR_KEY) == obj.name)]
    if len(layers) < 2:
        return
    layers.sort(key=lambda o: float(o.get("sub_push", 0.0)))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera = m.drone_camera
    origin = camera.matrix_world.translation.copy()
    turn = camera.matrix_world.to_3x3()
    tr, br, bl, tl = camera.data.view_frame(scene=scene)
    reach = camera.data.clip_end * max(camera.matrix_world.to_scale())
    trees = []
    for k, o in enumerate(layers):
        evaluated = o.evaluated_get(depsgraph)
        matrix = evaluated.matrix_world.copy()
        # A level parked out of sight (moved far down) or wholly past the clip end can't be hit
        box = [matrix @ Vector(c) for c in evaluated.bound_box]
        middle = sum(box, Vector()) / 8.0
        if (middle - origin).length - max((c - middle).length for c in box) > reach:
            continue
        # (A level shrunk to nothing in a fast climb has no inverse: nothing of it to hit)
        inverse = matrix.inverted(None)
        if inverse is None:
            continue
        trees.append((k, BVHTree.FromObject(o, depsgraph), matrix, inverse))
    columns, rows = GRID, max(2, GRID * scene.render.resolution_y // scene.render.resolution_x)
    hit_rays = 0
    faults = np.zeros(len(layers), dtype=np.int64)
    for j in range(rows):
        b = (j + 0.5) / rows
        left, right = bl.lerp(tl, b), br.lerp(tr, b)
        for i in range(columns):
            direction = (turn @ left.lerp(right, (i + 0.5) / columns)).normalized()
            hits = []
            for k, tree, matrix, inverse in trees:
                location, normal, _index, _distance = tree.ray_cast(inverse @ origin,
                                                                   (inverse.to_3x3() @ direction).normalized())
                if location is None:
                    continue
                d = (matrix @ location - origin).length
                if d <= reach:
                    hits.append((d, k, (inverse.to_3x3().transposed() @ normal).normalized()))
            if not hits:
                continue
            hit_rays += 1
            d_shown, k_shown, n_shown = min(hits, key=lambda h: h[0])
            across = abs(direction.dot(n_shown))
            if any(k < k_shown and d > d_shown and (d - d_shown) * across < SHEET * d_shown for d, k, _n in hits):
                faults[k_shown] += 1
    share = faults.sum() / max(hit_rays, 1)
    if faults.max() > WHO_SHARE * max(hit_rays, 1):
        o = layers[int(np.argmax(faults))]
        st = ss._states.get(obj.name)
        found = next(((d, e[1]) for d, e in (st.drawn.items() if st else ()) if e is not None and e[0] is o), None)
        if found is None:
            print(f"ORDER WHO f{number} {o.name}: not a drawn level (the floor?)", flush=True)
        else:
            d, record = found
            job = st.jobs.get(d)
            print(f"ORDER WHO f{number} {o.name}: level {d}, built at n {record.n} (now {surface.world_of(obj).n}), "
                  f"coarse {record.coarse}; its rebuild "
                  f"{'not asked' if job is None else f'asked {time.perf_counter() - job[0]:.2f} s ago'}", flush=True)
    ROWS.append((t, number, share))
    print(f"ORDER t={t:.2f} f{number}: coarser over finer on {100 * share:.3f}% of the rays that hit ({hit_rays}); by "
          f"the layer shown (finest first, the floor last): {faults.tolist()}", flush=True)


def report(out):
    if not ROWS:
        print("ORDER no samples")
        return
    shares = np.array([r[2] for r in ROWS])
    worst = ROWS[int(np.argmax(shares))]
    print(f"ORDER {len(ROWS)} samples: faults in {100 * (shares > 0).mean():.0f}% of them, median "
          f"{100 * np.median(shares):.3f}%, worst {100 * worst[2]:.3f}% at t={worst[0]:.2f} (f{worst[1]})", flush=True)
