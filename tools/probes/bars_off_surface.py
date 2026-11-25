"""Flight harness probe (tools/flight_sim.py --probe): every 2 s, a sample of the motif levels' vertices (the map's
"<map>.substitution" objects, as Blender shows them, pushes included) and how far each is off the surface, over its
distance from the camera (BAR lines; the end: the worst). The layers' push puts a level a few thousandths of its
distance under the next finer one; much more is a bar placed off the surface."""

import random
import sys

import numpy as np

EVERY = 120                  # frames between samples
SAMPLE = 60                  # vertices a level object
ROWS = []


def frame(t, number, m, scene):
    if number % EVERY or t < 1:
        return
    import bpy
    surface = sys.modules[m.__name__.rpartition(".")[0] + ".surface"]
    obj = next((o for o in scene.objects if surface.is_surface(o)), None)
    if obj is None:
        return
    world = surface.world_of(obj)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    camera = m.drone_camera.matrix_world.translation
    unit = float(np.linalg.norm(surface.world_vector(obj, (1, 0, 0))))
    off = []
    random.seed(number)
    for level in (o for o in scene.objects if o.name.startswith(obj.name + ".substitution") and o.type == 'MESH'):
        evaluated = level.evaluated_get(depsgraph)
        vertices, matrix = evaluated.data.vertices, evaluated.matrix_world
        for k in random.sample(range(len(vertices)), min(SAMPLE, len(vertices))):
            point = matrix @ vertices[k].co
            below = surface.below_of(obj, world, surface.map_point(obj, point))
            off.append(abs(below.height) * world.rho * unit / max((point - camera).length, 1e-30))
    if off:
        ROWS.append((t, float(np.median(off)), float(np.percentile(off, 99)), float(max(off))))
        print(f"BAR t={t:.0f}: level vertices off the surface / distance: median {ROWS[-1][1]:.3g}, p99 "
              f"{ROWS[-1][2]:.3g}, max {ROWS[-1][3]:.3g}")


def report(out):
    if ROWS:
        print(f"BAR worst max {max(r[3] for r in ROWS):.3g}, worst p99 {max(r[2] for r in ROWS):.3g}")
        sys.stdout.flush()
