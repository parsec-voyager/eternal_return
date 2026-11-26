"""Flight harness probe (tools/flight_sim.py --probe): how much of the view is missing, the floor's levels and the 3D
frames not keeping up with the drone, against the map fully built where the camera is. The test that matters for any
change to an endless map's builds, scheduling or speed (2026-11-25, the user: "test whether frames and floors are
keeping up, not just fps"): fly it --realtime, diving and flying backward away from the floor fast
(tools/dive_back_plan.py), on every map the change touches, and look at the worst pictures.

Every MISSING_EVERY frames from MISSING_FROM s: the view rendered as it is (live); then every surface map settled here
and now (the worker threads off, build budgets unbounded, MISSING_ROUNDS rounds) and rendered again (settled). The
share of the view's pixels that changed (by more than CHANGED in any channel) is what the pilot was missing. The settle
builds again only the floor levels missing or no longer covering the drone, not those merely due for a refresh
(outlines round where the camera is now, finer cuts, trims), whose detail changed the picture by some 3% at every
sample without anything missing. MISSING lines (and which levels the settle added or built again); MISSING_OUT gets
the worst samples as live|settled pictures. The settle builds what was missing, so the flight goes on from a fully
built map after a sample (fewer misses after it than a real flight would have): compare versions on the same plan."""

import importlib
import math
import os

import numpy as np

EVERY = int(os.environ.get("MISSING_EVERY", "90"))
FROM = float(os.environ.get("MISSING_FROM", "1.0"))
ROUNDS = int(os.environ.get("MISSING_ROUNDS", "40"))
CHANGED = 0.12
ROWS = []


def setup(m):
    pass


def _render(scene, path):
    import bpy
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(path, check_existing=False)
    w, h = image.size
    pixels = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.reshape(h, w, 4)[::-1, :, :3]


def frame(t, number, m, scene):
    if number % EVERY or t < FROM:
        return
    import bpy
    package = m.__name__.rpartition(".")[0]
    surface = importlib.import_module(package + ".surface")
    parts = importlib.import_module(package + ".parts")
    ss = importlib.import_module(package + ".surface_substitution")
    bg = importlib.import_module(package + ".background")
    out = os.environ.get("MISSING_OUT") or os.path.dirname(bpy.data.filepath)
    maps = [o for o in scene.objects if surface.is_surface(o)]
    if not maps:
        return

    def levels():
        found = {}
        for obj in maps:
            st = ss._states.get(obj.name)
            for d, entry in (st.drawn.items() if st else ()):
                if entry is not None:
                    found[("floor", d)] = id(entry[1])
            for o in obj.children:
                if o.get(parts.OWNER_KEY) == obj.name and o.get(parts.LEVEL_KEY) is not None:
                    found[("frames", int(o[parts.LEVEL_KEY]))] = id(parts._built.get(o.name))
        return found

    def covering_only(world, text, record, foot, window, camera, outline_within, height, reach_wanted=None):
        """stale() for the settle: only a level no longer covering the drone's window"""
        _text, center, _camera, scale = ss.placed(world, text, record)
        return not record.covers(math.dist(center, foot), window, scale)

    live = _render(scene, os.path.join(out, "_live.png"))
    before = levels()
    saved = (bg.enabled, ss.BUILD_BUDGET, parts.BUILD_BUDGET, ss.stale)
    try:
        bg.enabled, ss.BUILD_BUDGET, parts.BUILD_BUDGET, ss.stale = False, 60.0, 60.0, covering_only
        for _ in range(ROUNDS):
            camera = m.drone_camera.matrix_world
            for obj in maps:
                surface.settle(obj, camera.translation)
                if parts.is_parts(obj):
                    parts.settle(obj, camera.translation, budget=60.0, view=camera.to_3x3())
            bpy.context.view_layer.update()
    finally:
        bg.enabled, ss.BUILD_BUDGET, parts.BUILD_BUDGET, ss.stale = saved
    settled = _render(scene, os.path.join(out, "_settled.png"))
    after = levels()
    added = sorted(k for k in after if k not in before)
    rebuilt = sorted(k for k in after if k in before and after[k] != before[k])
    share = float((np.abs(live - settled).max(axis=2) > CHANGED).mean())
    ROWS.append((t, number, share))
    print(f"MISSING t={t:.2f} f{number}: {100 * share:.1f}% of the view; the settle added {added}, built again "
          f"{rebuilt}", flush=True)
    if share >= max(r[2] for r in ROWS) and share > 0.01:
        h, w = live.shape[:2]
        both = np.zeros((h, 2 * w + 8, 4), np.float32)
        both[..., 3] = 1.0
        both[:, :w, :3] = live
        both[:, w + 8:, :3] = settled
        image = bpy.data.images.new("_missing_pair", 2 * w + 8, h)
        image.pixels.foreach_set(both[::-1].ravel())
        image.filepath_raw = os.path.join(out, f"missing_{number:05d}.png")
        image.file_format = 'PNG'
        image.save()
        bpy.data.images.remove(image)


def report(out):
    if not ROWS:
        print("MISSING no samples", flush=True)
        return
    shares = np.array([r[2] for r in ROWS])
    worst = ROWS[int(np.argmax(shares))]
    print(f"MISSING {len(ROWS)} samples: over 1% of the view in {100 * (shares > 0.01).mean():.0f}% of them, median "
          f"{100 * np.median(shares):.1f}%, mean {100 * shares.mean():.1f}%, worst {100 * worst[2]:.1f}% at "
          f"t={worst[0]:.2f} (f{worst[1]})", flush=True)
