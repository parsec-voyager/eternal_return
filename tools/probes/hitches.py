"""Flight harness probe (tools/flight_sim.py --probe): every frame whose wall-clock gap from the last overran HITCH_MS,
or that a loop moved the map's depth (n), with the main thread's time in the surface engines' functions within it
(HITCH lines). A gap with little of it in them is spent elsewhere: Python's lock held by a worker, a garbage
collection (tools/probes/gc_times.py), Blender's own evaluation. Fly with --realtime --timing."""

import sys
import threading
import time
from collections import defaultdict

HITCH_MS = 50.0
SPENT = defaultdict(float)
STATE = {"last": None, "rows": []}
MAIN = threading.main_thread()
WATCHED = {
    "surface": ("settle", "build_floor", "below_of", "recenter_move", "loop", "save", "settle_floor"),
    "parts": ("settle", "build_level", "settle_exact", "write_exact", "land_exact"),
    "surface_substitution": ("settle", "giant_tint", "cut_levels", "kept_levels"),
    "tiling": ("write_mesh", "split_near"),
}


def wrap(owner, name, label):
    function = getattr(owner, name)

    def timed(*args, **kwargs):
        if threading.current_thread() is not MAIN:
            return function(*args, **kwargs)
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            SPENT[label] += time.perf_counter() - start
    setattr(owner, name, timed)


def setup(m):
    package = m.__name__.rpartition(".")[0]
    for module, names in WATCHED.items():
        owner = sys.modules.get(f"{package}.{module}")
        for name in names if owner is not None else ():
            if hasattr(owner, name):
                wrap(owner, name, f"{module}.{name}")


def frame(t, number, m, scene):
    now = time.perf_counter()
    surface = sys.modules[m.__name__.rpartition(".")[0] + ".surface"]
    obj = next((o for o in scene.objects if surface.is_surface(o)), None)
    n = surface.world_of(obj).n if obj is not None else None
    if STATE["last"] is not None:
        STATE["rows"].append((t, (now - STATE["last"]) * 1000, dict(SPENT), n))
    SPENT.clear()
    STATE["last"] = now


def report(out):
    import numpy as np
    rows = STATE["rows"]
    if not rows:
        return
    gaps = np.array([r[1] for r in rows])
    print(f"HITCH frames {len(rows)}, gap median {np.median(gaps):.1f} ms, p99 {np.percentile(gaps, 99):.1f}, "
          f"max {gaps.max():.0f}; over {HITCH_MS:g} ms: {(gaps > HITCH_MS).sum()}")
    last_n = None
    for t, gap, spent, n in rows:
        looped = last_n is not None and n != last_n
        last_n = n
        if gap > HITCH_MS or looped:
            top = ", ".join(f"{k} {v * 1000:.0f}" for k, v in sorted(spent.items(), key=lambda kv: -kv[1]) if v > 0.003)
            print(f"HITCH t={t:6.2f} n={n} gap {gap:6.0f} ms{' LOOP' if looped else ''}: {top}")
    sys.stdout.flush()
