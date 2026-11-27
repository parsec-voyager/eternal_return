"""Flight harness probe (tools/flight_sim.py --probe): the main thread's time in the surface engines' main functions,
in ms per frame from 2 s on (the start's left out), with their calls, printed at the end (TICK lines). Only the main
thread is timed: the worker threads run the same functions for their builds. Fly with --realtime --timing for a real
flight's numbers."""

import functools
import sys
import threading
import time
from collections import defaultdict

FROM = 2.0                   # seconds into the flight the counting starts
SPENT, CALLS = defaultdict(float), defaultdict(int)
STATE = {"frames": 0, "counting": False}
MAIN = threading.main_thread()
WATCHED = {
    "surface": ("settle", "settle_floor", "below_of", "recenter_move", "crossing", "surface_vector", "center_vector"),
    "surface_substitution": ("settle", "giant_tint", "kept_levels", "built_level", "cut_levels", "keep_near"),
    "parts": ("settle", "settle_exact", "build_level", "write_level"),
    "tiling": ("write_mesh",),
}


def wrap(owner, name, label):
    function = getattr(owner, name)
    static = isinstance(owner, type) and isinstance(owner.__dict__.get(name), staticmethod)

    # (functools.wraps: the harness reads its parameters, inspect.signature(surface.settle): see hitches.py)
    @functools.wraps(function)
    def timed(*args, **kwargs):
        if threading.current_thread() is not MAIN:
            return function(*args, **kwargs)
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            SPENT[label] += time.perf_counter() - start
            CALLS[label] += 1
    setattr(owner, name, staticmethod(timed) if static else timed)


def setup(m):
    package = m.__name__.rpartition(".")[0]
    for module, names in WATCHED.items():
        owner = sys.modules.get(f"{package}.{module}")
        for name in names if owner is not None else ():
            if hasattr(owner, name):
                wrap(owner, name, f"{module}.{name}")
    surface = sys.modules[package + ".surface"]
    wrap(surface.Below, "global_face", "Below.global_face")


def frame(t, number, m, scene):
    if t < FROM:
        return
    if not STATE["counting"]:
        SPENT.clear()
        CALLS.clear()
        STATE["counting"] = True
    STATE["frames"] += 1


def report(out):
    frames = max(STATE["frames"], 1)
    for label in sorted(SPENT, key=lambda k: -SPENT[k]):
        print(f"TICK {label:32s} {SPENT[label] / frames * 1000:7.2f} ms/frame  ({CALLS[label]} calls)")
    sys.stdout.flush()
