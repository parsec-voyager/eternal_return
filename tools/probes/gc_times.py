"""Flight harness probe (tools/flight_sim.py --probe): Python's garbage collections, timed, by phase (the start, before
the first frame; the flight), thread and generation (GC lines). A collection holds Python's lock through all of it,
whichever thread runs it: a full one over the engines' caches stopped the frame for over a second until the flight
froze what's long-lived (background.hold_heap()). At the end, everything is thawed and collected once: what that
finds unreachable is what the freezing kept from being collected. GC_THAW=1 turns hold_heap off for the flight, to
compare."""

import gc
import os
import sys
import threading
import time
from collections import defaultdict

STATE = {"t": 0.0, "started": {}, "background": None}
SEEN = defaultdict(lambda: [0, 0.0, 0.0])          # (phase, thread, generation) -> [count, seconds, longest]


def timed(phase, info):
    key = threading.get_ident()
    if phase == "start":
        STATE["started"][key] = time.perf_counter()
        return
    start = STATE["started"].pop(key, None)
    if start is None:
        return
    spent = time.perf_counter() - start
    thread = "main" if threading.current_thread() is threading.main_thread() else "worker"
    entry = SEEN[("start" if STATE["t"] == 0.0 else "flight", thread, info["generation"])]
    entry[0] += 1
    entry[1] += spent
    entry[2] = max(entry[2], spent)


def setup(m):
    import importlib
    STATE["background"] = importlib.import_module(m.__name__.rpartition(".")[0] + ".background")
    if os.environ.get("GC_THAW"):
        STATE["background"].hold_heap(False)
    gc.callbacks.append(timed)


def frame(t, number, m, scene):
    STATE["t"] = t


def report(out):
    import resource
    for key in sorted(SEEN):
        count, spent, longest = SEEN[key]
        print(f"GC {key}: {count} collections, {spent * 1000:.0f} ms in all, longest {longest * 1000:.0f} ms")
    print(f"GC tracked objects {len(gc.get_objects())}, frozen {gc.get_freeze_count()}, most resident "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss >> 20} MB")
    STATE["t"] = -1.0
    STATE["background"].hold_heap(False)
    start = time.perf_counter()
    found = gc.collect()
    print(f"GC after the flight: thawed, a full collection found {found} unreachable objects in "
          f"{(time.perf_counter() - start) * 1000:.0f} ms")
    sys.stdout.flush()
