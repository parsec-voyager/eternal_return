"""Flight harness probe (tools/flight_sim.py --probe): calls of chosen functions and methods of the add-on in a
flight, saved as they went in and as they came out, to run again later (tools/checks/replay_calls.py) and to hold a
version written anew against, like record_builds.py's builds but for anything: the level machinery's steps run on
Blender's worker threads and its main thread, not in a build process.

    RECORD_CALLS="surface_substitution.Level.keep_near surface_substitution.level_zero" RECORD_OUT=<folder> ...

For each call saved, <function>_<n>.pickle: {"function", "before": the pickled (self or None, args, kwargs) as they
were when it was called, "after": the pickled (self or None, result), "seconds"}, pickled like a build's (shared
objects by key: shared_<key>.pickle beside them, as record_builds.py writes them). A level's `source` and `origin`
(the kept levels it's a copy of) are left out, as they are of what a build is sent.

An object that is a shared object's own part (a Topology's chart: its trees and caches don't pickle, and it's made
again from its Topology anyway) is saved as that part of it, so a chart's methods can be recorded too:
RECORD_CALLS="subdivision.SubdivisionChart._evaluate".

Of each function RECORD_MOST (24) calls are kept, spread over the flight (thinned as it goes). A call whose state
won't pickle (something another thread is changing) is skipped, and counted.
"""

import copy
import os
import pickle
import sys
import threading
import time
from collections import defaultdict

OUT = os.environ.get("RECORD_OUT") or os.path.join(os.getcwd(), "recorded_calls")
NAMES = os.environ.get("RECORD_CALLS", "").split()
MOST = int(os.environ.get("RECORD_MOST", 24))
SEEN, SKIPPED = defaultdict(int), defaultdict(int)
STRIDE = defaultdict(lambda: 1)
KEPT = defaultdict(list)
SHARED = set()
LOCK = threading.Lock()
LEFT_OUT = ("source", "origin")


# A shared object's attributes whose objects are saved as that attribute of it
PARTS = ("chart",)
SHARED_OBJECTS = {}      # background._shared, once the add-on is there


class Part:
    """An attribute of a shared object, standing in for the object there: pickled as getattr(the shared one, name)"""

    def __init__(self, owner, name):
        self.owner, self.name = owner, name

    def __reduce__(self):
        return getattr, (self.owner, self.name)


def plain(obj):
    """obj as it's saved: a level without the kept levels it's a copy of, a shared object's part as that part of it"""
    for name in PARTS:
        for owner in list(SHARED_OBJECTS.values()):
            # (an owner that has no such part mustn't claim a None)
            if obj is not None and getattr(owner, name, None) is obj:
                return Part(owner, name)
    if type(obj).__name__.endswith("Chart"):
        # (A chart whose Topology no build has shared yet: the call is skipped)
        raise ValueError("a chart whose topology isn't shared yet")
    if any(name in getattr(obj, "__dict__", ()) for name in LEFT_OUT):
        obj = copy.copy(obj)
        for name in LEFT_OUT:
            obj.__dict__.pop(name, None)
    return obj


def setup(m):
    package = m.__name__.rpartition(".")[0]
    background = sys.modules[package + ".background"]
    os.makedirs(OUT, exist_ok=True)
    global SHARED_OBJECTS
    SHARED_OBJECTS = background._shared

    def packed(value):
        data, keys = background.dumps(value)
        for key in sorted(keys):
            with LOCK:
                new = key not in SHARED
                SHARED.add(key)
            if new:
                obj = background._shared[key]
                with open(os.path.join(OUT, f"shared_{key}.pickle"), "wb") as f:
                    pickle.dump((key, type(obj).__module__.rpartition(".")[2], type(obj).__qualname__,
                                 obj.shared_data()), f, pickle.HIGHEST_PROTOCOL)
        return data

    def watch(owner, attribute, name, method):
        real = getattr(owner, attribute)

        def call(*args, **kwargs):
            with LOCK:
                SEEN[name] += 1
                if len(KEPT[name]) >= MOST:
                    for path in KEPT[name][1::2]:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                    KEPT[name] = KEPT[name][0::2]
                    STRIDE[name] *= 2
                keep = SEEN[name] % STRIDE[name] == 0
                number = SEEN[name]
            if not keep:
                return real(*args, **kwargs)
            own = args[0] if method else None
            rest = args[1:] if method else args
            try:
                before = packed((plain(own), tuple(plain(a) for a in rest), kwargs))
            except Exception:
                SKIPPED[name] += 1
                return real(*args, **kwargs)
            start = time.perf_counter()
            result = real(*args, **kwargs)
            seconds = time.perf_counter() - start
            try:
                after = packed((plain(own), plain(result)))
            except Exception:
                SKIPPED[name] += 1
                return result
            path = os.path.join(OUT, f"{name}_{number:06d}.pickle")
            with open(path, "wb") as f:
                pickle.dump({"function": name, "before": before, "after": after, "seconds": seconds}, f,
                            pickle.HIGHEST_PROTOCOL)
            with LOCK:
                KEPT[name].append(path)
            return result
        setattr(owner, attribute, call)

    for name in NAMES:
        module_name, _, rest = name.partition(".")
        owner = sys.modules[f"{package}.{module_name}"]
        parts = rest.split(".")
        for part in parts[:-1]:
            owner = getattr(owner, part)
        watch(owner, parts[-1], name, method=len(parts) > 1)


def report(out):
    for name in NAMES:
        print(f"CALLS {name}: {len(KEPT[name])} of {SEEN[name]} calls kept"
              + (f", {SKIPPED[name]} skipped (their state wouldn't pickle)" if SKIPPED[name] else ""))
    print(f"CALLS in {OUT}: {len(SHARED)} shared objects")
