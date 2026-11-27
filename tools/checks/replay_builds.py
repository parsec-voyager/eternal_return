"""Recorded builds (tools/probes/record_builds.py) run again, each answer held against the one recorded: a build is
meant to be a function of what it's sent and nothing else, and one written anew (in Rust) is checked against the same
recordings. Also what each costs with nothing else running (the seconds a build process took in the flight, and here).

    <Blender's python> -I tools/checks/replay_builds.py [--core <folder>] <folder of recordings> [name's start ...]

--core: where an eternal_return_core wheel is unpacked, for the builds to use its Rust (without it, a bare Python has
no core and every build is the add-on's own Python: the answers must be the same either way).

Run outside Blender, as a build process is (build_process.py): the add-on's modules under their package's name,
without its __init__. Numbers must match exactly; arrays that don't are printed with how far apart they are.
PASS or FAIL at the end.
"""

import importlib
import os
import pickle
import statistics
import sys
import time
import types
from fractions import Fraction

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGE = "eternal_return"


def different(a, b, where, found, most=6, seen=None):
    """Where a and b differ, added to found (strings), down through tuples, lists, dicts, arrays and objects' own
    values. A pair of objects is gone into once (seen): they may hold each other."""
    if len(found) >= most:
        return
    seen = set() if seen is None else seen
    if not isinstance(a, (int, float, str, bytes, bool, Fraction, complex, type(None), np.generic)):
        if (id(a), id(b)) in seen:
            return
        seen.add((id(a), id(b)))
    if type(a) is not type(b):
        found.append(f"{where}: {type(a).__name__} against {type(b).__name__}")
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape or a.dtype != b.dtype:
            found.append(f"{where}: array {a.shape} {a.dtype} against {b.shape} {b.dtype}")
        elif a.dtype == object:
            for i, (x, y) in enumerate(zip(a.ravel(), b.ravel())):
                different(x, y, f"{where}[{i}]", found, most, seen)
        elif not np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"):
            off = np.abs(a.astype(float) - b.astype(float))
            off = off[np.isfinite(off)]
            found.append(f"{where}: {int((a != b).sum())} of {a.size} values differ, by up to "
                         f"{float(off.max()) if len(off) else float('nan'):.3g}")
    elif isinstance(a, float):
        if a != b and not (a != a and b != b):
            found.append(f"{where}: {a!r} against {b!r}")
    elif isinstance(a, (int, str, bytes, bool, Fraction, complex, type(None), np.generic)):
        if a != b:
            found.append(f"{where}: {a!r:.60} against {b!r:.60}")
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            found.append(f"{where}: {len(a)} items against {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            different(x, y, f"{where}[{i}]", found, most, seen)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            found.append(f"{where}: keys differ ({len(a)} against {len(b)})")
        for key in a.keys() & b.keys():
            different(a[key], b[key], f"{where}[{key!r:.30}]", found, most, seen)
    elif isinstance(a, (set, frozenset)):
        if a != b:
            found.append(f"{where}: sets differ")
    else:
        state = lambda obj: {**{name: getattr(obj, name) for cls in type(obj).__mro__
                                for name in getattr(cls, "__slots__", ()) if hasattr(obj, name)},
                             **getattr(obj, "__dict__", {})}
        mine, theirs = state(a), state(b)
        if not mine and not theirs:
            if a is not b and a != b and getattr(type(a), "shared_by_key", None) is None:
                found.append(f"{where}: {type(a).__name__} values differ")
        else:
            different(mine, theirs, f"{where}.", found, most, seen)


def main():
    arguments = sys.argv[1:]
    if arguments[:1] == ["--core"]:
        sys.path.insert(0, arguments[1])
        arguments = arguments[2:]
    folder = arguments[0]
    wanted = tuple(arguments[1:])
    # The add-on's modules by their names in the recordings, as build_process.main() sets them up
    package = types.ModuleType(PACKAGE)
    package.__path__ = [os.path.join(REPO, PACKAGE)]
    sys.modules[PACKAGE] = package
    background = importlib.import_module(PACKAGE + ".background")
    for name in sorted(os.listdir(folder)):
        if name.startswith("shared_"):
            with open(os.path.join(folder, name), "rb") as f:
                key, module, cls, data = pickle.load(f)
            cls = getattr(importlib.import_module(f"{PACKAGE}.{module}"), cls)
            background.share(cls.from_shared(data), key)
    results = {}
    failed = 0
    for name in sorted(os.listdir(folder)):
        if name.startswith("shared_") or not name.endswith(".pickle") or (wanted and not name.startswith(wanted)):
            continue
        with open(os.path.join(folder, name), "rb") as f:
            record = pickle.load(f)
        function, args = background.loads(record["call"])
        start = time.perf_counter()
        answer = function(*args)
        seconds = time.perf_counter() - start
        # Through the same pickling the recorded answer went through
        answer = background.loads(background.dumps(answer)[0])
        found = []
        different(background.loads(record["answer"]), answer, "answer", found)
        row = results.setdefault(record["function"], {"calls": 0, "same": 0, "then": [], "now": []})
        row["calls"] += 1
        row["same"] += not found
        row["then"].append(record["seconds"])
        row["now"].append(seconds)
        if found:
            failed += 1
            print(f"DIFFERENT {name}:")
            for line in found:
                print(f"    {line}")
    for function, row in sorted(results.items()):
        print(f"{function}: {row['same']} of {row['calls']} the same | in the flight median "
              f"{1000 * statistics.median(row['then']):.1f} ms, total {sum(row['then']):.2f} s | here median "
              f"{1000 * statistics.median(row['now']):.1f} ms, total {sum(row['now']):.2f} s")
    core = sys.modules.get("eternal_return_core")
    print("FAIL" if failed or not results else "PASS", f"({sum(r['calls'] for r in results.values())} builds;",
          f"core: {os.path.dirname(core.__file__)})" if core else "no core: all Python)")
    sys.exit(1 if failed or not results else 0)


if __name__ == "__main__":
    main()
