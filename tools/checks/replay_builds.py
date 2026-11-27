"""Recorded builds (tools/probes/record_builds.py) run again, each answer held against the one recorded: a build is
meant to be a function of what it's sent and nothing else, and one written anew (in Rust) is checked against the same
recordings. Also what each costs with nothing else running (the seconds a build process took in the flight, and here).

    <Blender's python> -I tools/checks/replay_builds.py [--core <folder>] [--tolerance <t>] [--ignore <attribute>]
                                                        <folder of recordings> [name's start ...]

--core: where an eternal_return_core wheel is unpacked, for the builds to use its Rust (without it, a bare Python has
no core and every build is the add-on's own Python).
--tolerance: floats may differ by this share of the biggest number of their array (1e-9, say). Without it they must
match exactly, which the Rust does where it does the same arithmetic in the same order (tiling's cuts); where numpy
runs matrix products (a smooth chart's patches), its sums round in another order and the last bits differ. What
isn't a float (how many pieces, which rows, whole numbers) must match either way. The biggest difference let through
is printed. Two more things pass under a tolerance, and are counted: an array of rows that holds the same rows in
another order, and numbers that are rounding noise by nature (NOISE).
--ignore: an attribute left out of the comparison, given once for each: what an object keeps only to save working
something out again (a level's `_face_sides`), which a version written anew may not keep.

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
TOLERANCE = 0.0
LET_THROUGH = [0.0]          # the biggest difference a tolerance has let through
REORDERED = [0]              # how many arrays held the same rows in another order
# Numbers that are rounding noise where they aren't 0, held to a difference of their own (under a tolerance): a
# lattice's slopes are sqrt(1 - c^2) of two normals' c, 1e-8 for two that are the same to the last bit but one
NOISE = {"_slopes": 1e-6}
# Attributes left out of the comparison: what an object keeps only to save working it out again (--ignore)
IGNORED = set()


def different(a, b, where, found, most=6, seen=None):
    """Where a and b differ, added to found (strings), down through tuples, lists, dicts, arrays and objects' own
    values. A pair of objects is gone into once (seen): they may hold each other."""
    if len(found) >= most or a is b:
        # (The very same object on both sides, a rule's tables or a module, is the same: gone into, a module's own
        # NaT constants differ from themselves)
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
            with np.errstate(invalid="ignore", over="ignore"):
                off = np.abs(a.astype(float) - b.astype(float))
                finite = np.isfinite(a.astype(float)) & np.isfinite(b.astype(float))
                size = max(float(np.abs(a[finite]).max(initial=0.0)), float(np.abs(b[finite]).max(initial=0.0)))
            share = float(off[finite].max(initial=0.0)) / size if size else float("inf")
            # Within the tolerance: floats, all finite in the same places, none further apart than its share
            noise = next((n for name, n in NOISE.items() if where.endswith(f"[{name!r}]")), None)
            same_places = (a.dtype.kind == "f" and np.array_equal(np.isnan(a), np.isnan(b))
                           and np.array_equal(a[~finite], b[~finite], equal_nan=True))
            if TOLERANCE and same_places and share <= TOLERANCE:
                LET_THROUGH[0] = max(LET_THROUGH[0], share)
            elif TOLERANCE and same_places and noise is not None and float(off[finite].max(initial=0.0)) <= noise:
                pass
            elif TOLERANCE and same_places and a.ndim == 2 and reordered(a, b, size):
                # (Pieces sorted by a number that two of them all but share come out either way round)
                REORDERED[0] += 1
            else:
                found.append(f"{where}: {int((a != b).sum())} of {a.size} values differ, by up to "
                             f"{float(off[finite].max(initial=0.0)):.3g} ({share:.1e} of the biggest)")
    elif isinstance(a, float):
        if a != b and not (a != a and b != b):
            share = abs(a - b) / max(abs(a), abs(b)) if a - b == a - b and max(abs(a), abs(b)) > 0 else float("inf")
            if TOLERANCE and share <= TOLERANCE:
                LET_THROUGH[0] = max(LET_THROUGH[0], share)
            else:
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
        state = lambda obj: {name: value for name, value in {
            **{name: getattr(obj, name) for cls in type(obj).__mro__ for name in getattr(cls, "__slots__", ())
               if hasattr(obj, name)}, **getattr(obj, "__dict__", {})}.items() if name not in IGNORED}
        mine, theirs = state(a), state(b)
        if not mine and not theirs:
            if a is not b and a != b and getattr(type(a), "shared_by_key", None) is None:
                found.append(f"{where}: {type(a).__name__} values differ")
        else:
            different(mine, theirs, f"{where}.", found, most, seen)


def reordered(a, b, size):
    """Whether two arrays of rows hold the same rows, to within the tolerance, in another order"""
    order = lambda m: m[np.lexsort(np.round(m / (size * max(TOLERANCE, 1e-12) * 1000.0)).T[::-1])]
    with np.errstate(invalid="ignore"):
        return bool(np.all(np.abs(order(a.astype(float)) - order(b.astype(float))) <= TOLERANCE * size * 2000.0))


def main():
    global TOLERANCE
    arguments = sys.argv[1:]
    while arguments[:1] and arguments[0] in ("--core", "--tolerance", "--ignore"):
        if arguments[0] == "--core":
            sys.path.insert(0, arguments[1])
        elif arguments[0] == "--ignore":
            IGNORED.add(arguments[1])
        else:
            TOLERANCE = float(arguments[1])
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
          f"core: {os.path.dirname(core.__file__)})" if core else "no core: all Python)",
          f"| floats within {TOLERANCE:g}: the furthest apart {LET_THROUGH[0]:.1e}"
          + (f"; {REORDERED[0]} arrays the same rows in another order" if REORDERED[0] else "") if TOLERANCE else "")
    sys.exit(1 if failed or not results else 0)


if __name__ == "__main__":
    main()
