"""Recorded calls (tools/probes/record_calls.py) run again, each held against what was recorded: the object it was
called on as it was left, and what it returned. Like replay_builds.py (the same comparison, tolerance and all), for
the functions and methods a flight runs outside its build processes.

    <Blender's python> -I tools/checks/replay_calls.py [--core <folder>] [--tolerance <t>] [--ignore <attribute>]
                                                       <folder> [name's start ...]

--ignore: an attribute left out of the comparison, given once for each (what an object keeps only to save working
something out again: a version written anew may not keep it).

PASS or FAIL at the end.
"""

import importlib
import os
import pickle
import statistics
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import replay_builds as R  # noqa: E402


def main():
    arguments = sys.argv[1:]
    while arguments[:1] and arguments[0] in ("--core", "--tolerance", "--ignore"):
        if arguments[0] == "--core":
            sys.path.insert(1, arguments[1])
        elif arguments[0] == "--ignore":
            R.IGNORED.add(arguments[1])
        else:
            R.TOLERANCE = float(arguments[1])
        arguments = arguments[2:]
    folder, wanted = arguments[0], tuple(arguments[1:])
    package = types.ModuleType(R.PACKAGE)
    package.__path__ = [os.path.join(R.REPO, R.PACKAGE)]
    sys.modules[R.PACKAGE] = package
    background = importlib.import_module(R.PACKAGE + ".background")
    for name in sorted(os.listdir(folder)):
        if name.startswith("shared_"):
            with open(os.path.join(folder, name), "rb") as f:
                key, module, cls, data = pickle.load(f)
            cls = getattr(importlib.import_module(f"{R.PACKAGE}.{module}"), cls)
            background.share(cls.from_shared(data), key)
    results, failed = {}, 0
    for name in sorted(os.listdir(folder)):
        if name.startswith("shared_") or not name.endswith(".pickle") or (wanted and not name.startswith(wanted)):
            continue
        with open(os.path.join(folder, name), "rb") as f:
            record = pickle.load(f)
        own, args, kwargs = background.loads(record["before"])
        module_name, _, rest = record["function"].partition(".")
        target = importlib.import_module(f"{R.PACKAGE}.{module_name}")
        parts = rest.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        function = getattr(target, parts[-1])
        start = time.perf_counter()
        result = function(own, *args, **kwargs) if len(parts) > 1 else function(*args, **kwargs)
        seconds = time.perf_counter() - start
        # Through the same pickling the recorded ones went through
        again = background.loads(background.dumps((own, result))[0])
        found = []
        R.different(background.loads(record["after"]), again, "(self, result)", found)
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
              f"{1000 * statistics.median(row['then']):.2f} ms | here median {1000 * statistics.median(row['now']):.2f} ms, "
              f"total {sum(row['now']):.2f} s")
    core = sys.modules.get("eternal_return_core")
    print("FAIL" if failed or not results else "PASS", f"({sum(r['calls'] for r in results.values())} calls;",
          "with the core)" if core else "no core: all Python)",
          f"| floats within {R.TOLERANCE:g}: the furthest apart {R.LET_THROUGH[0]:.1e}" if R.TOLERANCE else "")
    sys.exit(1 if failed or not results else 0)


if __name__ == "__main__":
    main()
