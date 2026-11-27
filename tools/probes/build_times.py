"""Flight harness probe (tools/flight_sim.py --probe): the builds a flight's worker threads run, counted and timed by
function: how many, how long each took (median and the slowest tenth, ms, the thread's wait for its build process
included), and the seconds they took in all. Printed at the end (BUILDS lines), from FROM seconds on.

What a change to the builds' speed does to a realtime flight: more builds done in the same flight, or the same builds
sooner. (tools/probes/where_time_goes.py says where the time goes inside them, but it samples: it can't count, and a
function that lets go of Python's lock gets more than its share of the samples.)"""

import statistics
import sys
import threading
import time
from collections import defaultdict

FROM = 2.0
STATE = {"counting": False}
TIMES = defaultdict(list)
LOCK = threading.Lock()


def setup(m):
    package = m.__name__.rpartition(".")[0]
    background = sys.modules[package + ".background"]
    real = background.call

    def call(function, *args):
        start = time.perf_counter()
        try:
            return real(function, *args)
        finally:
            if STATE["counting"]:
                with LOCK:
                    TIMES[f"{function.__module__.rpartition('.')[2]}.{function.__qualname__}"].append(
                        time.perf_counter() - start)
    background.call = call


def frame(t, number, m, scene):
    if t >= FROM:
        STATE["counting"] = True


def report(out):
    total = 0.0
    for name, times in sorted(TIMES.items(), key=lambda item: -sum(item[1])):
        times = sorted(times)
        total += sum(times)
        print(f"BUILDS {name}: {len(times)} builds, median {1000 * statistics.median(times):.1f} ms, slowest tenth from "
              f"{1000 * times[int(0.9 * (len(times) - 1))]:.1f} ms, {sum(times):.2f} s in all")
    print(f"BUILDS in all: {sum(len(t) for t in TIMES.values())} builds, {total:.2f} s")
