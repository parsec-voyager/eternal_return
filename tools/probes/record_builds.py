"""Flight harness probe (tools/flight_sim.py --probe): the builds a flight sends to its build processes, saved as they
went out and as they came back, to run again later (tools/checks/replay_builds.py) and to hold a build written anew
against (the same call must give the same answer).

A build sent to a process is already plain data: background.py pickles the function and its arguments, with the
objects every build shares (a surface's Topology) named by key and sent once. That's what's saved, in RECORD_OUT (or
./recorded_builds):

    shared_<key>.pickle          (key, module, class, shared_data()): one for each shared object a saved call names
    <function>_<n>.pickle        {"function", "call": the pickled (function, args), "keys", "answer": the pickled
                                 result, "seconds": how long the process took}

Of each function, one call in RECORD_EVERY (1) is saved, RECORD_MOST (40) at most: a level's answer is megabytes.
"""

import os
import pickle
import sys
import threading
import time
from collections import defaultdict

OUT = os.environ.get("RECORD_OUT") or os.path.join(os.getcwd(), "recorded_builds")
EVERY = int(os.environ.get("RECORD_EVERY", 1))
MOST = int(os.environ.get("RECORD_MOST", 40))
SEEN, SAVED = defaultdict(int), defaultdict(int)
SHARED = set()
LOCK = threading.Lock()


def setup(m):
    package = m.__name__.rpartition(".")[0]
    background = sys.modules[package + ".background"]
    os.makedirs(OUT, exist_ok=True)
    process = background._Process

    def call(self, function, args):
        """_Process.call(), keeping what it sends and what comes back"""
        name = f"{function.__module__.rpartition('.')[2]}.{function.__qualname__}"
        with LOCK:
            SEEN[name] += 1
            keep = SEEN[name] % EVERY == 0 and SAVED[name] < MOST
            if keep:
                SAVED[name] += 1
                number = SAVED[name]
        if not keep:
            return real(self, function, args)
        data, keys = background.dumps((function, args))
        for key in sorted(keys):
            with LOCK:
                new = key not in SHARED
                SHARED.add(key)
            if new:
                obj = background._shared[key]
                with open(os.path.join(OUT, f"shared_{key}.pickle"), "wb") as f:
                    pickle.dump((key, type(obj).__module__.rpartition(".")[2], type(obj).__qualname__,
                                 obj.shared_data()), f, pickle.HIGHEST_PROTOCOL)
        # The same steps as _Process.call(), with the answer kept as it came
        for key in sorted(keys - self.sent):
            obj = background._shared[key]
            self.connection.send(("define", key, type(obj).__module__, type(obj).__qualname__, obj.shared_data()))
            self.sent.add(key)
        start = time.perf_counter()
        self.connection.send(("call", data))
        kind, payload = self.connection.recv()
        seconds = time.perf_counter() - start
        if kind == "error":
            raise RuntimeError(f"in a build process:\n{payload}")
        with open(os.path.join(OUT, f"{name}_{number:03d}.pickle"), "wb") as f:
            pickle.dump({"function": name, "call": data, "keys": sorted(keys), "answer": payload,
                         "seconds": seconds}, f, pickle.HIGHEST_PROTOCOL)
        return background.loads(payload)

    real = process.call
    process.call = call


def report(out):
    for name in sorted(SEEN):
        print(f"RECORDED {name}: {SAVED[name]} of {SEEN[name]} calls")
    print(f"RECORDED in {OUT}: {len(SHARED)} shared objects")
