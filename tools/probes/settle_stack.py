"""flight_sim probe: where the main thread is while a floor settle (surface_substitution.settle()) takes long: once
one has run LONG seconds, the main thread's stack every EVERY seconds (STACK lines), from a watching thread."""
import functools
import importlib
import sys
import threading
import time
import traceback

LONG, EVERY = 1.5, 2.0
MAIN = threading.main_thread()
STATE = {"since": None, "dumped": 0.0}


def watch():
    while True:
        time.sleep(0.5)
        since, now = STATE["since"], time.perf_counter()
        if since is not None and now - since > LONG and now - STATE["dumped"] > EVERY:
            STATE["dumped"] = now
            frame = sys._current_frames().get(MAIN.ident)
            stack = "".join(traceback.format_stack(frame)[-14:]) if frame else "?"
            print(f"STACK {now - since:.1f} s into settle:\n{stack}", flush=True)


def setup(m):
    ss = importlib.import_module(m.__name__.rpartition(".")[0] + ".surface_substitution")
    original = ss.settle

    # (functools.wraps: the harness reads settle's parameters, see hitches.py)
    @functools.wraps(original)
    def settle(*args, **kwargs):
        STATE["since"] = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            STATE["since"] = None

    ss.settle = settle
    threading.Thread(target=watch, daemon=True).start()
