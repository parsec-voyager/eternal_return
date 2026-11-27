"""Flight harness probe (tools/flight_sim.py --probe): where the add-on's time goes, on every thread and in every
build process, by sampling: every few milliseconds, what each thread is in. Timing wrappers (tick_times.py) need the
functions named beforehand and only see the main thread; cProfile slows the many small calls far more than the few
big ones. A sample costs the same whatever runs.

For each sample of a thread, three things are kept, and the time since the sampler's last look is added to them:
- the job: the add-on function the thread's work began in, past background.py's and build_process.py's own (a
  build process: the build it was asked for; a worker thread: what it took off the queue; the main thread: the
  tick's entry),
- the leaf: the innermost add-on function running,
- outside: when the innermost function of all isn't the add-on's (a wait, a pickle, a numpy wrapper), which.
Only Python functions have frames: time inside numpy's C code counts for the add-on function that called it.

Printed at the end (WHERE lines), from FROM seconds on, and written whole to where.json in the flight's folder. Fly
with --realtime --timing for a real flight's numbers.

The same file is what a build process starts from while the probe is on (as __main__): it starts the sampler there
and then runs build_process.main().
"""

import atexit
import importlib.util
import json
import os
import sys
import threading
import time
from collections import defaultdict

FROM = 2.0                   # seconds into the flight the report starts
EVERY = 0.002                # seconds between looks
SAVE_EVERY = 1.0             # a build process writes what it has this often (it may be ended without warning)
# Innermost functions that are a wait, not work
WAITS = {"wait", "recv", "_recv", "recv_bytes", "_recv_bytes", "poll", "_poll", "select", "sleep", "accept", "get",
         "acquire", "wait_for"}
STATE = {}


class Sampler(threading.Thread):
    """Looks at every other thread each EVERY seconds. seconds[role][second][job|leaf|outside] += the time since
    the last look (not EVERY: a sampler kept waiting for Python's lock would count short)."""

    def __init__(self, folder, role):
        super().__init__(daemon=True, name="where_time_goes")
        self.folder = os.path.join(os.path.realpath(folder), "")
        self.role = role                      # of a thread's id -> its role's name (None: not looked at)
        self.seconds = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.names = {}                       # a code object -> "module.function", once
        self.save = None

    def name_of(self, code):
        name = self.names.get(code)
        if name is None:
            base = os.path.splitext(os.path.basename(code.co_filename))[0]
            inside = os.path.realpath(code.co_filename).startswith(self.folder)
            name = self.names[code] = (inside, f"{base}.{code.co_qualname}", code.co_name)
        return name

    def run(self):
        own = threading.get_ident()
        last = time.perf_counter()
        saved = last
        while True:
            time.sleep(EVERY)
            now = time.perf_counter()
            step, last = min(now - last, 10 * EVERY), now
            second = int(time.time())
            for ident, frame in sys._current_frames().items():
                role = None if ident == own else self.role(ident)
                if role is None:
                    continue
                inside, top, plain = self.name_of(frame.f_code)
                outside = "" if inside else ("(waiting)" if plain in WAITS else top)
                leaf = job = own_job = None
                while frame is not None:
                    inside, name, _plain = self.name_of(frame.f_code)
                    if inside:
                        leaf = leaf or name
                        # The job: the outermost function that isn't the threads' and processes' own machinery
                        if name.startswith(("background.", "build_process.")):
                            own_job = name
                        else:
                            job = name
                    frame = frame.f_back
                self.seconds[role][second][f"{job or own_job}|{leaf}|{outside}"] += step
            if self.save and now - saved > SAVE_EVERY:
                saved = now
                self.save()

    def whole(self):
        return {role: {str(second): dict(keys) for second, keys in list(by_second.items())}
                for role, by_second in list(self.seconds.items())}


def boot():
    """A build process's start: the sampler, then build_process.main() as if it had been started itself."""
    folder = sys.argv[3]
    out = os.environ["WHERE_OUT"]
    main = threading.main_thread().ident
    sampler = Sampler(folder, lambda ident: "build processes" if ident == main else None)
    path = os.path.join(out, f"where_process_{os.getpid()}.json")

    def save():
        with open(path + ".part", "w") as f:
            json.dump(sampler.whole(), f)
        os.replace(path + ".part", path)
    sampler.save = save
    atexit.register(save)
    sampler.start()
    spec = importlib.util.spec_from_file_location("build_process", os.path.join(folder, "build_process.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


def setup(m):
    package = m.__name__.rpartition(".")[0]
    background = sys.modules[package + ".background"]
    folder = os.path.dirname(os.path.abspath(background.__file__))
    main = threading.main_thread().ident
    sampler = STATE["sampler"] = Sampler(folder, lambda ident: "main thread" if ident == main else "worker threads")
    sampler.start()

    # Build processes start from this file instead of build_process.py, told where to write
    real = background.subprocess.Popen

    def started(command, *args, **kwargs):
        if len(command) > 2 and os.path.basename(command[2]) == "build_process.py":
            command = command[:2] + [os.path.abspath(__file__)] + command[3:]
            kwargs["env"] = dict(os.environ, WHERE_OUT=STATE["out"])
        return real(command, *args, **kwargs)

    class Subprocess:
        """background's subprocess module, with its Popen swapped"""
        def __getattr__(self, name):
            return started if name == "Popen" else getattr(sys.modules["subprocess"], name)
    background.subprocess = Subprocess()
    STATE["out"] = os.environ.get("WHERE_OUT") or os.path.join(os.getcwd(), "where_time_goes")
    os.makedirs(STATE["out"], exist_ok=True)


def frame(t, number, m, scene):
    if t >= FROM and "from" not in STATE:
        STATE["from"] = int(time.time()) + 1
    STATE["until"] = int(time.time())


def table(title, seconds, total, most=12):
    print(f"WHERE {title}")
    for name, spent in sorted(seconds.items(), key=lambda item: -item[1])[:most]:
        if spent < 0.005 * total:
            break
        print(f"WHERE   {spent:7.2f} s {100 * spent / total:5.1f}%  {name}")


def report(out):
    whole = STATE["sampler"].whole()
    folder = STATE["out"]
    for name in sorted(os.listdir(folder)):
        if name.startswith("where_process_") and name.endswith(".json"):
            with open(os.path.join(folder, name)) as f:
                for role, by_second in json.load(f).items():
                    for second, keys in by_second.items():
                        mine = whole.setdefault(role, {}).setdefault(second, {})
                        for key, spent in keys.items():
                            mine[key] = mine.get(key, 0.0) + spent
    first, last = STATE.get("from", 0), STATE.get("until", 1 << 62)
    whole = {role: {second: keys for second, keys in by_second.items() if first <= int(second) < last}
             for role, by_second in whole.items()}
    with open(os.path.join(out, "where.json"), "w") as f:
        json.dump({"seconds": max(last - first, 0), "roles": whole}, f)
    print(f"WHERE {max(last - first, 0)} s of flight, from {FROM} s on")
    for role in ("main thread", "worker threads", "build processes"):
        jobs, leaves, modules, outsides = (defaultdict(float) for _ in range(4))
        waiting = total = 0.0
        for keys in whole.get(role, {}).values():
            for key, spent in keys.items():
                job, leaf, outside = key.split("|")
                total += spent
                if outside == "(waiting)" or leaf == "None":
                    waiting += spent if outside == "(waiting)" else 0.0
                    if leaf == "None" and outside != "(waiting)":
                        outsides[outside] += spent
                    continue
                jobs[job] += spent
                leaves[leaf] += spent
                modules[leaf.split(".")[0]] += spent
                if outside:
                    outsides[f"{outside} (from {leaf.split('.')[0]})"] += spent
        busy = sum(jobs.values())
        if not total:
            continue
        print(f"WHERE == {role}: {busy:.2f} s in the add-on, {waiting:.2f} s waiting, {total - busy - waiting:.2f} s "
              f"in neither")
        table("by job", jobs, busy)
        table("by module (innermost add-on function's)", modules, busy)
        table("by innermost add-on function", leaves, busy, most=16)
        table("innermost function outside the add-on", outsides, total, most=8)


if __name__ == "__main__":
    boot()
