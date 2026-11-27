"""Builds off the main thread.

What an endless map shows is built again and again around the drone, level by level, and a build is arithmetic on
arrays (which tiles are near, their bars' corners, cuts) with one short step at the end that only Blender's main
thread may take: writing the mesh. A frame at 60 a second has about 2 ms to spare for the flight's tick, and a build
takes 5 to 50, so every build done in the tick was a frame or three dropped (the Penrose surfaces at 37 to 49 frames a
second in a dive, a third of the time in builds).

Here a build's arithmetic runs on worker threads (WORKERS of them) while the main thread goes on flying and drawing:
the engine hands over a function and what it needs (submit()), goes on showing what it has, and takes the result in
a later tick (finished()) to write. Python runs one thread's Python at a time, but numpy lets go of that lock inside
its array loops, and Blender holds it only while it runs a script, so a worker whose time is in numpy runs beside
the frame's evaluation and drawing. The function mustn't touch bpy, and must be given its own copies of whatever the
main thread may change meanwhile.

In a background Blender (build scripts, the flight harness, renders) nothing is handed over (enabled is False):
every build is done where it's asked for, so a run comes out the same every time.

Jobs go by priority (lower first), and in the order they came at one priority: a level the drone is already past the
edge of comes before one built again in good time, which comes before one built ahead of need. In order of arrival
alone, a dive's new levels waited behind a dozen refreshes of levels still covering the view (the Penrose sphere's
levels drawn short of the drone in two thirds of a dive's ticks). A job not started yet can be handed in again
(submit(replace=True)): it's then done with the fresher arguments and the new priority, in its place among those as
good (handed in again every tick, a job went to the back of the line each time, and in a fast glide the floor's and the
3D parts' builds waited up to 2.4 s, then were built round where the drone had been).

The priorities are the engines' values (schedule.py: how much of the view a build puts right, how
soon, for its cost), worked out again every tick, better or worse, and a free worker takes the best job waiting at that
moment (from all of them: some 50 at most). In a line of fixed priorities, a fast dive's floor always had a level drawn
short of the drone to build first, and the 3D parts' new levels waited a third of a second and more behind them, built
as the drone flew past them: the finest three or four levels of arches missing in nine ticks of ten.

Taking turns: a thread that lets go of Python's lock often (every numpy call on a big array does) takes it from the
main thread each time the main thread lets go of it too, and the main thread, which runs the flight's tick, the frame's
handler and the draws, then waits for it at every such call: with a worker busy, each of them took half as long again
to twice as long. So the main thread's Python is marked (foreground), and a worker asks between its steps whether
the main thread is in it (give_way()), and if so waits until it's out: Blender's own work between them, the scene's
evaluation and the drawing, most of a frame, needs no Python, and is the workers' time.

A worker in pure Python (exact Fractions, deep down) never lets go of the lock by itself: Python takes it from it only
every switch interval, 5 ms by default, and the main thread waited that long each time it came into Python, which it
does many times a frame (the tick, each handler, each draw's), with or without foreground. Climbing out of a deep
dive, where every coarser level's tiles are placed exactly, the frame rate fell from 30 to 7. So the workers start with
a switch interval of SWITCH_INTERVAL, and their exact steps give way a face at a time.

Python's garbage collector holds the lock through a whole collection, whichever thread it runs in, and a full one goes
over every object Python is keeping: with a smooth map's subdivision trees kept (some 5 million objects) that was 1.3 s,
the frame stopped while a worker collected, and a start's dozen of them 3.4 s of 16 (the smooth UV sphere). So while
flying (hold_heap()), what has come through a collection of the younger generations is frozen out of the later ones
(gc.freeze()): the caches the engines keep are long-lived, and a full collection then goes over only what's new.
"""

import functools
import gc
import inspect
import io
import itertools
import math
import os
import pickle
import secrets
import subprocess
import sys
import threading
import time
import traceback

try:
    import bpy
    enabled = not bpy.app.background
except ImportError:
    enabled = False          # outside Blender (the engines' own tests)
FAILED = object()            # a result: the function raised (its traceback is printed)

_work_of = {}                # key -> (function, args) of its job still waiting
_lock = threading.Lock()
_ready = threading.Condition(_lock)     # notified when a job comes to wait
_pending = {}                # key -> (the job's number, its priority), while it's waiting or being worked on
_running = set()             # the keys of the jobs being worked on
_results = {}                # key -> result, finished and not yet taken
_ran = {}                    # key -> how long its last job ran (s): what a build costs (schedule.Costs)
_number = 0
_threads = []
# How many worker threads: numpy lets go of Python's lock in its array loops, so two builds run on two cores at once
# for the most part (on one, a fast glide's 3D parts, strapwork and cuts kept it busy 9 s of 10, and the strapwork's
# builds waited up to 2 s). Each runs its builds in a process of its own (PROCESSES): three were no more
# of the view drawn in time than two on an M1, and the flight's own frames over 33 ms went from 7 to 51 (three build
# processes and Blender's main thread filled its four fast cores)
WORKERS = 2
# Python's switch interval once the workers run, seconds (above: at 1 ms the climb still fell to 17 frames a second, at
# 0.2 to 23 at worst)
SWITCH_INTERVAL = 0.0002


# ----- the garbage collector -----

def _freeze_survivors(phase, info):
    # Right after a collection of the middle generation: what's left in them is long-lived (and nothing new is yet)
    if phase == "stop" and info.get("generation", 0) >= 1:
        gc.freeze()


def hold_heap(on):
    """On (a flight starts): from now on, what lives through a collection of Python's younger generations is frozen, so
    no collection goes over it again (see the top). Off (it ends): thawed, back in the oldest generation, for Python's
    own collections to go over in their time. A cycle that becomes garbage once frozen waits until then."""
    if on:
        if _freeze_survivors not in gc.callbacks:
            gc.callbacks.append(_freeze_survivors)
    elif _freeze_survivors in gc.callbacks:
        gc.callbacks.remove(_freeze_survivors)
        gc.unfreeze()


# ----- taking turns -----

# How deep the main thread is in its own Python (foreground), and set while it's out of it
_busy = 0
_free = threading.Event()
_free.set()
_main = threading.main_thread().ident
# The longest a worker waits for the main thread at once, seconds: its Python never runs that long, so past it the
# main thread has left without saying so (an error unwound it)
GIVE_WAY_MOST = 0.05
# Off: workers never wait (to measure what it's worth)
GIVE_WAY = True
# A worker gives way at most once this long of its own work (s): asked at every small step (a smooth floor's faces,
# some 920 a round), each wait up to the main thread's whole tick, a floor build of a few seconds' work took 35 s on
# the knot, and a 3D parts window 50 s, each a worker's whole flight
GIVE_WAY_EVERY = 0.0005
_last_wait = threading.local()


def foreground(function):
    """Decorator for the main thread's Python that a frame waits for (a flight tick, a frame handler, a draw): while
    it runs, the workers that give_way() keep out of Python.

    The function is handed no more arguments than it takes. Blender gives a handler the scene and the depsgraph, or
    the scene alone if it's a function of one argument, which it can't tell of the wrapper: the frame's handler,
    which takes the scene, was given both and failed every frame, and no stick reached the drone."""
    code = getattr(function, "__code__", None)
    most = code.co_argcount if code is not None and not code.co_flags & inspect.CO_VARARGS else None

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        global _busy
        _busy += 1
        _free.clear()
        try:
            return function(*args[:most], **kwargs)
        finally:
            _busy -= 1
            if _busy <= 0:
                _busy = 0
                _free.set()
    return wrapper


def give_way():
    """From a worker thread, between its steps: waits while the main thread is in its Python (foreground), if it's
    worked GIVE_WAY_EVERY since it last did. On the main thread itself, nothing."""
    if _busy and GIVE_WAY and threading.get_ident() != _main:
        now = time.perf_counter()
        if now - getattr(_last_wait, "at", -math.inf) < GIVE_WAY_EVERY:
            return
        _free.wait(GIVE_WAY_MOST)
        _last_wait.at = time.perf_counter()


def _take():
    """The next job: the waiting one with the best priority (the earliest of those), waited for if there's none:
    (number, key, its function and arguments)"""
    with _ready:
        while True:
            waiting = [key for key in _work_of if key not in _running]
            if waiting:
                break
            _ready.wait()
        key = min(waiting, key=lambda k: (_pending[k][1], _pending[k][0]))
        _running.add(key)
        return _pending[key][0], key, _work_of.pop(key)


def _work():
    while True:
        number, key, (function, args) = _take()
        # give way only before a job's work (giving way before each stale entry of the old line too, up
        # to GIVE_WAY_MOST each, kept both workers idle half a fast dive, the 3D frames' levels left unbuilt)
        give_way()
        started = time.perf_counter()
        try:
            result = call(function, *args)
        except Exception:
            traceback.print_exc()
            result = FAILED
        with _lock:
            _ran[key] = time.perf_counter() - started
            _running.discard(key)
            if _pending.get(key, (None,))[0] == number:
                del _pending[key]
                _results[key] = result
            elif key in _work_of:
                # Handed in again while this one ran (cancelled, then submitted): it can be taken now
                _ready.notify()


def submit(key, function, *args, priority=0, replace=False):
    """Have function(*args) worked out on a worker thread (True), unless key's last job is still waiting or under way
    (False); with replace, a job of key's still waiting gets these arguments and this priority (lower first), in its
    place among those as good (not one under way). key is a tuple whose first item names its owner (a map's object's
    name)."""
    global _number
    with _ready:
        if key in _pending:
            if not replace or key in _running:
                return False
            _pending[key] = (_pending[key][0], priority)
            _work_of[key] = (function, args)
            return True
        _number += 1
        _pending[key] = (_number, priority)
        _work_of[key] = (function, args)
        _results.pop(key, None)
        _ready.notify()
    if len(_threads) < WORKERS:
        sys.setswitchinterval(SWITCH_INTERVAL)
    while len(_threads) < WORKERS:
        _threads.append(threading.Thread(target=_work, name=f"er-builds-{len(_threads)}", daemon=True))
        _threads[-1].start()
    return True


def rate(key, priority):
    """A waiting job of key's gets a new priority (its engine's value of it now: schedule.py), better or worse"""
    with _lock:
        if key in _pending and key not in _running:
            _pending[key] = (_pending[key][0], priority)


def ran(key):
    """How long key's last job ran on a worker thread (s), or None"""
    return _ran.get(key)


def pending(key):
    return key in _pending


def running(key):
    """Whether key's job is being worked on now (not just waiting)"""
    return key in _running


def cancel(key):
    """Drop key's job and result, if any: its owner has done the work itself"""
    with _lock:
        _pending.pop(key, None)
        _work_of.pop(key, None)
        _results.pop(key, None)


def finished(owner):
    """The owner's finished jobs, as [(key, result)], each handed out once. A result is what the function returned, or
    FAILED."""
    with _lock:
        keys = [key for key in _results if key[0] == owner]
        return [(key, _results.pop(key)) for key in keys]


def forget(owner=None):
    """Drop the owner's jobs and results (everyone's with None): a file was loaded, or its map is gone. A job under
    way runs to its end and is thrown away. Everyone's: the build processes start again, the shared objects they had
    no longer the map's."""
    global _generation
    if owner is None:
        _generation += 1
        _shared.clear()
        _shared_key.clear()
    with _lock:
        for table in (_pending, _work_of, _results):
            for key in [key for key in table if owner is None or key[0] == owner]:
                del table[key]
        for key in [key for key in _ran if owner is None or key[0] == owner]:
            del _ran[key]


# ----- builds in processes of their own -----
#
# Python runs one thread's Python at a time, so the worker threads took turns with the flight's tick for one core of the
# machine's (an M1's eight), and a fast flight's builds waited in line: in a dive and a climb back, up to 50 builds
# waiting, the oldest a second, the floor's finer levels and the 3D frames' missing from 2 to 25% of the view
# . A function marked builds_apart (an attribute set on it: its arguments and result pickle, it touches no bpy and no
# state kept between builds) is worked out by a Python process of the worker thread's own
# (build_process.py), which it starts the first time: the thread waits on its connection without Python's lock.
# Objects every build shares and too big to send with each (a surface's Topology: its class has shared_by_key, and
# shared_data() and from_shared() to send and make it) are sent to a process once and then named by key in what's
# pickled (dumps(), loads()).

# Off: every build in the worker threads, as before
PROCESSES = True
# How long a new build process may take to connect (s), past which builds stay in the threads
PROCESS_START = 30.0
_BOOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_process.py")
_shared = {}                 # key -> an object sent to processes once, by key
_shared_key = {}             # id(object) -> its key
_shared_numbers = itertools.count(1)
_here = threading.local()    # a worker thread's process (_Process)
_generation = 0              # forget() moves it on: processes started before it are let go
_processes = []              # every process started, to let go of at the end


def share(obj, key=None):
    """obj shared by key (made one if None): sent to a process once, then named by key (dumps())"""
    if key is None:
        key = f"{os.getpid()}-{next(_shared_numbers)}"
    _shared[key] = obj
    _shared_key[id(obj)] = key
    return key


class _Pickler(pickle.Pickler):
    """pickle, with every shared object (its class's shared_by_key) named by its key, the keys noted in keys"""

    def __init__(self, file, keys):
        super().__init__(file, pickle.HIGHEST_PROTOCOL)
        self.keys = keys

    def persistent_id(self, obj):
        if not getattr(type(obj), "shared_by_key", False):
            return None
        key = _shared_key.get(id(obj))
        if key is None or _shared.get(key) is not obj:
            key = share(obj)
        self.keys.add(key)
        return key


class _Unpickler(pickle.Unpickler):
    def persistent_load(self, key):
        return _shared[key]


def dumps(value):
    """value pickled, its shared objects by key: (the bytes, the keys)"""
    buffer, keys = io.BytesIO(), set()
    _Pickler(buffer, keys).dump(value)
    return buffer.getvalue(), keys


def loads(data):
    return _Unpickler(io.BytesIO(data)).load()


# A connection's writes at most this many bytes at a time, retried while the system is out of buffers for a moment
# (sturdy())
SEND_CHUNK = 1 << 18
SEND_RETRIES = 2000


def sturdy(connection):
    """connection (multiprocessing's, over a socket) writing in chunks, each retried while the system has no buffer
    space for it (ENOBUFS, EAGAIN; a write that fails writes nothing, so the stream stays whole): one write of a
    level's 3 MB answer failed so on macOS, mid-glide, and the build process died with it. Not on Windows, whose sockets
    aren't written with os.write (multiprocessing's own send there)."""
    import errno
    if sys.platform == "win32":
        return connection
    handle = connection.fileno()
    write = os.write

    def send(buf, write=write):
        view = memoryview(buf).cast("B")
        tries = 0
        while len(view):
            try:
                n = write(handle, view[:SEND_CHUNK])
            except InterruptedError:
                continue
            except OSError as error:
                if error.errno not in (errno.ENOBUFS, errno.EAGAIN) or tries >= SEND_RETRIES:
                    raise
                tries += 1
                time.sleep(0.001)
                continue
            tries = 0
            view = view[n:]
    connection._send = send
    return connection


def _core_folder():
    """Where this Blender imports the eternal_return_core wheel from ("" if it has none): a build process is told,
    since Blender's own Python started bare knows nothing of an extension's wheels"""
    import importlib.util
    try:
        spec = importlib.util.find_spec("eternal_return_core")
    except (ImportError, ValueError):
        spec = None
    places = list(spec.submodule_search_locations or ()) if spec is not None else []
    return os.path.dirname(places[0]) if places else ""


class _Process:
    """A build process (build_process.py), connected: which shared objects it has, and the generation it's of"""

    def __init__(self):
        from multiprocessing.connection import Listener
        key = secrets.token_bytes(16)
        listener = Listener(("127.0.0.1", 0), authkey=key)
        try:
            # (Not for ever, if it never comes)
            socket = getattr(getattr(listener, "_listener", None), "_socket", None)
            if socket is not None:
                socket.settimeout(PROCESS_START)
            host, port = listener.address
            self.process = subprocess.Popen(
                [sys.executable, "-I", _BOOT, f"{host}:{port}", key.hex(), os.path.dirname(_BOOT),
                 __name__.rpartition(".")[0], _core_folder()], stdin=subprocess.DEVNULL)
            self.connection = sturdy(listener.accept())
        finally:
            listener.close()
        self.sent = set()
        self.generation = _generation

    def call(self, function, args):
        data, keys = dumps((function, args))
        for key in sorted(keys - self.sent):
            obj = _shared[key]
            self.connection.send(("define", key, type(obj).__module__, type(obj).__qualname__, obj.shared_data()))
            self.sent.add(key)
        self.connection.send(("call", data))
        kind, payload = self.connection.recv()
        if kind == "error":
            raise RuntimeError(f"in a build process:\n{payload}")
        return loads(payload)

    def close(self):
        try:
            self.connection.close()
        except OSError:
            pass
        try:
            self.process.wait(1.0)
        except subprocess.TimeoutExpired:
            self.process.kill()


def _process_here():
    """This worker thread's build process, started if it has none (None if processes are off or won't start)"""
    global PROCESSES
    if not PROCESSES or threading.get_ident() == _main:
        return None
    process = getattr(_here, "process", None)
    if process is not None and (process.generation != _generation or process.process.poll() is not None):
        process.close()
        process = None
    if process is None:
        try:
            process = _Process()
        except Exception:
            traceback.print_exc()
            print("Eternal Return: builds stay in the worker threads (no build process started)")
            PROCESSES = False
            return None
        _here.process = process
        _processes.append(process)
    return process


def call(function, *args):
    """function(*args): in this worker thread's build process if function builds_apart, else here"""
    if getattr(function, "builds_apart", False):
        process = _process_here()
        if process is not None:
            try:
                return process.call(function, args)
            except (EOFError, OSError):
                # The process went (killed, crashed): this build here, a new process for the next
                _here.process = None
    return function(*args)


def stop_processes():
    """Let every build process go (they end when their connection closes)"""
    global _generation
    _generation += 1
    for process in _processes:
        process.close()
    _processes.clear()
