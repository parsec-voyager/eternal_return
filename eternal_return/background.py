"""Builds off the main thread.

What an endless map shows is built again and again around the drone, level by level, and a build is arithmetic on
arrays (which tiles are near, their bars' corners, cuts) with one short step at the end that only Blender's main
thread may take: writing the mesh. A frame at 60 a second has about 2 ms to spare for the flight's tick, and a build
takes 5 to 50, so every build done in the tick was a frame or three dropped (2026-11-14: the Penrose surfaces at 37 to
49 frames a second in a dive, a third of the time in builds).

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
alone, a dive's new levels waited behind a dozen refreshes of levels still covering the view (2026-11-17: the Penrose
sphere's levels drawn short of the drone in two thirds of a dive's ticks). A job not started yet can be handed in again
(submit(replace=True)): it's then done with the fresher arguments and the new priority, in its place among those as
good (2026-11-19: handed in again every tick, a job went to the back of the line each time, and in a fast glide the
floor's and the 3D parts' builds waited up to 2.4 s, then were built round where the drone had been).

Since 2026-11-20 the priorities are the engines' values (schedule.py: how much of the view a build puts right, how
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
dive, where every coarser level's tiles are placed exactly, the frame rate fell from 30 to 7 (2026-11-19). So the
workers start with a switch interval of SWITCH_INTERVAL, and their exact steps give way a face at a time.

Python's garbage collector holds the lock through a whole collection, whichever thread it runs in, and a full one goes
over every object Python is keeping: with a smooth map's subdivision trees kept (some 5 million objects) that was 1.3 s,
the frame stopped while a worker collected, and a start's dozen of them 3.4 s of 16 (2026-11-24, the smooth UV sphere).
So while flying (hold_heap()), what has come through a collection of the younger generations is frozen out of the later
ones (gc.freeze()): the caches the engines keep are long-lived, and a full collection then goes over only what's new.
"""

import functools
import gc
import inspect
import math
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
# for the most part (2026-11-19: on one, a fast glide's 3D parts, strapwork and cuts kept it busy 9 s of 10, and the
# strapwork's builds waited up to 2 s)
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
# the knot, and a 3D parts window 50 s, each a worker's whole flight (2026-11-25)
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
        # give way only before a job's work (2026-11-19: giving way before each stale entry of the old line too, up
        # to GIVE_WAY_MOST each, kept both workers idle half a fast dive, the 3D frames' levels left unbuilt)
        give_way()
        started = time.perf_counter()
        try:
            result = function(*args)
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
    way runs to its end and is thrown away."""
    with _lock:
        for table in (_pending, _work_of, _results):
            for key in [key for key in table if owner is None or key[0] == owner]:
                del table[key]
        for key in [key for key in _ran if owner is None or key[0] == owner]:
            del _ran[key]
