"""What the endless engines build ahead of the drone, and in what order: one policy for the floor's strapwork
(surface_penrose.py) and the 3D parts (parts.py), which share the worker threads (background.py).

2026-11-19 (the user, about the floor's strips and the arches: should they share their logic?): each engine had worked
out for itself where the drone would be, how long its builds took and how urgent each was, and on a Penrose map with
frames each cut the same levels' tiles for itself: the floor in the tick (2.5 s of the main thread in a 10 s fast dive),
the parts again in each of their builds on a worker thread (some 40% of their time there). Shared here:
- the drone's motion over the surface (Motion): its velocity, smoothed in drone heights a second, one for a map, so
  both build round the same point ahead; and how long each one's builds take to be taken (its latency), which says
  how far ahead that is
- what a build is worth, both engines' by one rule (value(), below), the worker threads' priority
And in surface_penrose.py, a Penrose map's tiles near the drone (tiles()): each level's cut once, for every build that
needs it, floor or parts, on the worker thread, inside the first build that needs them, as wide as every build asking
for the level wants (needs()).

Each engine still decides when one of its own levels needs building (a floor level's disc against a parts level's
cells), and builds it its own way (parts.py is shared by every lattice).
"""

import math

import numpy as np

try:
    from . import background
except ImportError:          # tests import it by itself
    import background

# A built window is built again once it's this many times wider than a build now would make it (Window.overbuilt():
# the drone has shrunk under it). Kept, a Penrose level built when it was small on screen kept all its bars out to far
# past the view as the drone dived on, every level ever drawn (2026-11-13: 580k faces 36 loops down, and every line gone
# further down, its objects scaled by every loop since past what float32 and the push drivers can place); and a floor's
# or a giant parts window's corners are float32 from its middle, so they'd be a growing share of the drone's height off.
OVERBUILT = 4.0


class Window:
    """What a build round the drone was for, as either engine keeps it: the world (its text and n), the window's middle
    (the camera or the point below it then, map units then: what's built is from it) and half width; with where that is
    in today's map, kept until the world changes (transform: surface.transform()'s or the lattice's, placed: the
    engine's own)"""

    def __init__(self, world_text, n, center, window):
        self.world_text, self.n, self.center, self.window = world_text, n, center, window
        self.transform = None
        self.placed = None

    def covers(self, distance, reach, scale=1.0):
        """Whether the window still holds everything within reach of a point distance from its middle: in the build's
        units, or in today's with scale, how much bigger map units are now. False for a NaN (a place past float
        range)."""
        return distance + reach <= scale * self.window

    def overbuilt(self, wanted, scale=1.0):
        """Whether the window is more than OVERBUILT times as wide as wanted (units as covers()'s). True for a NaN."""
        return not scale * self.window <= OVERBUILT * wanted


# What a build is worth (2026-11-20), for both engines, as the worker threads' priority (background.submit(): lower
# first, so its value's negative): value = stake * urgency / cost
# - stake: the share of the view it puts right (view_share() of what it adds, times how visible that is when it lands: a floor
#   level still fading in counts for what it shows), REFRESH of it for a build that leaves nothing missing (cut finer for a
#   nearer camera, outlines moved, trimmed)
# - urgency: wanted needed_in s from now, 1 if that's no later than it can land (the engine's wait()), falling off by
#   URGENCY_TIME s past that
# - cost: what it'll take of the workers' time, learned from what its kind took (Costs)
# Until then each engine had a table of fixed priorities (the floor's "now", "soon", "ahead", "recoarse", coarse levels
# first; the 3D parts' "urgent" and "later", by rank), one line for both, and in a fast dive the floor always had a level
# short of the drone to build first: the parts' new levels waited a third of a second behind them and more, built as
# the drone flew past them, the finest three or four levels of arches missing in nine ticks of ten.
# REFRESH was 0.1 until 2026-11-20 (the user, hovering on the Penrose torus: "the floor is a solid color"): some twenty
# coarse floor levels came due at once (cut finer, trimmed), each worth a tenth of the whole view, and the finer levels
# not drawn at all, their discs a few percent of it, waited behind them in nine ticks of ten. Nothing left missing
# comes after anything that is.
URGENCY_TIME = 0.25
REFRESH = 0.001
# The view's reach over the surface, drone heights round the point below (it looks ahead and down)
VIEW_REACH = 1.5
# A cost guess for a kind never measured (s a unit), unless its engine gives one
COST_FIRST = 0.03


def view_share(radius, h):
    """The share of the view a disc of radius (map units) round the point below fills, the drone h high (map units)"""
    with np.errstate(over="ignore", invalid="ignore"):
        share = (radius / (VIEW_REACH * h)) ** 2 if h > 0.0 else 1.0
    return min(1.0, share) if share == share else 1.0


def value(stake, needed_in, wait, cost):
    """A build's worth a second of the workers' time (see above)"""
    late = max(0.0, needed_in - wait)
    return stake / (1.0 + late / URGENCY_TIME) / max(cost, 1e-4)


def priority(stake, needed_in, wait, cost):
    """value() as a priority for background.submit() (lower first)"""
    return -value(stake, needed_in, wait, cost)


class Costs:
    """What builds of each kind take of the workers' time, a unit (a build, or a 3D parts build's cells squared),
    smoothed"""

    def __init__(self):
        self.per = {}

    def guess(self, kind, units=1.0, first=COST_FIRST):
        return self.per.get(kind, first) * units

    def took(self, kind, seconds, units=1.0):
        if seconds is None or not units > 0.0:
            return
        was = self.per.get(kind)
        each = seconds / units
        self.per[kind] = each if was is None else was + 0.2 * (each - was)


# The drone's speed over the surface is smoothed over this long (s); a build's wait is counted as at most LATENCY_MOST
# (s), LATENCY_FIRST until one is measured
MOTION_TIME = 0.2
LATENCY_FIRST, LATENCY_MOST = 0.1, 0.5
# Asked again for the same point this soon, it's the same tick's (the floor's settle can take 50 ms before its 3D
# parts ask; a drone held still has the same point tick after tick, and its velocity goes to 0 this late)
SAME_TICK = 0.25


class _At:
    """A point of a map as some world had it (an engine's transform() takes it to another's)"""

    def __init__(self, world_text, n, point):
        self.world_text, self.n, self.center = world_text, n, point
        self.transform = None


class Motion:
    """A map's drone over its surface: the point below's velocity (velocity()) and how long each engine's builds on the
    worker threads take to be taken (wait(), took())"""

    def __init__(self):
        self.last = None             # (_At the last point below, when, its velocity in drone heights a second, the
                                     # drone's height then)
        self.latency = {}            # engine -> its builds' wait, smoothed (s)
        self.descent = 0.0           # d(ln h) / dt, smoothed over MOTION_TIME: below 0 diving, above climbing
        self.costs = Costs()         # what each kind of build takes

    def velocity(self, world, text, point, h, now, transform):
        """The point below's velocity over the surface (map units of this world a second), smoothed over MOTION_TIME in
        the drone's heights (h) a second; transform(world, text, record) -> (offset, scale) places a point of another
        world's map in this one's. Zero in a background Blender, whose builds are done where they're asked for. Asked
        again in the same tick (the floor, then its 3D parts), the same answer.

        In heights, not map units: diving with auto scale, the drone shrinks several times a second and the point's
        speed in map units with it (by a third a tick), then a loop makes it 8 times as much, while in heights it
        holds; smoothed in map units, the speed lagged its fall and each loop multiplied what was left, some 35 times
        the true speed, and every floor level near the finest was cut round a point 4 to 12 of its windows off, in the
        tick, every tick (2026-11-19: 2.6 s of a fast dive's 10, 5 to 9 frames a second)."""
        point = np.asarray(point, dtype=float)
        last = self.last
        if (last is not None and last[0].world_text == text and now - last[1] < SAME_TICK
                and np.array_equal(last[0].center, point)):
            return last[2] * h
        relative = np.zeros(3)
        descent = 0.0
        if background.enabled and last is not None:
            at, then, before, h_then = last
            offset, scale = (np.zeros(3), 1.0) if at.world_text == text else transform(world, text, at)
            dt = now - then
            if dt <= 0.0:
                relative, descent = before, self.descent
            elif dt < 4.0 * MOTION_TIME:
                share = min(1.0, dt / MOTION_TIME)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    seen = (point - (offset + scale * at.center)) / (dt * math.sqrt(h_then * scale * h))
                    relative = before + (seen - before) * share
                    then_h = h_then * scale
                    ratio = h / then_h if then_h > 0.0 else math.inf
                if not np.all(np.isfinite(relative)):
                    relative = np.zeros(3)
                if 0.0 < ratio < math.inf:
                    descent = self.descent + (math.log(ratio) / dt - self.descent) * share
        self.descent = descent
        self.last = (_At(text, world.n, point.copy()), now, relative, h)
        return relative * h

    def wait(self, engine):
        """How long a build of engine's asked for now will take to be taken, as far as building ahead goes (s)"""
        return min(self.latency.get(engine, LATENCY_FIRST), LATENCY_MOST)

    def took(self, engine, seconds):
        """A build of engine's was taken seconds after it was asked for"""
        latency = self.latency.get(engine, LATENCY_FIRST)
        self.latency[engine] = latency + 0.2 * (seconds - latency)


_motions = {}            # map object name -> Motion


def motion_of(name):
    found = _motions.get(name)
    if found is None:
        found = _motions[name] = Motion()
    return found


def forget(name=None):
    """A file was loaded (every map's motion), or a map's flight starts over (name)"""
    if name is None:
        _motions.clear()
    else:
        _motions.pop(name, None)
