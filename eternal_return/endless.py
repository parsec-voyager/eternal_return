"""Endless maps: maps whose levels go on forever, bigger as well as smaller, so loops are seamless anywhere.

A map built of levels on nested lattices (each level's cells a whole loop factor bigger than the next one's, like the
Inferno's 3 x 3 cells or 2 x 2 ones) is the same after a loop about a point where the lattices line up. But a map can
only build so many levels. Fly into the wall of the biggest pillar, and the loop has to turn it into a pillar a loop
factor bigger, which isn't there. Without a biggest level, diving at one spot forever means the drone's position,
measured from the map's center, grows by the loop factor every loop, past what float32 can hold in a dozen loops.

So an endless map keeps the drone's position in two parts: an exact anchor, whole numbers of steps in Python integers,
which can grow without limit, and the drone's location from the map's origin, which stays within a step. The map's
origin is wherever the anchor puts it in the endless lattice. The anchor counts along the axes the map repeats along
(the "axes" setting): X and Y for a floor that auto scale measures height above, one axis along a line, all three for a
lattice that fills space. Along the others, auto scale keeps the drone within a band, so nothing grows.
- Every level whose pattern repeats over the step doesn't see the anchor at all, since moving by whole steps doesn't
  change it. Their patches follow the drone as before.
- Bigger levels follow the drone in steps of their own period, offset by the anchor's residue there (the anchor modulo
  the period), which this module writes to the map's object for their drivers ("endless_rx", "endless_ry", ...).
- The biggest levels' parts can be hundreds of kilometers across, a meter away. A mesh scaled that big puts float32
  rounding of meters into its faces. So they're built from the anchor exactly, by tiling.py: the parts the map's
  lattice draws in every cell of every level (the Inferno's pillars and its floor's strips, say), every face placed
  exactly, then rounded once. A face close to the drone is close to the map's origin, so it's exact to within float32
  of that.

The anchor is a stack of digits, one per level: which of the factor x factor smaller cells the drone is in at that
level. Diving pushes digits onto it (where you fly decides them), and climbing pops them off. Below the ones pushed,
every digit is the same, the map's origin: climbing past them, the arches came back centered on it. So with "random"
settings, the stack is kept made up to a few levels beyond any you can see, with random digits (make_up_digits()).
Climbing pops one and a new random one goes on the far end; it's stored in the anchor like the rest, so diving back
down and climbing again meets the same ones.

Loops have to scale by a whole number: only then do the levels nest, and does the pattern of every level in view repeat
over the same distance, which moving the drone back by steps and the random digits both need. (A map that's only
self-similar about one point, like concentric spheres, can loop by any factor, but needs none of this.)

A map with "fractal" settings is a fractal that fills space, like a Sierpinski tetrahedron, whose every level
depends on the anchor: fractal.py builds its cells around the drone, its random digits are its kept ones, and auto
scale in mesh mode measures to the shapes it draws. See there.

The add-on calls scale_anchor() when it loops about an endless map's object, and recenter_steps() and shift_anchor()
each tick to move the drone back by whole steps (see ER_OT_Fly.recenter_endless()). The map's object holds:
- "endless": its settings (see SETTINGS), written by the map's build script
- "endless_anchor": the anchor, one whole number of steps per anchored axis, as text so it can be any size (in hex:
  Python won't turn an integer of over 4300 digits into decimal, which 9000 diving loops make it)
- "endless_horizon": the highest digit of the anchor made up so far (the stack's far end), a power of the factor in steps
"""

import contextlib
import itertools
import math
import random
from fractions import Fraction

import mathutils

from . import tiling

PROPERTY = "endless"
ANCHOR = "endless_anchor"
HORIZON = "endless_horizon"
AXIS_NAMES = "xyz"

# The "endless" property's keys:
# - factor: the loop factor the lattice repeats under, a whole number
# - step: how far the drone moves back to stay near the map's origin. Every level smaller than the residues' must
#   repeat over it.
# - axes: the axes of the map's object the anchor counts along, like [0, 1] (the default) for X and Y
# - periods: the periods to write the anchor's residues for, in order: the levels whose patterns repeat over more than
#   a step. Written as "endless_r" plus the axis's letter, one list per anchored axis.
# - tiling: the parts the lattice draws in every cell of every level, like pillars standing on the floor or the
#   floor's strips, whose levels too big for the map's patches are built here, from the anchor exactly (see tiling.py,
#   which lists its settings)
# - random: random digits made up beyond the stack, or omitted for the map's origin there:
#   - from: the lowest digit (power of the factor, in steps) that's out of sight, for every level that can be seen to
#     repeat over it. The digits below are where the drone is.
#   - to: the highest digit to keep made up, so a digit is made up a few levels before it can be seen
#   - background, background_depth: optional: the digits above "to", instead of 0, a pattern of digit tuples (one per
#     anchored axis) repeating upward from "to" + 1, kept in the anchor for background_depth digits (0 beyond them).
#     With 0 above, the drone is at the lattice's origin in every bigger level, which must be clear of everything; in
#     a 2 x 2 lattice that's a corner, and a digit made up just below then puts the drone half a cell or a quarter
#     from it, on the edges where its lines run: every digit but 0 was refused (choose_digit()), and climbing brought
#     the zeros into sight, every level lined up on the drone. [[-1, -1], [0, 0]] puts it a third of the way across
#     every bigger level's cell instead, in the middle of its star. A fractal's are its kept digits, in its lattice's
#     own digits (see fractal.keep_background()), like [[0, 0, 0], [2, 2, 2]] for Cantor dust.
SETTINGS = ("factor", "step", "axes", "periods", "tiling", "random")


def is_endless(obj):
    return obj is not None and PROPERTY in obj


def exact(value):
    """A setting as an exact fraction, as written: 0.035 as 7/200, not the binary float nearest it."""
    return Fraction(repr(float(value)))


def anchored_axes(obj):
    return [int(a) for a in obj[PROPERTY].get("axes", (0, 1))]


def get_anchor(obj):
    """The anchor, one whole number of steps per anchored axis"""
    count = len(anchored_axes(obj))
    # Hex since 2026-10-29, decimal before
    values = [int(v, 0) for v in obj.get(ANCHOR, "").split()][:count]
    return tuple(values + [0] * (count - len(values)))


def local_location(obj, world_location):
    """world_location in obj's own coordinates, where the endless lattice is measured, along its anchored axes, as
    Python floats"""
    local = obj.matrix_world.inverted() @ world_location
    return tuple(local[a] for a in anchored_axes(obj))


def along_axes(obj, values):
    """values along obj's anchored axes, as a vector in obj's coordinates (0 along the others)"""
    vector = mathutils.Vector((0.0, 0.0, 0.0))
    for axis, value in zip(anchored_axes(obj), values):
        vector[axis] = value
    return vector


def recenter_steps(obj, camera_location):
    """How many whole steps along each anchored axis the drone camera at camera_location should move back toward the
    map's origin, all 0 while it's within a step of it. Moving back to within half a step, it doesn't move back and
    forth."""
    step = float(obj[PROPERTY]["step"])
    return tuple(round(c / step) if abs(c) > step else 0 for c in local_location(obj, camera_location))


def shift_anchor(obj, steps):
    """Add steps to the anchor, as the drone moves back by them."""
    from . import fractal
    if fractal.is_fractal(obj):
        fractal.shift_frames(obj, steps)
    set_anchor(obj, tuple(a + s for a, s in zip(get_anchor(obj), steps)))


def scale_anchor(obj, loop_factor, camera_location):
    """Scale the anchor with a loop of loop_factor about obj's origin, with the drone camera at camera_location (in
    world space, after the loop). It only scales by whole powers of the lattice's factor, the only loops it repeats
    under (anything else is left alone). Scaling down leaves a remainder of less than a step, which the drone has to
    move by, along obj's anchored axes in its units: returned as a tuple, or None."""
    factor = int(obj[PROPERTY]["factor"])
    power = round(math.log(loop_factor) / math.log(factor))
    if power == 0 or abs(factor ** power / loop_factor - 1.0) > 1e-6:
        return None

    anchor = before = get_anchor(obj)
    horizon = obj.get(HORIZON)
    if power > 0:
        anchor, remainder = tuple(a * factor ** power for a in anchor), None
    else:
        # Floor division, so the remainder is always toward + along each axis
        divisor = factor ** -power
        anchor, remainders = zip(*(divmod(a, divisor) for a in anchor))
        step = exact(obj[PROPERTY]["step"])
        remainder = tuple(float(r * step / divisor) for r in remainders)
    # The stack's digits moved up or down with the anchor's
    if horizon is not None:
        horizon += power
    anchor, horizon = make_up_digits(obj, anchor, horizon, camera_location, remainder)
    # A fractal's level objects hold the same cells after the loop, a level bigger or smaller (see fractal.py)
    from . import fractal
    if fractal.is_fractal(obj):
        fractal.scale_frames(obj, power, before)
    set_anchor(obj, anchor)
    if horizon is not None:
        obj[HORIZON] = horizon
    return remainder


def prepare(obj, camera_location):
    """Make up the random digits a flight needs before it starts (see make_up_digits()). Made up on the first loop
    instead, 40 of them at once (for 2 x 2) took 70 ms, a hitch."""
    anchor, horizon = make_up_digits(obj, get_anchor(obj), obj.get(HORIZON), camera_location)
    # Set even if no digits were made up, so the tiling is built by this version of the add-on: a file saved before
    # 2026-10-29 had its pillars shaded flat, which the first loop would have changed (see tiling.box_mesh())
    set_anchor(obj, anchor)
    if horizon is not None and horizon != obj.get(HORIZON):
        obj[HORIZON] = horizon
    # A fractal's levels are built for where the drone is, from scratch, since the file may have been saved mid-flight
    # or edited since
    from . import fractal
    if fractal.is_fractal(obj):
        fractal.refresh(obj, camera_location, rebuild=True)


def digits(factor):
    """The digits a random one is drawn from: one for each of the factor cells along an axis, around 0"""
    return range(-(factor // 2), factor - factor // 2)


def make_up_digits(obj, anchor, horizon, camera_location, remainder=None):
    """The anchor with random digits added beyond horizon, up to the random settings' "to", and the new horizon. A
    digit at a power of the factor picks one of the factor^axes cells for every level that doesn't repeat over that
    power of steps, the first of them (tiling.level_for()) by far the most. Each one is chosen so nothing the drone
    could see changes (no piece of those levels moves within the window, now or after; see choose_digit()). Unchanged
    without random settings. remainder: what the drone is about to move by, from scale_anchor()."""
    made_up = obj[PROPERTY].get("random")
    if made_up is None:
        return anchor, horizon
    first, last = int(made_up["from"]), int(made_up["to"])
    horizon = first - 1 if horizon is None else max(horizon, first - 1)
    if horizon >= last:
        return anchor, horizon

    factor = int(obj[PROPERTY]["factor"])
    # Where the drone is in the lattice, near enough, which a fractal's digits go by (fractal.choose_digit())
    drone = local_location(obj, camera_location)
    if remainder:
        drone = tuple(d + r for d, r in zip(drone, remainder))
    background = made_up.get("background")
    if background:
        # A fractal's are in its lattice's own digits (see fractal.keep_background())
        from . import fractal
        keep = fractal.keep_background if fractal.is_fractal(obj) else keep_background
        anchor = keep(anchor, background, last, last + int(made_up.get("background_depth", 12)), factor)
    while horizon < last:
        horizon += 1
        # Chosen instead of the digit there now: 0, or with a background, its digit that a climb moved down
        current = tuple(digit_at(a, factor, horizon) for a in anchor)
        anchor = choose_digit(obj, anchor, factor ** horizon, drone, current)
    return anchor, horizon


def keep_background(anchor, background, last, top, factor):
    """The anchor with its background's digits (see the random settings) up to top: a climb moves them all down a
    place, so the top one is continued from the pattern below it, as it stands; a file without them gets the pattern.
    Rewriting the pattern by place instead flipped its phase on every climb, a change in sight."""
    period = len(background)
    kept = []
    for axis, a in enumerate(anchor):
        if not any(digit_at(a, factor, place) for place in range(last + 1, top + 1)):
            a += background_value(background, axis, last + 1, top, last, factor)
        for place in range(top - period + 1, top + 1):
            a += (digit_at(a, factor, place - period) - digit_at(a, factor, place)) * factor ** place
        kept.append(a)
    return tuple(kept)


def digit_at(value, factor, place):
    """value's digit (see digits()) at factor^place"""
    return (split_digits(value, factor, place + 1)[0] - split_digits(value, factor, place)[0]) // factor ** place


def split_digits(value, factor, places):
    """value as (low, high): low written with the digits (see digits()) below factor^places, high the rest, a multiple
    of it"""
    lowest = -(factor // 2)
    low, rest, scale = 0, value, 1
    for _place in range(places):
        digit = (rest - lowest) % factor + lowest
        low += digit * scale
        rest = (rest - digit) // factor
        scale *= factor
    return low, rest * scale


def background_digit(background, axis, place, last):
    """The background's digit at a power of the factor (see the random settings' "background")"""
    return int(background[(place - last - 1) % len(background)][axis])


def background_value(background, axis, first, top, last, factor):
    """The background's digits from powers first to top along one axis, as a number of steps"""
    return sum(background_digit(background, axis, place, last) * factor ** place for place in range(first, top + 1))


def choose_digit(obj, anchor, place, drone, current=None):
    """The anchor with a random digit at place (a power of the factor, in steps) along each anchored axis, -1, 0 or 1
    for a factor of 3, instead of the current one there (default 0; see make_up_digits()): one that moves no piece of
    the tiling's parts in sight, nor any out of sight into it (tiling.in_window()). If one is in sight already, the
    digit is left as it is. A fractal's digit is one of its kept ones (see fractal.choose_digit()).
    """
    from . import fractal
    if fractal.is_fractal(obj):
        return fractal.choose_digit(obj, anchor, place, drone)
    settings = obj[PROPERTY]
    factor, step = int(settings["factor"]), exact(settings["step"])
    current = current or (0,) * len(anchor)
    shifted = lambda digit: tuple(a + (d - c) * place for a, d, c in zip(anchor, digit, current))
    candidates = list(itertools.product(digits(factor), repeat=len(anchor)))
    if "tiling" in settings:
        parts = tiling.parts(settings["tiling"])
        moved = place * step
        in_sight = lambda a: any(tiling.in_window(part, factor, anchor_position(obj, a), moved) for part in parts)
        if in_sight(anchor):
            return anchor
        candidates = [d for d in candidates if not in_sight(shifted(d))]
    return shifted(random.choice(candidates or [current]))


def anchor_position(obj, anchor):
    """Where the anchor puts the map's origin in the endless lattice, exactly, as (x, y, z)"""
    step = exact(obj[PROPERTY]["step"])
    position = [Fraction(0)] * 3
    for axis, a in zip(anchored_axes(obj), anchor):
        position[axis] = a * step
    return position


# The maps whose tilings wait to be built at the end of deferred_builds(), by name, or None outside it
_unbuilt = None


@contextlib.contextmanager
def deferred_builds():
    """Within it, set_anchor() only notes the maps whose tilings need building, and each is built once at its end, for
    its anchor then. A diving loop moves the anchor twice in one tick: the loop scales it, which takes the drone
    further from the map's origin, and the move back shifts it by whole steps. Each build of a big tiling takes tens of
    milliseconds (the strapwork floor's, 40 to 60 in the user's viewport), and building both froze the drone for up to
    137 ms at every loop (the user, 2026-11-05). Only the second was ever drawn. Nothing in a tick reads the built
    levels, only the anchor and the settings (choose_digit()), so they can wait. Nested, the outermost one builds."""
    global _unbuilt
    if _unbuilt is not None:
        yield
        return
    _unbuilt = {}
    try:
        yield
    finally:
        unbuilt, _unbuilt = _unbuilt, None
        for obj in unbuilt.values():
            build_tiling(obj)


def build_unbuilt():
    """Build the tilings deferred_builds() is holding back, now, for their anchors as they are: for whatever shows the
    map in the middle of a tick, like the flight harness, which renders right after a loop to measure its seam"""
    if _unbuilt:
        unbuilt = list(_unbuilt.values())
        _unbuilt.clear()
        for obj in unbuilt:
            build_tiling(obj)


def build_tiling(obj):
    """Build the map's tiling's biggest levels for its anchor (tiling.build()), and tag the map, so the drivers that
    read what the build wrote on it see the change: how many slots it filled and whether a piece covers the window,
    which push the floor's layers apart in depth. Built at the end of a tick, after the loop's own tag had already been
    evaluated mid-tick (scale() updates the view layer right after loop_scale()), they kept the old values until the
    next loop, and the floor's layers, pushed by the wrong amounts, flickered (the user, 2026-11-05)."""
    settings = obj[PROPERTY]
    tiling.build(obj, settings["tiling"], int(settings["factor"]), anchor_position(obj, get_anchor(obj)))
    obj.update_tag()


def set_anchor(obj, anchor):
    """Store the anchor and bring everything that depends on it up to date: the residues and the tiling's biggest
    levels (tiling.build(); within deferred_builds(), at its end)."""
    obj[ANCHOR] = " ".join(hex(a) for a in anchor)
    settings = obj[PROPERTY]
    step = exact(settings["step"])
    for axis, a in zip(anchored_axes(obj), anchor):
        obj["endless_r" + AXIS_NAMES[axis]] = [float((a * step) % exact(p)) for p in settings.get("periods", [])]
    if "tiling" in settings:
        if _unbuilt is None:
            build_tiling(obj)
        else:
            _unbuilt[obj.name] = obj
    # Drivers that read the residues only see them change once the object is tagged. Only then: tagged, its children
    # are evaluated again too, and an endless fractal's level objects ran their geometry nodes again mid-tick, on every
    # loop and move back (see fractal.settle())
    if settings.get("periods"):
        obj.update_tag()
