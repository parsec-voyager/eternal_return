"""How far and how fast the drone really flies, and how big it really is, for the HUD's readouts (readouts.py).

Much of what moves the drone in Blender's coordinates doesn't move it in the world it flies through. A wrap or an
endless map's recenter moves it back by a whole range, a step or a lattice vector, to the same place in the world
again. Keep Drone at Origin moves the whole scene under it, the drone with it. A loop scales the drone up or down by the
loop factor, about the point the world repeats around, to the same view at a scale of the world's that many times
finer or coarser. The old readouts measured Blender's coordinates and the drone's Blender scale: each of these showed as
a burst of speed (a wrap of 1458 m in a frame is 315,000 km/h), and the scale started over at every loop. These are
jumps, not flight: an odometer leaves them out (jump()).

A loop also relabels the world. Its factor's worth of finer world is now the size the coarser one was in Blender's
coordinates, so a Blender unit is the factor's worth fewer meters. The meters a Blender unit is are kept as a log2:
the scene's "er_unit_log2", which loops add to and Reset Scale clears, plus the scene's unit scale. A dive of a
thousand loops of 10 is 10^-1000 meters a unit, past any float, and in logs it's exact enough. On the Earth (a sphere
map with a planet or 3D tiles), the map knows its own size instead: its R = rho N map units are the Earth's radius
(sphere.meters_log2()), so it reads the Earth's real meters, from flight to flight, whatever the drone's scale.

Everything else that moves the camera is flight, and counts: the simulation's, orbit and zoom, and being held at a
surface or gliding along the 3D tiles' floor. The path is added up between jumps, each stretch in the meters a unit
was then, and read off every PERIOD frames, over the frames that took, in the simulation's time: speed in meters a
second. Scale Speed is the true scale's change over the same frames.
"""
import math

import numpy as np

# Frames the readouts are measured over: 10 times a second at 60 fps, as often as they're written anew
PERIOD = 6

UNIT_KEY = "er_unit_log2"


def scene_unit_log2(scene):
    """log2 of the meters a Blender unit is in the scene's world, after its loops (see the module's docstring)"""
    unit_scale = scene.unit_settings.scale_length
    return float(scene.get(UNIT_KEY, 0.0)) + (math.log2(unit_scale) if unit_scale > 0.0 else 0.0)


def relabel(scene, factor):
    """A loop scaled the drone by factor: a Blender unit is now factor times fewer meters"""
    if factor > 0.0 and factor != 1.0:
        scene[UNIT_KEY] = float(scene.get(UNIT_KEY, 0.0)) - math.log2(factor)


def reset(scene):
    """The world's meters as they were before any loop: Reset Scale's scale 1 is a scale of 1 again"""
    if UNIT_KEY in scene:
        del scene[UNIT_KEY]


class Odometer:
    """The path the drone flies, jumps left out, and its readings every PERIOD frames. Locations are the drone
    camera's, in Blender's world space; units, the meters a Blender unit is there, as log2."""

    def __init__(self, location, unit, scale, frame):
        self._location = np.array(location, dtype=np.float64)
        # The drone camera's world scale (Blender units), which the tick keeps up to date
        self.size = 0.0
        self._unit = unit
        self._depth = 0
        self._start(unit, scale, frame)

    def _start(self, unit, scale, frame):
        # The path since the window started, in meters times 2^-ref, ref the unit then: the units of a window are a few
        # loops apart at most, so it never leaves float range, whatever the depth
        self._ref = unit
        self._path = 0.0
        self._start_scale = scale
        self._start_frame = frame

    def _add(self, location):
        location = np.array(location, dtype=np.float64)
        distance = float(np.linalg.norm(location - self._location))
        # Within a couple of float32 steps of the last place (or of the drone's size, near the world origin), it's
        # rounding, not flight: a drone at rest read as 12 fm/s. The place isn't moved on, so a drone creeping slower
        # than that a tick still adds up to its moves.
        magnitude = max(np.abs(location).max(), np.abs(self._location).max(), self.size)
        rounding = 2.0 * float(np.spacing(np.float32(magnitude)))
        if distance <= rounding:
            return
        self._path += distance * 2.0 ** (self._unit - self._ref)
        self._location = location

    def leave(self, location):
        """A jump starts with the camera at location: the flight up to here counts. Jumps inside jumps (a wrap's
        remeasure, a loop's recenter) are one jump."""
        if self._depth == 0:
            self._add(location)
        self._depth += 1

    def arrive(self, location, unit):
        """The jump ended with the camera at location, a Blender unit now 2^unit meters"""
        self._depth -= 1
        if self._depth == 0:
            self._location = np.array(location, dtype=np.float64)
            self._unit = unit

    def read(self, location, unit, scale, frame, fps):
        """At the end of a tick, with the camera at location: None, or once PERIOD frames have gone by since the last
        reading, (log10 of the speed in meters a second, or None standing still; the true scale's change a second, as
        a log10). scale is the drone's true scale, as a log2."""
        self._add(location)
        self._unit = unit
        frames = frame - self._start_frame
        if frames < 0 or fps <= 0.0:
            # Played round from the frame range's end to its start
            self._start(unit, scale, frame)
            return None
        if frames < PERIOD:
            return None
        seconds = frames / fps
        speed = (math.log10(self._path) + self._ref * math.log10(2.0) - math.log10(seconds)
                 if self._path > 0.0 else None)
        scale_speed = (scale - self._start_scale) * math.log10(2.0) / seconds
        self._start(unit, scale, frame)
        return speed, scale_speed
