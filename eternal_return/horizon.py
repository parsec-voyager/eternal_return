"""The HUD's horizon: a line across the drone camera's view, level with what's down, two circles 45 degrees either side
of it, and dots on down and the point opposite it.

Down is the selected object (toward its origin, or its closest surface point with Horizon's Mesh, along auto scale's
Split Axes axes only), or with nothing selected, the drone's gravity. See ER_OT_Fly.get_horizon_level(). Level is
the plane through the camera square to down, and the horizon is its directions: the circle of directions 90 degrees from
down. Thinner circles at LATITUDE_STEP degrees from down, 45 either side of the horizon, work like latitude lines
around it, centered on the selected object's origin (the nadir) and on the point opposite it (the zenith). Each gets a
dot, the way to the object and the way straight away from it. Each circle is drawn where its directions cross the HUD's
own plane, a fixed distance ahead of the camera, so it lies exactly where those directions are at any pitch, roll and
lens, and stays there all the way around a planet. The horizon, a great circle, comes out straight. It's the level
line, like an aircraft's artificial horizon: from high above a planet, the planet's edge sits lower, by how far its
surface curves away.

Going around a circle, a direction's tilt toward the middle of the view rises and falls like a sine wave. So the part of
it that's in view, in front of the camera, is two arcs either side of where it comes nearest the middle, whose ends are
known exactly, and each circle is drawn as those two arcs, spread over just that part. Directions behind the camera, or
too far to the side to be in view, are left out, since they'd be projected to infinity or in front.

It's drawn by hud_draw.py's draw handler, in the drone camera's view as Blender draws it, from the view matrix of that
very draw. Drone_Camera is vertex parented to the drone's simulated geometry, so where it is and how it's turned for a
frame are only known once that frame is evaluated, after the add-on's tick has set the horizon
(ER_OT_Fly.horizon_ui()). Toward the selected object's origin, the add-on sends the point (with Split Axes' axes),
and the draw takes the direction to it from where the camera is in it. Measured in the tick instead, it was a frame
stale whenever Blender drew a frame before a tick had run since its frame step: now and then in a real flight, and
every frame in the flight harness, off by how far the direction turned in a frame, which flying fast past an object is
degrees. Toward a closest surface point (Mesh) or along gravity, it sends the direction, which doesn't change as the
camera slides along a flat face, or as the simulation turns the drone, and the draw turns it into the view by the view's
own rotation.

Until 2026-11-17 it was a geometry nodes object under the camera, its tubes made again at every depsgraph
evaluation that moved the camera: twice a frame, about 1 ms each, mostly the node graph's own overhead for a
few hundred points (2026-11-17, Tokyo). Drawn, it costs no evaluation at all, and a draw's own work is a few hundred
points of numpy. Its lines are flat, the same thickness on screen as the tubes were, instead of lit.

Everything is drawn inside the HUD's ring too, where the click aims: the click is view3d.select at the middle of the
view, which never picks what a draw handler draws.
"""
import math

import numpy as np
from mathutils import Euler, Vector

# Where the HUD is drawn, in HUD units, which hud_draw.py draws in (hud.py): 0.1 ahead is the camera's clip start. The
# ring and gimbals are 0.1 ahead, and the text is drawn over everything. The line is just behind them.
DEPTH = 0.104

# A circle every LATITUDE_STEP degrees from down: at 45, 90 (the horizon) and 135
LATITUDE_STEP = 45

# The horizon's thickness (half of it, as the tubes' radius was), three quarters of the ring's, and the other circles',
# under half the horizon's: about 9 and 4 px on a 2752 px wide view
RADIUS = 0.0006
LATITUDE_RADIUS = 0.00025

# Points along each arc. A circle is two arcs, spread over just the part in view, so a whole circle in view is 96
# straight pieces, whose chords bow at most 0.03 degrees from it.
ARC_POINTS = 49

# Directions this close to square to the view (their Z, of 1) are left out of the circles: about 81 degrees off, past
# the corners of a 10 mm lens's view (64 degrees), and closer to 90 they'd be projected out toward infinity
MIN_VIEW_Z = 0.15

# The dots on down and the point opposite it, a bit in front of the lines, behind the ring and readouts. A quarter of
# the ring's size: about 37 px across on a 2752 px wide view.
DOT_RADIUS = 0.0025
DOT_DEPTH = 0.1035

# What the next draws are level with: set_level()'s arguments, or None while hidden
_level = None


def set_level(down=None, pivot=None, pivot_axes=(1.0, 1.0, 1.0), pivot_rotation=(0.0, 0.0, 0.0)):
    """Level the horizon with down, a world space direction, or with the direction from the drone camera to pivot, a
    world space point, along pivot_axes only (1 or 0 for each of X, Y and Z, turned by the Euler pivot_rotation), like
    Split Axes. With neither, hide it."""
    global _level
    if down is None and pivot is None:
        _level = None
        return
    _level = {"down": None if down is None else Vector(down).normalized(),
              "pivot": None if pivot is None else Vector(pivot),
              "pivot_axes": Vector([float(axis) for axis in pivot_axes]),
              "pivot_rotation": Euler(pivot_rotation).to_matrix()}


def world_down(camera_location):
    """The world space direction the horizon is level with, from the drone camera at camera_location (where the draw
    has it), not normalized; None while hidden or the camera is right on the pivot"""
    if _level is None:
        return None
    down = Vector((0.0, 0.0, 0.0)) if _level["down"] is None else _level["down"].copy()
    if _level["pivot"] is not None:
        # Toward the pivot, along Pivot Axes only: turned into the axes' frame, masked, and back
        rotation = _level["pivot_rotation"]
        along = rotation.transposed() @ (_level["pivot"] - camera_location)
        down += rotation @ Vector([a * b for a, b in zip(along, _level["pivot_axes"])])
    return down if down.length > 0.0 else None


def lines(hud_down):
    """The horizon in the HUD's space (x right, y up, z ahead, in HUD units), level with hud_down (a direction
    there): its circles, each a polyline of points (an array, (n, 3)) with its thickness (radius), and its dots, each a
    center and radius, in the HUD's plane. Empty for a zero hud_down."""
    down = np.asarray(hud_down, dtype=np.float64)
    length = float(np.linalg.norm(down))
    if not length > 0.0:
        return [], []
    down = down / length
    # Two directions square to down and each other, for going around it
    helper = np.array([1.0, 0.0, 0.0]) if abs(down[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    across = np.cross(down, helper)
    across /= np.linalg.norm(across)
    across_too = np.cross(down, across)

    curves = []
    u = np.linspace(-1.0, 1.0, 2 * ARC_POINTS - 1)
    for step in range(1, round(180 / LATITUDE_STEP)):
        latitude = math.radians(step * LATITUDE_STEP)
        cos_latitude, sin_latitude = math.cos(latitude), math.sin(latitude)
        # Around the circle, a direction's Z (toward the middle of the view) is middle + swing * cos(angle - widest):
        # it's in view from MIN_VIEW_Z up, which bounds the angle to two arcs, from widest either way to where it
        # leaves the view
        middle = cos_latitude * down[2]
        swing = sin_latitude * math.hypot(across[2], across_too[2])
        widest = math.atan2(across_too[2], across[2])
        sweep = math.acos(min(max((MIN_VIEW_Z - middle) / max(swing, 1e-6), -1.0), 1.0))
        # Nothing of it in view
        if sweep < 1e-5:
            continue
        angle = widest + u * sweep
        directions = (cos_latitude * down + sin_latitude * (np.cos(angle)[:, None] * across
                                                              + np.sin(angle)[:, None] * across_too))
        # Carried out to the HUD plane
        points = directions * (DEPTH / directions[:, 2:3])
        curves.append((points, RADIUS if 2 * step * LATITUDE_STEP == 180 else LATITUDE_RADIUS))

    dots = []
    for direction in (down, -down):
        if direction[2] > MIN_VIEW_Z:
            dots.append((direction * (DOT_DEPTH / direction[2]), DOT_RADIUS))
    return curves, dots
