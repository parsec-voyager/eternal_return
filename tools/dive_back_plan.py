"""Flight harness plans (tools/flight_sim.py) for the keeping-up test (tools/probes/missing_view.py): the surface map in
this .blend flown in, then flown backward away from its floor at full throttle. A point of its surface (its own:
the chart's) near the file's drone camera, inside a face away from its corners, is the target, with a line in at the
mode's angle (or steeper) that keeps off the surface; the drone starts 15% of the map's size out along it (on an
endless plane, as far as the file's camera is) and flies in (throttle 0.5) until DIVE seconds, then goes full throttle
backward until DIVE + BACK. How it flies in (the pilot's ways: fast and low over the surface, turning, more than
straight down):

- dive: at 50 degrees, tracked (a perfect pilot holding the line at the target)
- glide: at 12 degrees, tracked: low over the surface, the view along it, far more of it in sight
- turns: at 12 degrees, turning left and right in turn (TURN_PERIOD seconds a sweep): the view swings across the
  surface, so what's built ahead for the view (frames' view reach, floor windows) is outrun sideways. With the camera
  tilted 90 degrees (the user's: looking along the thrust), that's the roll stick; yaw only spins the view about its
  own axis. Tracked like the others: after each tick's roll the pilot slides the drone back onto its line from the
  target along its view, so it sweeps sideways across the surface as it dives in (untracked, on round and tube maps
  it flew off the surface: 0 to 6 loops in 30 s, against a tracked dive's 20 to 70)

    Blender -b <map.blend> --factory-startup --python tools/dive_back_plan.py -- <out.json> [DIVE] [BACK] [dive|glide|turns]
"""

import importlib
import json
import math
import os
import sys
import types

import bpy
import numpy as np
from mathutils.kdtree import KDTree

ANGLES = {"dive": 50.0, "glide": 12.0, "turns": 12.0}
TURN_PERIOD = 2.0
# How far each sweep turns the drone (degrees): the roll stick's peak is worked out from the file's roll rate at the
# stick's center (scene.stick_rates, Actual rates, nearly linear there; the add-on's default 145 degrees a second if
# the file has none). At a fixed 0.5 it turned 73 degrees a sweep, and even tracked the view swung off the target and
# the drone with it (0 to 2 loops in 30 s)
TURN_SWEEP = 30.0
CENTER_RATE = 145.0
GRID = 9

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
path = args[0]
dive = float(args[1]) if len(args) > 1 else 18.0
back = float(args[2]) if len(args) > 2 else 12.0
mode = args[3] if len(args) > 3 else "dive"
if mode not in ANGLES:
    raise SystemExit(f"dive_back_plan: {mode}: one of {', '.join(ANGLES)}")
obj = next(o for o in bpy.data.objects if o.get("surface_source") is not None)
source = obj["surface_source"]
matrix = np.array(source.matrix_world)
mesh = source.data
camera = np.array(bpy.data.objects["Drone_Camera"].matrix_world.translation)
# The map's own surface (its chart's, from the repo's add-on): a smooth map's limit surface lies inside its mesh, most at
# rounded edges (a text's thin letters: 3% of its size), and lines aimed at the mesh's faces passed over the surface
# and flew on out of the map, the drone growing forever
package = types.ModuleType("er_plan")
package.__path__ = [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eternal_return")]
sys.modules["er_plan"] = package
chart = importlib.import_module("er_plan.surface").topology_of(obj).chart


def world(p):
    return (matrix @ np.c_[p, np.ones(len(p))].T)[:3].T


# The surface sampled: a GRID x GRID grid of every face's (parameters), for picking the target and the clear-way check
k = np.linspace(0.0, 1.0, GRID)
gu, gv = (a.ravel() for a in np.meshgrid(k, k))
faces, us, vs = [], [], []
for poly in mesh.polygons:
    keep = (gu + gv <= 1.0) if len(poly.vertices) == 3 else np.ones(len(gu), bool)
    faces.append(np.full(int(keep.sum()), poly.index))
    us.append(gu[keep])
    vs.append(gv[keep])
faces, us, vs = np.concatenate(faces), np.concatenate(us), np.concatenate(vs)
samples = world(chart.points(faces, us, vs))
size = float(np.ptp(samples, axis=0).max())
# How far out the drone starts: 15% of the map's size. On an endless plane (one face, wrapped) the face says nothing of
# that: 15% of it started the drone 0.2 units out at full scale, which auto scale took as a plunge (482 loops in 30 s,
# 77% of the view missing, against 25 loops on the plane's own plan). There, as far as the file's camera is.
START_SHARE = 0.15
reach = (START_SHARE * size if len(mesh.polygons) > 1
         else float(np.linalg.norm(camera - samples[np.argmin(np.linalg.norm(samples - camera, axis=1))])))
tree = KDTree(len(samples))
for n, p in enumerate(samples.tolist()):
    tree.insert(p, n)
tree.balance()
spacing = float(np.median([np.linalg.norm(samples[n + 1] - samples[n]) for n in range(0, len(samples) - 1, 97)]))

# The target: a sample nearest the camera, in a face's middle part (away from its corners' posts), its normal the
# surface's there (from its neighbors in the grid), toward the camera
middle = (us > 0.2) & (us < 0.8) & (vs > 0.2) & (vs < 0.8) & (us + vs < 1.6)
order = np.flatnonzero(middle)[np.argsort(np.linalg.norm(samples[middle] - camera, axis=1))]


def surface_normal(n):
    f, u, v = int(faces[n]), float(us[n]), float(vs[n])
    du = world(chart.points(np.array([f, f]), np.array([u - 0.01, u + 0.01]), np.array([v, v])))
    dv = world(chart.points(np.array([f, f]), np.array([u, u]), np.array([v - 0.01, v + 0.01])))
    normal = np.cross(du[1] - du[0], dv[1] - dv[0])
    normal /= np.linalg.norm(normal)
    return normal if normal @ (camera - samples[n]) > 0 else -normal


def clear(start, target, normal):
    """Whether the line from start to target keeps off the surface till it meets it: each point of it at least half
    as far from the surface as it is above the target's tangent plane (a surface bulging up into the way stops it),
    the samples' spacing allowed for"""
    for t in np.linspace(0.0, 0.95, 96):
        p = start + t * (target - start)
        above = float((p - target) @ normal)
        if tree.find(tuple(p))[2] + spacing < 0.5 * above:
            return False
    return True


start = None
for n in order[:24].tolist():
    target, normal = samples[n], surface_normal(n)
    along = np.cross(normal, (0, 0, 1)) if abs(normal[2]) < 0.9 else np.cross(normal, (1, 0, 0))
    along /= np.linalg.norm(along)
    side = np.cross(normal, along)
    for degrees in (ANGLES[mode], 20.0, 30.0, 50.0, 70.0):
        if degrees < ANGLES[mode]:
            continue
        angle = np.radians(degrees)
        for h in range(16):
            heading = np.cos(2 * np.pi * h / 16) * along + np.sin(2 * np.pi * h / 16) * side
            place = target + reach * (np.cos(angle) * heading + np.sin(angle) * normal)
            if clear(place, target, normal):
                start = place
                break
        if start is not None:
            break
    if start is not None and degrees == ANGLES[mode]:
        break
if start is None:
    raise SystemExit(f"dive_back_plan: no clear way in near {obj.name}'s points by the camera")
sticks = [{"t": 0.0, "throttle": 0.0, "click": 0}, {"t": 0.02, "click": 1}, {"t": 0.5, "throttle": 0.0},
          {"t": 1.5, "throttle": 0.5}, {"t": dive, "throttle": 0.5}, {"t": dive + 0.5, "throttle": -1.0}]
if mode == "turns":
    rates = bpy.context.scene.get("stick_rates")
    center = math.degrees(rates.get("roll_center_sensitivity", math.radians(CENTER_RATE))) if rates else CENTER_RATE
    # (A ramp up to roll and back down over TURN_PERIOD turns center x roll x TURN_PERIOD / 2)
    roll = min(1.0, 2.0 * TURN_SWEEP / (center * TURN_PERIOD))
    t, side = 1.5, 1.0
    while t < dive + back:
        sticks.append({"t": round(t + TURN_PERIOD / 2, 3), "roll": side * roll})
        sticks.append({"t": round(t + TURN_PERIOD, 3), "roll": 0.0})
        t, side = t + TURN_PERIOD, -side
    sticks.sort(key=lambda s: s["t"])
plan = {"start": {"location": [round(float(x), 4) for x in start], "look_at": [round(float(x), 4) for x in target],
                  "click_select": obj.name, "track": True},
        "duration": dive + back, "sticks": sticks}
with open(path, "w") as f:
    json.dump(plan, f)
print(f"dive_back_plan: {obj.name}, {mode} at {degrees:.0f} degrees, in {dive} s then back {back} s -> {path}")
