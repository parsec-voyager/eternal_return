"""Fly the Eternal Return drone headlessly, with a virtual controller in place of the RadioMaster or Xbox pad, through the
add-on's own flight code, and record what the drone camera sees.

The add-on's ER_OT_Fly.tick() runs every 0.01 s of simulated time, as its modal timer does, and the timeline steps
one frame every 1 / fps, as playback does, so the drone's geometry nodes simulation (the physics) and the frame change
handler that feeds it the sticks run as in a real flight. Loop, wrap, auto scale, Keep Drone at Origin, braking, clicking
and scaling configs are all the add-on's own code. The harness only replaces what needs a window or hardware:

- SDL: controls.read_axes returns the virtual controller's axes, built from the plan
  through the file's own axis mapping and invert settings, so deadzones, rates, expo and throttle power all apply.
- The clock: time.perf_counter, seen by tick(), is the simulated time.
- Clicking: click_center_of_viewport ray casts from the camera's center instead of selecting in a 3D view.
- Sound: off.

Usage:
    Blender -b <map.blend> --python tools/flight_sim.py -- <plan.json> [--out DIR] [--every N] [--size WxH]
    Blender -b <map.blend> --python tools/flight_sim.py -- --replay <recorded camera object> [same options]
        [--addon installed|working] [--video] [--engine workbench|scene] [--seed N] [--realtime] [--timing]
        [--probe SCRIPT]   (probes in tools/probes/)

Nothing is saved to the .blend. Output in DIR (default: a "flight_<plan>" folder next to the plan): frames/NNNNN.png
(every Nth frame), log.csv (one row per rendered frame), events.txt (loops, wraps, origin moves, clicks, jump cuts,
being inside geometry), and flight.mp4 with --video. A jump cut is a frame that changed far more than those around it.
Every frame is also re-rendered with the map as it was a rendered frame earlier: a POP is the map itself changing
the view (a level switching, a patch moving) by more than POP_MIN, however fast the camera moves, and a jump's "the map
did N%" says how much of it was the map rather than the camera (moving fast, passing close to something). Each loop
and wrap is rendered just before and after, at the same instant: a SEAM is one that changes the view. The TIMING
line at the end is the main thread's time: the start (until the first frame), and each frame's ticks and frame step;
with --timing (no seam renders) and --realtime, what a real flight spends.

The plan is JSON:
{
  "start": {                          optional; without it, the drone starts where the file left it
    "location": [x, y, z],            camera location, relative to "relative_to"'s origin if given
    "look_at": [x, y, z],             a point to aim the camera at (with world Z up)
    "relative_to": "Inferno",         an object name
    "scale": 1.0,                     the drone's scale
    "clip": [0.5, 50000.0],           camera clip start and end at scale 1 (times the scale); default: keep the file's
                                      clip_end / clip_start ratio, with clip_start its Default Clip Start x scale
    "click_select": "Gyre+0",         a click selects this object, as if it had hit it, instead of ray casting
    "click_at": [x, y, z],            aim here until "turn_at" seconds (default 0.1), then turn to look_at, like
    "turn_at": 0.1,                   clicking the floor and then pitching up. The add-on only clicks on a press after
                                      it has seen the switch released, so key click 0 at t=0 and 1 just after.
    "track": true                     keep the drone on course for look_at all flight, a perfect pilot: each tick it's
                                      moved sideways back onto the line from look_at along the camera's view, which
                                      moves along with every loop, wrap and endless recenter. Diving at one point, any
                                      aiming error grows by the loop factor every loop, so held sticks miss it within a
                                      dozen. (Re-aiming the camera at it instead fought the drone's momentum and spun.)
                                      The biggest move, against the distance to look_at, is logged at the end.
  },
  "settings": {"keep_drone_at_origin": true},     optional scene.control_settings overrides
  "other_settings": {"record_camera": true, "record_obj": "RecCam"},   scene.flight_options, e.g. to record
  "object_properties": {"Habitat": {"spin_rpm": 0.0}},   custom properties to set on objects first
  "duration": 10.0,
  "start_frame": 1000,                optional; real flights start wherever the timeline was left (default 1000)
  "continue": true,                   optional: on the core, go on from the drone's state in the file (position,
                                      orientation, velocity), as real flights do, instead of starting at rest
  "sticks": [                         keyframes: each control moves linearly to its value at "t", and holds after
    {"t": 0.0, "throttle": 0.0},
    {"t": 0.3, "click": 1}, {"t": 0.4, "click": 0},
    {"t": 1.0, "throttle": 0.6, "pitch": 0.2}
  ]
}
Stick values are what the add-on sees after its invert settings. throttle (in 3D mode), yaw, pitch, roll, scale, zoom,
camera and lens_clip are centered: -1 to 1, 0 at rest. click, orbit, brake and throttle outside 3D mode go from 0
(released) to 1 (fully on).
"""

import argparse
import csv
import importlib
import json
import math
import os
import random
import sys
import time
import types
from fractions import Fraction

import bpy
import numpy as np
from mathutils import Matrix, Vector

TICK = 0.01                      # the add-on's modal timer interval
CENTERED = {"yaw", "pitch", "roll", "scale", "zoom", "camera", "lens_clip"}
UNIDIRECTIONAL = {"throttle", "orbit", "click", "brake"}
CONTROLS = ["throttle", "yaw", "pitch", "roll", "scale", "zoom", "orbit", "click", "camera", "lens_clip", "brake"]
# A rendered frame counts as a jump cut when this much more of it changes than in the frames around it (JUMP_WINDOW
# rendered frames each side), and at least JUMP_MIN percentage points more
JUMP_RATIO, JUMP_MIN, JUMP_WINDOW = 2.5, 15.0, 10
# A POP is the map alone changing this much of the view between rendered frames (a level appearing or vanishing, a
# patch moving to where it doesn't match), and a SEAM a loop or wrap changing this much of it at the instant it happens.
# Both are measured with the camera held still, so they don't depend on how fast the drone flies.
# How far, in multiples of the drone's scale, track (start.track) keeps steering the drone at its point (see main())
TRACK_LIMIT = 10000.0
POP_MIN, SEAM_MIN = 1.5, 1.5
# Directions to test whether the camera is inside a closed mesh: the faces around it are seen from behind
ENCLOSURE_DIRECTIONS = [Vector(d).normalized() for d in
                        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] +
                        [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]]


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="flight_sim.py")
    parser.add_argument("plan", nargs="?", help="the flight plan (JSON); not used with --replay")
    parser.add_argument("--out")
    parser.add_argument("--every", type=int, default=1, help="render every Nth frame")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed Python's random numbers, so a map's random choices (like an endless map's made-up "
                             "digits) are the same every run")
    parser.add_argument("--size", default="640x360")
    parser.add_argument("--addon", choices=("installed", "working"), default="installed",
                        help="fly the installed extension (what Blender loads) or the repo's working copy")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--timing", action="store_true",
                        help="no renders at loops and wraps (seams), so a loop's frame shows the add-on's own time: "
                             "with --realtime, the TIMING line is what a real flight's main thread spends")
    parser.add_argument("--realtime", action="store_true",
                        help="keep simulated time no faster than the wall clock, and hand builds to the add-on's worker "
                             "threads as a real flight does (background.py keeps them off in a background Blender): "
                             "they get as much time a frame as in a real flight (their schedulers read the real clock)")
    parser.add_argument("--probe", metavar="SCRIPT",
                        help="a Python file whose frame(t, frame, m, scene) runs after every frame and whose "
                             "report(out) runs at the end, e.g. to count what a map hasn't built yet; setup(m), if "
                             "it has one, runs before the first tick (tools/probes/)")
    parser.add_argument("--save-as", metavar="PATH", help="save the .blend after the flight, e.g. with a recording")
    parser.add_argument("--replay", metavar="OBJECT",
                        help="instead of flying a plan, replay a flight recorded with the add-on's Record Camera into "
                             "OBJECT: render each recorded frame from it, with the same checks")
    parser.add_argument("--engine", choices=("workbench", "scene"), default="workbench",
                        help="render like Solid mode (Workbench), or with the scene's engine, like the Rendered viewport "
                             "(EEVEE, with its materials, e.g. fog)")
    return parser.parse_args(argv)


def load_addon(which):
    """The add-on's module that flies: flight.py, which holds the flight's code and state.
    The installed one is already registered; the working copy is imported beside it and uses the installed one's
    registered properties, so property changes need a reinstall first."""
    if which == "installed":
        return importlib.import_module("bl_ext.user_default.eternal_return.flight")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    return importlib.import_module("eternal_return.flight")


def fly_operator(m):
    """The add-on's Fly operator class (operators.py), made from flight.FlightFeatures"""
    return importlib.import_module(m.__package__ + ".operators").ER_OT_Fly


class VirtualController:
    """Stick positions over time from the plan's keyframes, turned into raw axis values the way the file's axis
    mapping and invert settings would read them."""

    def __init__(self, keyframes, scene):
        self.scene = scene
        self.tracks = {c: [] for c in CONTROLS}
        for key in sorted(keyframes, key=lambda k: k["t"]):
            for control, value in key.items():
                if control != "t":
                    if control not in self.tracks:
                        raise ValueError(f"unknown control {control!r}")
                    self.tracks[control].append((key["t"], float(value)))
        mapping = scene.control_axes
        self.axis_of = {c: getattr(mapping, c) for c in CONTROLS}
        for control, track in self.tracks.items():
            if track and self.axis_of[control] == 'none':
                raise ValueError(f"{control} isn't mapped to an axis in this file, so it can't be moved")
        self.num_axes = max([8] + [int(a) + 1 for a in self.axis_of.values() if a != 'none'])
        self.values = {c: 0.0 for c in CONTROLS}

    def value_at(self, control, t):
        track = self.tracks[control]
        if not track:
            return 0.0
        if t <= track[0][0]:
            return track[0][1] if track[0][0] <= 0.0 else 0.0
        for (t0, v0), (t1, v1) in zip(track, track[1:]):
            if t0 <= t < t1:
                # A key at the same value holds; one after a different value ramps
                return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
        return track[-1][1]

    def axes(self, t):
        cp = self.scene.control_settings
        axes = [0.0] * self.num_axes
        for control in CONTROLS:
            value = self.value_at(control, t)
            self.values[control] = value
            axis = self.axis_of[control]
            if axis == 'none':
                continue
            raw = value
            if control in UNIDIRECTIONAL and not (control == "throttle" and cp.flight_mode == "3D"):
                if not getattr(cp, f"{control}_centered"):
                    raw = 2.0 * value - 1.0          # the add-on maps -1..1 to 0..1
            if getattr(cp, f"{control}_invert"):
                raw = -raw
            axes[int(axis)] = max(-1.0, min(1.0, raw))
        return axes


def aim(m, origin, location, target):
    drone, camera = m.drone_obj, m.drone_camera
    in_drone = (camera.matrix_parent_inverse @ camera.matrix_basis).to_3x3().normalized()
    rotation = (origin @ Vector(target) - origin @ Vector(location)).to_track_quat('-Z', 'Y').to_matrix()
    drone.rotation_euler = (rotation @ in_drone.inverted()).to_euler('XYZ')
    bpy.context.view_layer.update()


def exact_points(start):
    """A plan's points may be exact, each coordinate a string like "-585/8" or "1/3": a point of an endless fractal,
    which a float can't hold (a ternary digit isn't a binary fraction), for "track" to dive at forever. Kept exact (as
    <key>_exact) for tracking, and as floats for everything else. A number is taken as the decimal it's written as, like
    endless.exact()."""
    for key in ("location", "click_at", "look_at"):
        if key in start and any(isinstance(c, str) for c in start[key]):
            start[key + "_exact"] = [Fraction(c) if isinstance(c, str) else Fraction(repr(float(c))) for c in start[key]]
            start[key] = [float(v) for v in start[key + "_exact"]]


def start_origin(start):
    if start.get("relative_to"):
        return bpy.context.scene.objects[start["relative_to"]].matrix_world.copy()
    return Matrix.Identity(4)


def viewport_shading(scene):
    """Workbench renders with the scene's display shading, but the user flies in a 3D view with its own: copy the file's
    biggest Solid 3D view's shading onto the scene's, so the frames show what the user sees. The substitution maps
    color their bars by vertex color in the view only: rendered with the scene's (material gray) the strapwork was
    invisible, and every pop and seam check on those maps was blind to it (2026-11-22)."""
    views = [(area.width * area.height, space) for screen in bpy.data.screens for area in screen.areas
             if area.type == 'VIEW_3D' for space in area.spaces[:1] if space.shading.type == 'SOLID']
    if not views:
        return None
    space = max(views, key=lambda v: v[0])[1]
    source, target = space.shading, scene.display.shading
    for prop in source.bl_rna.properties:
        if prop.is_readonly or prop.type in ('POINTER', 'COLLECTION') or prop.identifier in ("type", "rna_type"):
            continue
        try:
            setattr(target, prop.identifier, getattr(source, prop.identifier))
        except (AttributeError, TypeError, ValueError):
            pass
    return source.color_type


def place_drone(m, start):
    """Put the drone camera at the plan's start: location, aim, scale and clip."""
    scene = bpy.context.scene
    drone, camera = m.drone_obj, m.drone_camera
    origin = start_origin(start)
    scale = float(start.get("scale", drone.matrix_world.to_scale().z))
    ratio = camera.data.clip_end / camera.data.clip_start
    default = scene.control_settings.default_clip_start
    clip = start.get("clip") or [default, default * ratio]
    drone.scale = (scale, scale, scale)
    camera.data.clip_start, camera.data.clip_end = clip[0] * scale, clip[1] * scale
    bpy.context.view_layer.update()
    if "location" in start:
        target = start.get("click_at") or start.get("look_at")
        if target:
            aim(m, origin, start["location"], target)
        drone.location += origin @ Vector(start["location"]) - camera.matrix_world.translation
        bpy.context.view_layer.update()


def ray_cast_select(m, named=None):
    """Stands in for the add-on's click_center_of_viewport: select the first object the camera's center ray hits within
    its clip range, skipping the drone and anything unselectable, as clicking the center of the 3D view would. With
    named, select that object, as if the click had hit it (for open maps where an aimed ray slips through the gaps)."""

    def click_center_of_viewport(self, context):
        if named:
            bpy.ops.object.select_all(action='DESELECT')
            context.scene.objects[named].select_set(True)
            context.view_layer.objects.active = context.scene.objects[named]
            return
        camera = m.drone_camera
        depsgraph = context.evaluated_depsgraph_get()
        direction = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        start = camera.matrix_world.translation + direction * camera.data.clip_start
        remaining = camera.data.clip_end - camera.data.clip_start
        while remaining > 0.0:
            hit, location, _normal, _index, obj, _matrix = context.scene.ray_cast(depsgraph, start, direction,
                                                                                  distance=remaining)
            if not hit:
                print("CLICK hit nothing")
                return
            obj = obj.original
            # An instance's object outside the view layer (a part of a parts map, parts.py) can't be selected: the ray
            # goes on through it
            if obj.name in context.view_layer.objects and not (self.is_drone_part(obj) or obj.hide_select):
                bpy.ops.object.select_all(action='DESELECT')
                obj.select_set(True)
                context.view_layer.objects.active = obj
                return
            step = (location - start).length + 1e-6 * camera.data.clip_end
            start, remaining = start + direction * step, remaining - step

    return click_center_of_viewport


def make_operator(m, events, clock, snap, click_select=None, moved=None, out=None, snap_world=None):
    """A plain class with ER_OT_Fly's methods (a registered operator can't be made outside an operator call),
    wrapped to log loops and wraps, and to measure their seams: the view rendered (by snap(), from the drone camera)
    just before and just after, at the same instant. moved(scale, before, after), if given, is told how each loop, wrap
    and recenter moved the drone camera (scaling by scale, from world location before to after)."""
    def tell(scale, before, **exact):
        if moved:
            moved(scale, before, m.drone_camera.matrix_world.translation.copy(), **exact)
    raw_snap = snap

    def snap():
        """The view as the add-on leaves it at the end of the tick: an endless fractal's level objects and eye are only
        put in place then (its fractal.settle()), and a sphere or surface map's floor and 3D parts (surface.settle(),
        parts.settle(), 2026-11-21: after a loop they're still the size they were, the whole torus 8 times too small),
        so a render just after a loop, mid-tick, has to do it first"""
        if hasattr(getattr(m, "fractal", None), "settle"):
            bpy.context.view_layer.update()
            for obj in bpy.context.scene.objects:
                if m.fractal.is_fractal(obj):
                    m.fractal.settle(obj, m.drone_camera.matrix_world)
                    m.fractal.follow(obj, m.drone_camera.matrix_world)
        if hasattr(m, "surface") and hasattr(m, "sphere"):
            bpy.context.view_layer.update()
            camera = m.drone_camera.matrix_world
            for obj in m.sphere.sphere_objects(bpy.context.scene) + m.surface.surface_objects(bpy.context.scene):
                m.surface.settle(obj, camera.translation)
                if m.parts.is_parts(obj):
                    m.parts.settle(obj, camera.translation, view=camera.to_3x3())
            bpy.context.view_layer.update()
        return raw_snap()

    def anchors(op):
        """The sphere and surface maps' anchors, which a recenter moves (their state text)"""
        if not hasattr(op, "sphere_maps"):
            return {}
        return {obj.name: obj.get(m.anchored_map(obj).STATE) for obj in op.sphere_maps()}

    # The operator's own methods and its mixins' (flight.FlightFeatures), not bpy's
    methods = {}
    for cls in reversed(fly_operator(m).__mro__):
        if cls.__module__.startswith(m.__package__):
            methods.update({k: v for k, v in cls.__dict__.items() if not k.startswith("__")})
    Fly = type("SimulatedFly", (), methods)
    Fly.report = lambda self, kind, message: events.append((clock.t, "REPORT", f"{kind} {message}"))
    Fly.click_center_of_viewport = ray_cast_select(m, click_select)

    original_loop_scale, original_wrap = Fly.loop_scale, Fly.wrap

    def seam(kind, text, before, after):
        change = block_change(before, after)
        events.append((clock.t, kind, f"{text}, seam {change:.1f}%"))
        if change > SEAM_MIN:
            events.append((clock.t, "SEAM", f"the {kind.lower()} changed {change:.1f}% of the view"))
            if out:
                for name, pixels in (("before", before), ("after", after)):
                    save_pixels(pixels, os.path.join(out, f"seam_{clock.t:07.3f}_{name}.png"))

    def loop_scale(self, loop_factor):
        bpy.context.view_layer.update()
        # SAVE_LOOP_STATES=1: the file just before each loop, kept as seam_<t>_state.blend when the loop seams
        if out and os.environ.get("SAVE_LOOP_STATES"):
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "loop_state.blend"), copy=True)
        before, camera = snap(), m.drone_camera.matrix_world.translation.copy()
        endless_map = m.selected_obj if getattr(m, "endless", None) and m.endless.is_endless(m.selected_obj) else None
        anchor = m.endless.get_anchor(endless_map) if endless_map else None
        result = original_loop_scale(self, loop_factor)
        bpy.context.view_layer.update()
        tell(loop_factor, camera, anchor_before=anchor)
        text = f"x{loop_factor:g}"
        if endless_map:
            text += f" (anchor {endless_map[m.endless.ANCHOR]}, horizon {endless_map.get(m.endless.HORIZON)})"
        seam("LOOP", text, before, snap())
        if out and os.environ.get("SAVE_LOOP_STATES") and events[-1][1] == "SEAM":
            os.replace(os.path.join(out, "loop_state.blend"), os.path.join(out, f"seam_{clock.t:07.3f}_state.blend"))
        return result

    def wrap(self):
        # Up to date first: a camera matrix left stale by something earlier in the tick read as a jump
        bpy.context.view_layer.update()
        start = m.drone_camera.matrix_world.translation.copy()
        result = original_wrap(self)
        bpy.context.view_layer.update()
        jump = m.drone_camera.matrix_world.translation - start
        if jump.length > 0.0:
            tell(1.0, start)
        if jump.length > 10 * m.drone_camera.data.clip_start:
            # wrap() moves only the Drone object, so moving it back shows the view from just before
            after = snap()
            m.drone_obj.matrix_world = Matrix.Translation(-jump) @ m.drone_obj.matrix_world
            bpy.context.view_layer.update()
            before = snap()
            m.drone_obj.matrix_world = Matrix.Translation(jump) @ m.drone_obj.matrix_world
            bpy.context.view_layer.update()
            seam("WRAP", f"by {tuple(round(c, 3) for c in jump)}", before, after)
        return result

    Fly.loop_scale, Fly.wrap = loop_scale, wrap

    # An endless map moves the drone back toward its origin and its anchor with it, a wrap of its own: its seam is
    # measured the same way, only when it happens, since snap() renders
    if hasattr(Fly, "recenter_endless"):
        original_recenter = Fly.recenter_endless

        def recenter_endless(self):
            bpy.context.view_layer.update()
            camera = m.drone_camera.matrix_world.translation
            steps = [m.endless.recenter_steps(obj, camera) for obj in self.endless_maps()]
            if not any(any(s) for s in steps):
                # A sphere or surface map moves its anchor under the drone and the drone back by as much: a recenter
                # can't be foreseen cheaply, so the view before it is the map as it was, moved with the drone (the
                # pop check had each one as a pop of the whole map). Measured as a seam, like the others.
                held = anchors(self)
                if not held or snap_world is None:
                    return original_recenter(self)
                old_world, start = world_snapshot(m, bpy.context.scene), camera.copy()
                result = original_recenter(self)
                if anchors(self) == held:
                    return result
                bpy.context.view_layer.update()
                shift = m.drone_camera.matrix_world.translation - start
                before = snap_world((({name: Matrix.Translation(shift) @ matrix
                                       for name, matrix in old_world[0].items()}), *old_world[1:]))
                tell(1.0, start)
                seam("WRAP", f"anchor recenter by {tuple(round(c, 3) for c in shift)}", before, snap())
                return result
            before, start = snap(), camera.copy()
            result = original_recenter(self)
            bpy.context.view_layer.update()
            tell(1.0, start, steps=next(s for s in steps if any(s)))
            seam("WRAP", f"endless recenter by {steps} steps", before, snap())
            return result

        Fly.recenter_endless = recenter_endless

    return Fly()


def frame_step_handler(m):
    """The add-on's frame change handler that flies each frame"""
    return m.fly_frame


def start_flight(m, op, clock, keep_state=False):
    """What ER_OT_Fly.execute() does (with flight.FlightFeatures.prepare_flight() and start_flight()), minus SDL,
    sound, playback and the timer. The drone starts at rest, upright at the Drone object's origin, unless keep_state
    (the plan's "continue"): then it goes on from where the file left it, as real flights do."""
    context = bpy.context
    op.get_drone_objects(context)
    op.reset_objects()
    if getattr(m, "endless", None) and hasattr(m.endless, "prepare"):
        for obj in context.scene.objects:
            if m.endless.is_endless(obj):
                m.endless.prepare(obj, m.drone_camera.matrix_world.translation)
            if hasattr(m.fractal, "settle") and m.fractal.is_fractal(obj):
                m.fractal.settle(obj, m.drone_camera.matrix_world)
                m.fractal.follow(obj, m.drone_camera.matrix_world)
    if m.start_physics(context) is False:
        raise RuntimeError("the add-on can't fly: no Eternal Return core")
    if m.physics.is_flying() and not keep_state:
        m.physics.reset(m.drone_obj)
    context.view_layer.update()
    op._controller = object()
    op._gamepad = None
    op._sound = types.SimpleNamespace(update=lambda *a: None, stop=lambda: None)
    context.scene.sync_mode = 'NONE'
    # As FlightFeatures.start_flight() does: the garbage collector leaves what's long-lived alone while flying
    background = importlib.import_module(m.__package__ + ".background")
    if hasattr(background, "hold_heap"):
        background.hold_heap(True)
    frame_handler = frame_step_handler(m)
    if frame_handler not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(frame_handler)
    # The add-on's own handlers, the flown copy's only: endless fractals' eyes follow the camera once a frame
    if hasattr(getattr(m, "fractal", None), "follow_eyes"):
        for handler in [h for h in bpy.app.handlers.frame_change_pre if getattr(h, "__name__", "") == "follow_eyes"]:
            bpy.app.handlers.frame_change_pre.remove(handler)
        bpy.app.handlers.frame_change_pre.append(m.fractal.follow_eyes)
    context.window_manager.er_fly_modal_running = True
    op._last_time = clock.t
    # The flight's path for the HUD's readouts, from where the drone starts (odometer.py)
    if hasattr(m, "odometer"):
        context.view_layer.update()
        op._earths = [obj for obj in m.sphere.sphere_objects(context.scene) if m.sphere.meters_log2(obj) is not None]
        camera_matrix = m.get_camera_matrix()
        unit = op.get_unit_log2()
        op._odometer = m.odometer.Odometer(camera_matrix.translation, unit,
                                           math.log2(camera_matrix.to_scale().z) + unit, context.scene.frame_current)


def readout_texts(m):
    """The HUD's readouts as the add-on last wrote them (hud.py), by name"""
    return dict(m.hud.lines)


def camera_state(m, scene):
    camera = m.drone_camera
    offset = Vector(scene.drone_rig.world_offset)
    return {
        "location": camera.matrix_world.translation + offset,      # where it is in the map, undoing origin moves
        "scale": m.drone_obj.matrix_world.to_scale().z,
        "clip_end": camera.data.clip_end,
        "selected": (m.selected_obj.name if m.selected_obj_bool else ""),
    }


def enclosure(m, scene, depsgraph):
    """The fraction of directions in which the nearest face around the camera is seen from behind, within 1000 clip
    starts: near 1 inside a closed mesh, like a pillar."""
    camera = m.drone_camera
    origin = camera.matrix_world.translation
    reach = 1000 * camera.data.clip_start
    behind = 0
    for direction in ENCLOSURE_DIRECTIONS:
        hit, _loc, normal, _i, obj, _mat = scene.ray_cast(depsgraph, origin, direction, distance=reach)
        if hit and not fly_operator(m).is_drone_part(None, obj.original) and normal.dot(direction) > 0.0:
            behind += 1
    return behind / len(ENCLOSURE_DIRECTIONS)


def mesh_token(obj):
    """What tells a mesh written again from the one before: its vertex count and first vertex (a rebuild writes its
    points from its new middle)"""
    data = obj.data if obj.type == 'MESH' else None
    if data is None:
        return None
    return len(data.vertices), tuple(data.vertices[0].co) if len(data.vertices) else ()


def world_snapshot(m, scene, exclude=()):
    """Everything about the map that can change during a flight, in the map's own coordinates (undoing Keep Drone at
    Origin): each object's matrix_world, and the float inputs of its geometry nodes modifiers. A layer a map pushes
    away from the drone camera along its rays (custom property "camera_push_scale": its scale without the push, like
    the strip floor's layers) moves with the camera without changing the view, so it's kept unpushed, with its push.
    And each mesh's mesh_token(): a mesh written again since can't be put back, so neither is its matrix
    (render_with_world())."""
    offset = Vector(scene.drone_rig.world_offset)
    drone_parts = {m.drone_obj, m.drone_camera, m.clip_end_obj, *exclude}
    matrices, inputs, pushes, tokens = {}, {}, {}, {}
    eye = offset + m.drone_camera.matrix_world.translation
    for obj in scene.objects:
        if obj in drone_parts:
            continue
        # An endless fractal's level objects (fractal.py) are rebuilt with new points as the drone flies, and moved to
        # match: their old matrix with their new points was a jumble, every rebuild a fake pop. What they draw follows
        # the map's eye, which is put back instead.
        if "fractal_pool" in obj:
            continue
        matrices[obj.name] = Matrix.Translation(offset) @ obj.matrix_world
        tokens[obj.name] = mesh_token(obj)
        # (A pushed object its drivers haven't placed yet reads a scale of 0: left as it is)
        if "camera_push_scale" in obj and obj.scale.x and obj["camera_push_scale"]:
            pushes[obj.name] = obj.scale.x / obj["camera_push_scale"]
            matrices[obj.name] = scaled_about(eye, 1.0 / pushes[obj.name]) @ matrices[obj.name]
        for modifier in obj.modifiers:
            if modifier.type == 'NODES' and modifier.node_group:
                for item in modifier.node_group.interface.items_tree:
                    if item.item_type == 'SOCKET' and item.in_out == 'INPUT' and item.socket_type == 'NodeSocketFloat':
                        socket = getattr(modifier.properties.inputs, item.identifier)
                        inputs[obj.name, modifier.name, item.identifier] = socket.value
    return matrices, inputs, pushes, tokens


def scaled_about(point, factor):
    """A matrix scaling by factor about point"""
    return Matrix.Translation(point) @ Matrix.Scale(factor, 4) @ Matrix.Translation(-point)


def world_differences(old, new, limit=6):
    """What changed between two world_snapshots: objects moved, turned or scaled, and modifier inputs, for POP events."""
    (old_matrices, old_inputs, _old_pushes, old_tokens), (new_matrices, new_inputs, _new_pushes, new_tokens) = old, new
    changed = []
    for name, matrix in new_matrices.items():
        before = old_matrices.get(name)
        if before is None:
            changed.append(f"{name} appeared")
            continue
        if old_tokens.get(name) != new_tokens.get(name):
            changed.append(f"{name} rebuilt")
        moved = matrix.translation - before.translation
        if moved.length > 1e-9 * max(1.0, matrix.translation.length):
            far = abs(moved.z) > 1e5        # the map's patches hide by moving 1e7 down
            changed.append(f"{name} {'shown' if far and moved.z > 0 else 'hidden' if far else 'moved'}"
                           + ("" if far else f" by {tuple(round(c, 4) for c in moved)}"))
        elif abs(matrix.to_scale().x - before.to_scale().x) > 1e-9 * matrix.to_scale().x:
            changed.append(f"{name} scaled x{matrix.to_scale().x / before.to_scale().x:.4g}")
    for key, value in new_inputs.items():
        if old_inputs.get(key) != value:
            changed.append(f"{key[0]} {key[2]} {old_inputs.get(key)} -> {value:.4g}")
    return "; ".join(changed[:limit]) + (f"; and {len(changed) - limit} more" if len(changed) > limit else "")


def render_with_world(m, scene, snapshot, path):
    """Render the current camera view with the map put back to an earlier world_snapshot, then restore it. Comparing
    that with the frame as rendered separates what the map did (a level switching, a patch moving) from what the
    camera did (moving, passing close to something). An object whose mesh was written again since stays as it is: its
    new points are from its new middle, which it's moved to (a floor's rebuild round the camera), and its old matrix
    with them drew the floor a frame's rebuild away, a slab across the view (every smooth torus rebuild a fake pop,
    2026-11-24)."""
    matrices, inputs, pushes, tokens = snapshot
    offset = Vector(scene.drone_rig.world_offset)
    # The pushed layers, pushed as they were, but about the camera now, as the map's drivers do
    eye = offset + m.drone_camera.matrix_world.translation
    objects = [scene.objects[name] for name in matrices
               if name in scene.objects and mesh_token(scene.objects[name]) == tokens.get(name)]
    saved_basis = {obj.name: obj.matrix_basis.copy() for obj in objects}
    saved_inputs = {}
    muted = []
    for obj in objects:
        if obj.animation_data:
            for fcurve in obj.animation_data.drivers:
                if not fcurve.mute:
                    fcurve.mute = True
                    muted.append(fcurve)
    for (name, modifier, identifier), value in inputs.items():
        socket = getattr(scene.objects[name].modifiers[modifier].properties.inputs, identifier)
        saved_inputs[name, modifier, identifier] = socket.value
        socket.value = value
    # Parents first, so each child is placed against its parent's restored matrix
    depth = lambda obj: 0 if obj.parent is None else 1 + depth(obj.parent)
    for level in sorted({depth(obj) for obj in objects}):
        for obj in objects:
            if depth(obj) == level:
                pushed = scaled_about(eye, pushes[obj.name]) if obj.name in pushes else Matrix()
                obj.matrix_world = Matrix.Translation(-offset) @ pushed @ matrices[obj.name]
        bpy.context.view_layer.update()
    # Rendering runs the frame change handlers again, and an endless fractal's (the add-on's fractal.follow_eyes()) would
    # move its eye, put back with the map, to the camera: every fractal flight read 0.00% map change
    eyes = [h for h in bpy.app.handlers.frame_change_pre if getattr(h, "__name__", "") == "follow_eyes"]
    for handler in eyes:
        bpy.app.handlers.frame_change_pre.remove(handler)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    bpy.app.handlers.frame_change_pre.extend(eyes)
    for obj in objects:
        obj.matrix_basis = saved_basis[obj.name]
    for (name, modifier, identifier), value in saved_inputs.items():
        getattr(scene.objects[name].modifiers[modifier].properties.inputs, identifier).value = value
    for fcurve in muted:
        fcurve.mute = False
    bpy.context.view_layer.update()


def save_pixels(pixels, path):
    """Write an (h, w, 3) array from load_pixels() as a PNG"""
    height, width = pixels.shape[:2]
    image = bpy.data.images.new("saved_pixels", width, height)
    image.pixels.foreach_set(np.concatenate([pixels[::-1], np.ones((height, width, 1))], axis=2)
                             .astype(np.float32).ravel())
    image.filepath_raw = path
    image.file_format = 'PNG'
    image.save()
    bpy.data.images.remove(image)


def load_pixels(path, width, height):
    image = bpy.data.images.load(path, check_existing=False)
    pixels = np.array(image.pixels[:]).reshape(height, width, 4)[::-1, :, :3]
    bpy.data.images.remove(image)
    return pixels


def invalid_drivers():
    """Drivers Blender has switched off: once an expression errors (dividing by zero, log of 0 or NaN), Blender stops
    evaluating it for the rest of the session, which freezes whatever it drives."""
    blocks = [*bpy.data.objects, *bpy.data.materials, *bpy.data.node_groups, *bpy.data.cameras, *bpy.data.meshes,
              *(mat.node_tree for mat in bpy.data.materials if mat.node_tree)]
    return [f"{block.name}: {fcurve.data_path}[{fcurve.array_index}] {fcurve.driver.expression!r}"
            for block in blocks if block.animation_data for fcurve in block.animation_data.drivers
            if not fcurve.driver.is_valid]


def block_change(a, b):
    h, w = a.shape[0] // 8 * 8, a.shape[1] // 8 * 8
    blocks = np.abs(a[:h, :w] - b[:h, :w]).reshape(h // 8, 8, w // 8, 8, 3).mean(axis=(1, 3, 4))
    return 100.0 * (blocks > 0.08).mean()


def make_video(frames_dir, out_path, fps):
    images = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    scene = bpy.data.scenes.new("FlightVideo")
    first = bpy.data.images.load(os.path.join(frames_dir, images[0]))
    r = scene.render
    r.resolution_x, r.resolution_y = (s - s % 2 for s in first.size)
    r.resolution_percentage, r.fps, r.fps_base = 100, round(fps), round(fps) / fps
    scene.view_settings.view_transform = 'Standard'
    r.image_settings.media_type = 'VIDEO'
    r.image_settings.file_format = 'FFMPEG'
    r.ffmpeg.format, r.ffmpeg.codec, r.ffmpeg.constant_rate_factor = 'MPEG4', 'H264', 'HIGH'
    r.filepath = out_path
    strip = scene.sequence_editor_create().strips.new_image("flight", os.path.join(frames_dir, images[0]), channel=1,
                                                           frame_start=1, fit_method='FIT')
    for name in images[1:]:
        strip.elements.append(name)
    scene.frame_end = len(images)
    bpy.ops.render.render(animation=True, scene=scene.name)


def follow_recording(camera, recorded):
    """Point everything that follows the drone camera (constraints, and driver variables on objects, materials and
    node groups) at the recorded camera instead."""
    for obj in bpy.data.objects:
        for constraint in obj.constraints:
            if getattr(constraint, "target", None) == camera:
                constraint.target = recorded
    blocks = [*bpy.data.objects, *bpy.data.materials, *bpy.data.node_groups, *bpy.data.cameras, *bpy.data.meshes,
              *(mat.node_tree for mat in bpy.data.materials if mat.node_tree)]
    for block in blocks:
        if block.animation_data:
            for fcurve in block.animation_data.drivers:
                for variable in fcurve.driver.variables:
                    for target in variable.targets:
                        if target.id == camera:
                            target.id = recorded


def replay(args):
    """Render a flight recorded with the add-on's Record Camera, frame by frame from the recorded camera, with the map
    following it as it followed the drone, and check each frame like a simulated flight. Loops, wraps and Keep Drone at
    Origin moves are in the recording (as jumps of the camera, and keyed delta_location of the map's top objects)."""
    scene = bpy.context.scene
    m = load_addon(args.addon)
    m.drone_obj = scene.drone_rig.drone
    m.drone_camera = next(c for c in m.drone_obj.children if c.type == 'CAMERA')
    m.clip_end_obj = next((c for c in m.drone_camera.children if c.name == 'clip_end'), None)
    recorded = scene.objects[args.replay]
    follow_recording(m.drone_camera, recorded)
    first, last = (int(f) for f in recorded.animation_data.action.frame_range)

    out = args.out or os.path.join(os.path.dirname(bpy.data.filepath), "replay_" + recorded.name)
    frames_dir = os.path.join(out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for name in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, name))
    width, height = (int(v) for v in args.size.split("x"))
    r = scene.render
    if args.engine == "workbench":
        r.engine = 'BLENDER_WORKBENCH'
        viewport_shading(scene)
    elif r.engine == 'BLENDER_EEVEE':
        scene.eevee.taa_render_samples = 1
    r.resolution_x, r.resolution_y, r.resolution_percentage = width, height, 100
    r.image_settings.media_type = 'IMAGE'
    r.image_settings.file_format = 'PNG'
    scene.camera = recorded
    for obj in (m.drone_obj, m.clip_end_obj):
        if obj:
            obj.hide_render = True
    snap_path = os.path.join(out, "snap.png")
    roots = [obj for obj in scene.objects if not obj.parent and obj not in (m.drone_obj, recorded)]
    fps = r.fps / r.fps_base
    print(f"REPLAY {recorded.name}: frames {first}-{last} ({(last - first) / fps:.1f} s)")

    events, rows, changes, map_changes = [], [], [], {}
    previous = None              # (camera matrix, scale, root positions, image, world)
    # An endless fractal's eye follows the drone camera (the add-on's fractal.follow_eyes()): here, the recording
    fractals = [obj for obj in scene.objects if hasattr(getattr(m, "fractal", None), "follow") and m.fractal.is_fractal(obj)]
    for handler in [h for h in bpy.app.handlers.frame_change_pre if getattr(h, "__name__", "") == "follow_eyes"]:
        bpy.app.handlers.frame_change_pre.remove(handler)
    for frame in range(first, last + 1, args.every):
        scene.frame_set(frame)
        if fractals:
            for obj in fractals:
                m.fractal.follow(obj, recorded.matrix_world)
            bpy.context.view_layer.update()
        bpy.context.view_layer.update()
        t = (frame - first) / fps
        matrix = recorded.matrix_world.copy()
        scale = matrix.to_scale().x
        root_positions = [obj.matrix_world.translation.copy() for obj in roots]
        path = os.path.join(frames_dir, f"{frame:06d}.png")
        r.filepath = path
        bpy.ops.render.render(write_still=True)
        pixels = load_pixels(path, width, height)
        change, seam = 0.0, None
        if previous is not None:
            change = block_change(pixels, previous[3])
            ratio = scale / previous[1]
            moved = (matrix.translation - previous[0].translation).length
            if ratio > 1.5 or ratio < 1 / 1.5:
                seam = f"loop x{ratio:.3g}"
            elif moved > 1000 * recorded.data.clip_start:
                seam = f"wrap or jump by {moved:.4g}"
            elif any((a - b).length > 1e-6 for a, b in zip(root_positions, previous[2])):
                seam = "Keep Drone at Origin moved the world"
            if seam:
                events.append((t, "SEAM?" if "Keep" not in seam else "ORIGIN", f"frame {frame}: {seam}, "
                                                                                f"{change:.1f}% of the view changed"))
            else:
                render_with_world(m, scene, previous[4], snap_path)
                map_change = block_change(pixels, load_pixels(snap_path, width, height))
                map_changes[frame] = map_change
                if map_change > POP_MIN:
                    os.replace(snap_path, os.path.join(out, f"pop_{frame:06d}_before.png"))
                    what = world_differences(previous[4], world_snapshot(m, scene, exclude=(recorded,)))
                    events.append((t, "POP", f"frame {frame}: the map changed {map_change:.1f}% of the view. "
                                             f"Changed: {what}"))
        previous = (matrix, scale, root_positions, pixels, world_snapshot(m, scene, exclude=(recorded,)))
        changes.append((t, frame, change))
        loc = matrix.translation
        rows.append([f"{t:.3f}", frame, f"{loc.x:.6g}", f"{loc.y:.6g}", f"{loc.z:.6g}", f"{scale:.6g}",
                     f"{recorded.data.clip_end:.6g}", f"{change:.2f}", f"{map_changes.get(frame, float('nan')):.2f}"])

    values = [c for _t, _f, c in changes]
    for i, (t, frame, change) in enumerate(changes[1:], start=1):
        around = values[max(1, i - JUMP_WINDOW):i] + values[i + 1:i + 1 + JUMP_WINDOW]
        if len(around) >= 4:
            typical = float(np.median(around))
            if change > max(JUMP_RATIO * typical, typical + JUMP_MIN):
                cause = f"; the map did {map_changes[frame]:.1f}% of it" if frame in map_changes else ""
                events.append((t, "JUMP", f"frame {frame}: {change:.1f}% of the view changed, {typical:.1f}% around "
                                          f"it{cause}"))
    for broken in invalid_drivers():
        events.append(((last - first) / fps, "DRIVER", f"switched off after an error: {broken}"))
    events.sort(key=lambda e: e[0])
    with open(os.path.join(out, "log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "frame", "x", "y", "z", "scale", "clip_end", "view_change_%", "map_change_%"])
        writer.writerows(rows)
    with open(os.path.join(out, "events.txt"), "w") as f:
        for t, kind, text in events:
            f.write(f"{t:8.3f}  {kind:8s} {text}\n")
    for t, kind, text in events:
        print(f"EVENT {t:8.3f}  {kind:8s} {text}")
    if os.path.exists(snap_path):
        os.remove(snap_path)
    count = lambda kind: sum(1 for _t, k, _ in events if k == kind)
    print(f"REPLAY done: {len(rows)} frames rendered, max map change {max(map_changes.values(), default=0):.2f}%; "
          f"{count('SEAM?')} loops/wraps, {count('POP')} pops, {count('JUMP')} jumps. Output in {out}")
    if args.video and rows:
        make_video(frames_dir, os.path.join(out, "replay.mp4"), fps / args.every)
        print(f"REPLAY video {os.path.join(out, 'replay.mp4')}")


def main():
    args = parse_args()
    if args.replay:
        replay(args)
        return
    with open(args.plan) as f:
        plan = json.load(f)
    exact_points(plan.get("start", {}))
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.plan)),
                                   "flight_" + os.path.splitext(os.path.basename(args.plan))[0])
    frames_dir = os.path.join(out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for name in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, name))

    m = load_addon(args.addon)
    scene = bpy.context.scene
    for key, value in plan.get("settings", {}).items():
        setattr(scene.control_settings, key, value)
    for key, value in plan.get("other_settings", {}).items():
        setattr(scene.flight_options, key, value)
    # Custom properties on objects, e.g. a map's spin turned off, so the pop check only sees level changes
    for name, properties in plan.get("object_properties", {}).items():
        for key, value in properties.items():
            scene.objects[name][key] = value
    # Record Camera needs its camera object
    name = scene.flight_options.record_obj
    if scene.flight_options.record_camera and name and name not in scene.objects:
        scene.collection.objects.link(bpy.data.objects.new(name, bpy.data.cameras.new(name)))

    clock = types.SimpleNamespace(t=0.0)
    m.time = types.SimpleNamespace(perf_counter=lambda: clock.t)
    controller = VirtualController(plan.get("sticks", []), scene)
    m.controls.read_axes = lambda _controller: controller.axes(clock.t)

    width, height = (int(v) for v in args.size.split("x"))
    snap_path = os.path.join(out, "snap.png")

    def snap():
        if args.timing:
            return np.zeros((height, width, 3))
        scene.render.filepath = snap_path
        bpy.ops.render.render(write_still=True)
        return load_pixels(snap_path, width, height)

    def snap_world(snapshot):
        """The view with the map as in an earlier world_snapshot()"""
        if args.timing:
            return np.zeros((height, width, 3))
        render_with_world(m, scene, snapshot, snap_path)
        return load_pixels(snap_path, width, height)

    events = []
    # start.track: the point the harness keeps aiming at, in start_origin()'s coordinates, moved along with the drone
    # camera by loops, wraps and recenters (the same scaling and shift turns where it was looking into where it looks)
    tracked = {}

    def moved(scale, before, after, anchor_before=None, steps=None):
        if "exact" in tracked:
            # An endless map: the point from the map's origin, moved exactly as the add-on moves the drone (endless.py):
            # scaled about the origin by the loop's power of the factor, by a climbing loop's remainder, and back by a
            # recenter's steps. Following the camera in float32 instead, its rounding grew 3 times with every loop,
            # and 8 loops on it was steering the drone off the point. (Random digits the add-on makes up move where
            # the origin is in the lattice, not the drone or the point.)
            settings = tracked["map"]["endless"]
            factor, step = int(settings["factor"]), Fraction(repr(float(settings["step"])))
            axes = [int(axis) for axis in settings.get("axes", (0, 1))]
            point = list(tracked["exact"])
            if steps is not None:
                for axis, moved_steps in zip(axes, steps):
                    point[axis] -= moved_steps * step
            elif scale != 1.0:
                power = round(math.log(scale) / math.log(factor))
                if power > 0:
                    point = [c * factor ** power for c in point]
                else:
                    divisor = factor ** -power
                    point = [c / divisor for c in point]
                    for axis, a in zip(axes, anchor_before):
                        point[axis] += (a % divisor) * step / divisor
            tracked["exact"] = point
        elif "point" in tracked:
            to_local = start_origin(plan["start"]).inverted()
            tracked["point"] = scale * (tracked["point"] - to_local @ before) + to_local @ after

    if args.seed is not None:
        random.seed(args.seed)
    started = time.perf_counter()
    op = make_operator(m, events, clock, snap, plan.get("start", {}).get("click_select"), moved, out, snap_world)
    # Real flights start wherever the timeline was left, not at the simulation's start frame, so a cleared simulation
    # has nothing to show for a moment
    scene.frame_set(int(plan.get("start_frame", 1000)))
    start_flight(m, op, clock, plan.get("continue", False))
    if plan.get("start"):
        place_drone(m, plan["start"])

    r = scene.render
    if args.engine == "workbench":
        r.engine = 'BLENDER_WORKBENCH'
        viewport_shading(scene)
    elif r.engine == 'BLENDER_EEVEE':
        scene.eevee.taa_render_samples = 1       # like a viewport frame in motion
    r.resolution_x, r.resolution_y, r.resolution_percentage = width, height, 100
    r.image_settings.media_type = 'IMAGE'
    r.image_settings.file_format = 'PNG'
    scene.camera = m.drone_camera
    for obj in (m.drone_obj, m.clip_end_obj):
        if obj:
            obj.hide_render = True

    fps = scene.render.fps / scene.render.fps_base
    duration = float(plan.get("duration", 5.0))
    next_tick, next_frame, frame_count = 0.0, 1.0 / fps, 0
    rows, changes, previous_image, previous_world, map_changes = [], [], None, None, {}
    last_offset, last_selected, last_enclosed = tuple(scene.drone_rig.world_offset), "", False
    print(f"FLIGHT {os.path.basename(args.plan)}: {duration:g} s at {fps:g} fps, add-on {args.addon} ({m.__name__})")

    start = plan.get("start", {})
    turn_after_click = "click_at" in start and "look_at" in start
    probe = None
    if args.probe:
        probe = types.ModuleType("probe")
        probe.__file__ = os.path.abspath(args.probe)
        exec(compile(open(args.probe).read(), args.probe, "exec"), probe.__dict__)
        # (Before the first tick, which builds what the map shows first: wrappers set here see the start too)
        if hasattr(probe, "setup"):
            probe.setup(m)
    if args.realtime:
        importlib.import_module(m.__package__ + ".background").enabled = True
    wall_start = time.perf_counter()
    # The main thread's time: each frame's ticks (ms) and its frame step (the frame_change handlers: the simulation's
    # step), and how long until the first frame (the start: its first tick builds what the map shows first)
    tick_ms, step_ms, first_frame = [0.0], [], None
    while min(next_tick, next_frame) <= duration:
        if next_tick <= next_frame:
            clock.t = next_tick
            timer = time.perf_counter()
            op.tick(bpy.context)
            tick_ms[-1] += (time.perf_counter() - timer) * 1000
            next_tick += TICK
            if turn_after_click and clock.t >= start.get("turn_at", 0.1):
                # The origin may have moved, so aim from where the camera is now
                origin = start_origin(start)
                here = origin.inverted() @ m.drone_camera.matrix_world.translation
                aim(m, origin, here, start["look_at"])
                turn_after_click = False
                events.append((clock.t, "TURN", "the harness turned the camera to look_at (not a jump cut)"))
                if start.get("track"):
                    tracked["point"] = Vector(start["look_at"])
                    relative = bpy.context.scene.objects.get(start.get("relative_to", ""))
                    if relative is not None and "endless" in relative:
                        tracked["map"] = relative
                        tracked["exact"] = (start.get("look_at_exact")
                                            or [Fraction(repr(float(c))) for c in start["look_at"]])
            elif "point" in tracked:
                if "exact" in tracked:
                    tracked["point"] = Vector([float(c) for c in tracked["exact"]])
                # Back onto the line through the point along the view: the part of the camera's offset from the point
                # that's across the view
                to_world = start_origin(start)
                camera = to_world.inverted() @ m.drone_camera.matrix_world.translation
                view = (to_world.inverted().to_3x3() @ (m.drone_camera.matrix_world.to_3x3() @ Vector((0, 0, -1))))
                view.normalize()
                offset = camera - tracked["point"]
                across = offset - offset.dot(view) * view
                # Held at a surface short of the point (stop_at_surface), the drone stops closing in, and every loop
                # doubles its distance to the point. Over TRACK_LIMIT of its scale away, float32's error in the view
                # direction, at that distance, put the drone meters off the line every tick, and moving it back there
                # threw it about: auto scale's distance jumped, it looped by 4 and by 1/16, and the Sierpinski dive's
                # loops had seams. So the pilot lets go there.
                scale = m.drone_camera.matrix_world.to_scale().z
                if offset.length > TRACK_LIMIT * scale:
                    events.append((clock.t, "TRACK", f"the pilot let go: the point is {offset.length / scale:.3g} times "
                                                     f"the drone's scale away"))
                    del tracked["point"]
                    continue
                if offset.length > 0.0:
                    tracked["worst"] = max(tracked.get("worst", 0.0), across.length / offset.length)
                m.drone_obj.matrix_world = Matrix.Translation(to_world.to_3x3() @ -across) @ m.drone_obj.matrix_world
                # Seen by the camera now, or the next tick's hooks read the move as a wrap and moved the point with it
                bpy.context.view_layer.update()
            continue

        clock.t = next_frame
        if args.realtime:
            behind = clock.t - (time.perf_counter() - wall_start)
            if behind > 0.0:
                time.sleep(behind)
        camera_before = m.drone_camera.matrix_world.translation.copy()
        timer = time.perf_counter()
        scene.frame_set(scene.frame_current + 1)     # frame_change_pre feeds the sticks to the simulation
        step_ms.append((time.perf_counter() - timer) * 1000)
        if first_frame is None:
            first_frame = time.perf_counter() - started
        tick_ms.append(0.0)
        next_frame += 1.0 / fps
        frame_count += 1
        if probe is not None:
            probe.frame(clock.t, frame_count, m, scene)
        # The add-on moves the Drone object onto the simulated drone once it's drifted RECENTER_DISTANCE from it, and
        # the simulation subtracts the same offset, so the camera shouldn't move
        physics = m.physics
        if physics.is_flying() and physics.recentered and physics.recentered[0] == scene.frame_current:
            moved_by = m.drone_camera.matrix_world.translation - camera_before
            events.append((clock.t, "SIMRECENTER", f"the core recentered by "
                                                   f"{tuple(round(c, 2) for c in physics.recentered[1])} (drone "
                                                   f"units); the camera moved {moved_by.length:.4g} this frame"))

        state = camera_state(m, scene)
        offset = tuple(scene.drone_rig.world_offset)
        if offset != last_offset:
            events.append((clock.t, "ORIGIN", "Keep Drone at Origin moved the world"))
            last_offset = offset
        if state["selected"] != last_selected:
            events.append((clock.t, "SELECT", state["selected"] or "(none)"))
            last_selected = state["selected"]
        if frame_count % args.every:
            continue

        depsgraph = bpy.context.evaluated_depsgraph_get()
        inside = enclosure(m, scene, depsgraph)
        enclosed = inside >= 0.6
        if enclosed != last_enclosed:
            events.append((clock.t, "INSIDE" if enclosed else "OUTSIDE",
                           f"{inside:.0%} of directions see faces from behind"))
            last_enclosed = enclosed

        path = os.path.join(frames_dir, f"{frame_count:05d}.png")
        r.filepath = path
        bpy.ops.render.render(write_still=True)
        pixels = load_pixels(path, width, height)
        change = block_change(pixels, previous_image) if previous_image is not None else 0.0
        previous_image = pixels
        # How much of the view did the map change since the last rendered frame? The frame re-rendered with the map as
        # it was then, from this frame's camera. Not across a loop or wrap, which move the camera against the map on
        # purpose: their seams are measured as they happen.
        seam_between = changes and any(kind in ("LOOP", "WRAP") and changes[-1][0] < when for when, kind, _ in events)
        if previous_world is not None and not seam_between:
            render_with_world(m, scene, previous_world, snap_path)
            map_change = block_change(pixels, load_pixels(snap_path, width, height))
            map_changes[frame_count] = map_change
            if map_change > POP_MIN:
                # Keep the view as the map had it, next to the frame, to look at
                os.replace(snap_path, os.path.join(out, f"pop_{frame_count:05d}_before.png"))
                what = world_differences(previous_world, world_snapshot(m, scene))
                events.append((clock.t, "POP", f"frame {frame_count}: the map changed {map_change:.1f}% of the view "
                                               f"(before: pop_{frame_count:05d}_before.png). Changed: {what}"))
        # Brings the objects' matrices up to date first: a property set before the flight (like a spin turned off)
        # otherwise only reached them a frame later, which read as a pop
        bpy.context.view_layer.update()
        previous_world = world_snapshot(m, scene)
        changes.append((clock.t, frame_count, change))
        loc = state["location"]
        rows.append([f"{clock.t:.3f}", frame_count, f"{loc.x:.6g}", f"{loc.y:.6g}", f"{loc.z:.6g}",
                     f"{state['scale']:.6g}", f"{state['clip_end']:.6g}", state["selected"], f"{change:.2f}",
                     f"{inside:.2f}", " ".join(f"{c}={controller.values[c]:+.2f}" for c in CONTROLS
                                               if controller.values[c]),
                     " | ".join(text for text in readout_texts(m).values() if text)])

    # Jump cuts: frames that changed far more than the frames around them. Steady fast flight changes a lot every frame,
    # so a fixed threshold won't do.
    values = [c for _t, _f, c in changes]
    for i, (t, frame, change) in enumerate(changes[1:], start=1):
        around = values[max(1, i - JUMP_WINDOW):i] + values[i + 1:i + 1 + JUMP_WINDOW]
        if len(around) >= 4:
            typical = float(np.median(around))
            if change > max(JUMP_RATIO * typical, typical + JUMP_MIN):
                cause = ""
                previous_t = changes[i - 1][0]
                seams = [kind for when, kind, _text in events if kind in ("LOOP", "WRAP") and previous_t < when <= t]
                if any(kind == "TURN" and previous_t < when <= t for when, kind, _text in events):
                    continue                 # the harness turning the camera after the click
                if seams:
                    # The map check means nothing across a loop or wrap, which moves the camera on purpose
                    cause = f"; at a {' and '.join(sorted(set(seams))).lower()}, so that seam isn't seamless"
                elif frame in map_changes:
                    # Under a few percent, the map didn't change: the camera moved, or passed close to something
                    cause = f"; the map did {map_changes[frame]:.1f}% of it"
                events.append((t, "JUMP", f"frame {frame}: {change:.1f}% of the view changed, {typical:.1f}% "
                                          f"around it{cause}"))

    with open(os.path.join(out, "log.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "frame", "x", "y", "z", "scale", "clip_end", "selected", "view_change_%",
                         "inside", "sticks", "readouts", "map_change_%"])
        writer.writerows(row + [f"{map_changes.get(row[1], float('nan')):.2f}"] for row in rows)
    if len(step_ms) > 2:
        ticks, steps = np.array(tick_ms[1:-1]), np.array(step_ms[1:])
        frames = ticks + steps[:len(ticks)]
        print(f"TIMING start {first_frame:.2f} s; tick ms/frame median {np.median(ticks):.1f} p99 "
              f"{np.percentile(ticks, 99):.0f} max {ticks.max():.0f}; frame step median {np.median(steps):.1f} p99 "
              f"{np.percentile(steps, 99):.0f}; frames over 33 ms {(frames > 33).sum()}, over 100 ms "
              f"{(frames > 100).sum()}, worst {frames.max():.0f}")
    if probe is not None and hasattr(probe, "report"):
        probe.report(out)
    for broken in invalid_drivers():
        events.append((duration, "DRIVER", f"switched off after an error: {broken}"))
    events.sort(key=lambda e: e[0])
    with open(os.path.join(out, "events.txt"), "w") as f:
        for t, kind, text in events:
            f.write(f"{t:8.3f}  {kind:8s} {text}\n")
    for t, kind, text in events:
        print(f"EVENT {t:8.3f}  {kind:8s} {text}")
    if os.path.exists(snap_path):
        os.remove(snap_path)
    count = lambda kind: sum(1 for _t, k, _ in events if k == kind)
    print(f"FLIGHT {count('DRIVER')} drivers switched off by errors")
    if "worst" in tracked:
        print(f"FLIGHT track: the biggest correction was {tracked['worst']:.2e} of the distance to the point")
    print(f"FLIGHT done: {frame_count} frames, {len(rows)} rendered, median view change "
          f"{np.median(values) if values else 0:.1f}%, max map change "
          f"{max(map_changes.values(), default=0):.2f}%; {count('LOOP')} loops, {count('WRAP')} wraps, "
          f"{count('SEAM')} bad seams, {count('POP')} pops, {count('JUMP')} jumps. Output in {out}")
    if args.video and rows:
        make_video(frames_dir, os.path.join(out, "flight.mp4"), fps / args.every)
        print(f"FLIGHT video {os.path.join(out, 'flight.mp4')}")
    if args.save_as:
        # The flight's end, as cancel() ends it: a recording of inputs gets its keyframes' handles worked out
        physics = getattr(m, "physics", None)
        if physics is not None and getattr(physics, "recording", None) is not None:
            physics.stop()
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_as), copy=True)
        print(f"FLIGHT saved {args.save_as}")


main()
# Leave without Blender's normal shutdown: with EEVEE, GPU_exit() can wait forever on a shader
# compile it's cancelling, which once left a finished flight spinning for hours.
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
