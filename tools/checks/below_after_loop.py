"""Blender -b <a surface map.blend> --python tools/checks/below_after_loop.py: a point below the drone (surface.Below)
must be the same point of its face after a loop of its world as before. A loop changes the World in place, and a
Below that read its base at the world's new level was 8 times nearer its face's corner after a dive by 8: the walk
that keeps the point below in its dip (Below.stay()) then started from the wrong place. PASS or FAIL."""
import importlib
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
importlib.import_module("eternal_return.flight")
surface = importlib.import_module("eternal_return.surface")

obj = next(o for o in bpy.context.scene.objects if surface.is_surface(o))
world = surface.world_of(obj)
below = surface.Below(world, surface.map_point(obj, bpy.data.objects["Drone_Camera"].matrix_world.translation))
level = world.n + 3
before = [float(c) for c in below.cell_place(level)]
surface.dive(world, 3)        # (the file isn't saved)
after = [float(c) for c in below.cell_place(level)]
print(f"below_after_loop: face {below.face}, in level {level}'s cells {before} before the loop, {after} after: "
      f"{'PASS' if before == after else 'FAIL'}")
