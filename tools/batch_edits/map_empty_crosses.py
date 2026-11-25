"""An edit for batch_update.py: the map objects the drone selects, when they're empties (surface, sphere and endless
fractal maps, and any empty with a scaling config), drawn as nothing (empty_display_size 0). Selected, a plain-axes
empty's cross was drawn in the selection's orange while flying (the user, 2026-11-23: "i want the selected object to
be hidden, so that i dont see the orange lines in the cross pattern"). Hiding it would stop it being selected; drawn
at size 0 it's still there to select (the drone's clicks reach a map through its floor and parts, which name it).
The map builders make new ones so. Run again, it changes nothing."""
import bpy

SMALLEST = 0.0001


def edit(addon):
    surface, sphere, fractal = addon("surface"), addon("sphere"), addon("fractal")
    changes = []
    for obj in bpy.data.objects:
        # (Blender keeps an empty's size at least SMALLEST: set to 0, it's that)
        if obj.type != 'EMPTY' or obj.empty_display_size <= SMALLEST:
            continue
        if (surface.is_surface(obj) or sphere.is_sphere(obj) or fractal.is_fractal(obj)
                or obj.get("scaling_config") is not None):
            changes.append(f"{obj.name}: drawn as nothing (empty size {obj.empty_display_size:g} -> {SMALLEST:g})")
            obj.empty_display_size = 0.0
    return changes
