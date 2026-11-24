"""A survey, no changes: which of the add-on's engines each map's objects use (as notes), so retiring an engine's code
can be checked against every map first (2026-11-23, phase 5 step 3)."""
import bpy

KEYS = ("endless", "substitution", "surface_substitution", "surface", "sphere", "parts", "fractal", "mandelbrot")


def edit(addon):
    found = []
    for obj in bpy.data.objects:
        keys = [k for k in obj.keys() if k in KEYS]
        if not keys:
            continue
        extra = ""
        if "endless" in keys:
            s = obj["endless"].to_dict() if hasattr(obj["endless"], "to_dict") else {}
            extra = f" (endless factor {s.get('factor')}, tiling {'yes' if 'tiling' in s else 'no'})"
        found.append(f"note: {obj.name}: {', '.join(keys)}{extra}")
    for scene in bpy.data.scenes:
        if scene.get("mandelbrot_object"):
            found.append(f"note: scene {scene.name}: mandelbrot")
    return found
