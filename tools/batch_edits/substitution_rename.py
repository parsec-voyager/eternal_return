"""The 2026-11-23 rename of surface_penrose.py to surface_substitution.py (its engine takes any substitution rule, Darb-i
Imam's too): a substitution surface map's settings move from the map object's "surface_penrose" property to
"surface_substitution", and its saved level objects' "surface_penrose_level_of" key to "surface_substitution_level_of"
(level objects are named "<map>.substitution" now, "<map>.penrose" before, with Blender's .001 numbering: renamed
with their meshes). Run on maps already renamed, it changes nothing, which is what the read-back checks."""
import re

import bpy

RENAMED = {"surface_penrose": "surface_substitution", "surface_penrose_level_of": "surface_substitution_level_of"}


def edit(addon):
    changes = []
    for obj in bpy.data.objects:
        for old, new in RENAMED.items():
            if old not in obj:
                continue
            value = obj[old]
            obj[new] = value.to_dict() if hasattr(value, "to_dict") else value
            del obj[old]
            changes.append(f"{obj.name}: {old} -> {new}")
        old_name = re.match(r"(.*)\.penrose(\.\d+)?$", obj.name)
        if "surface_substitution_level_of" in obj and old_name:
            name = old_name.group(1) + ".substitution" + (old_name.group(2) or "")
            changes.append(f"{obj.name} renamed {name}")
            obj.name = name
            if obj.type == 'MESH' and obj.data.users == 1:
                obj.data.name = name
    return changes
