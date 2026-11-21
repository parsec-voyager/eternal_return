"""The add-on's names in files saved before it was Eternal Return (2026-11-20), when it was the Zero Gravity Drone, a
fork of DroneCam installed as "dronecam": the keys it put on the data it made, and the names of that data. They're
renamed to the add-on's own names when a file is opened (load_post), and in the open file when the add-on is enabled,
so the rest of the add-on only ever looks for the new ones. The renames are in a file once it's saved.

The scene's settings groups (scene.references, scene.drone_physics, ...) kept their attributes, so they need nothing.
Names that only ever mark DroneCam's own data (its rig's node groups, its HUD's objects, rig.py and hud.py) stay as they
are: those are found to be taken out, and nothing makes them any more."""
import bpy

from . import hud_render
from . import odometer
from . import parts
from . import physics
from . import rig

# Old key -> new key, by the bpy.data collection whose data carries it
KEYS = {
    "meshes": {"zero_gravity_rest": physics.REST_KEY},
    "actions": {"zero_gravity_recording": physics.RECORDING_KEY},
    "scenes": {"dronecam_unit_log2": odometer.UNIT_KEY},
    "objects": {"dronecam_hud_render": hud_render.OBJECT_KEY, "dronecam_hud_render_index": hud_render.INDEX_KEY},
    "node_groups": {"dronecam_version": parts.VERSION_KEY},
}
# Old name -> new name, by collection: data named so, or so with Blender's ".001" kind of suffix
NAMES = {
    "meshes": {"_DroneCam_Drone": rig.MESH_NAME},
    "node_groups": {"_DroneCam_Parts": parts.NODE_GROUP},
    "materials": {"_DroneCam_Parts": parts.MATERIAL},
}


def plain(value):
    """An ID property's value as plain Python, to be set under another key (a group or array can't be set as is)"""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "to_list"):
        return value.to_list()
    return value


def rename_all():
    """Rename the old keys and names in the open file to the new ones. Returns how many were renamed."""
    renamed = 0
    for collection, keys in KEYS.items():
        for data in getattr(bpy.data, collection):
            # Linked data is read-only here, and its own file gets renamed when it's opened
            if data.library is not None:
                continue
            for old, new in keys.items():
                if old in data:
                    if new not in data:
                        data[new] = plain(data[old])
                    del data[old]
                    renamed += 1
    for collection, names in NAMES.items():
        for data in getattr(bpy.data, collection):
            if data.library is not None:
                continue
            for old, new in names.items():
                if data.name == old or data.name.startswith(old + "."):
                    data.name = new + data.name[len(old):]
                    renamed += 1
    return renamed


@bpy.app.handlers.persistent
def on_load(_filepath=None):
    """load_post, and a timer once: rename the file's old names (before the other modules' load_post handlers look for
    them). Returns None, so the timer doesn't run again."""
    rename_all()


def register():
    if on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load)
    # The file open when the add-on is enabled: bpy.data can't be written during registration, so right after it
    bpy.app.timers.register(on_load, first_interval=0.0)


def unregister():
    if on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load)
