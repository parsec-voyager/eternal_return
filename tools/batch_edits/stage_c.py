"""Stage C's re-save (2026-11-21), as an edit for batch_update.py: DroneCam's HUD out from under each scene's drone
camera, its switches kept as the scene's HUD settings (hud.migrate()), DroneCam's HUDs anywhere else in the file
deleted, and _DroneCam_ node groups nothing uses removed. The old names and keys need nothing here: names.py renames
them when the map is opened, and the save keeps that. A map already done changes nothing, so this also checks maps
for anything left from before."""
import bpy


def edit(addon):
    hud, rig = addon("hud"), addon("rig")
    changes = []
    for scene in bpy.data.scenes:
        references = scene.references
        if references.drone_obj is None:
            continue
        view_layer = scene.view_layers[0]
        view_layer.update()
        if rig.migrate(scene, references.drone_obj, references.camera_obj, view_layer.depsgraph):
            changes.append("rig migrated")
        if hud.migrate(scene, references.camera_obj):
            changes.append("HUD migrated")
    for obj in [obj for obj in bpy.data.objects if hud.legacy_hud(obj.parent) == obj]:
        hud.delete_legacy(obj)
        changes.append("other DroneCam HUD deleted")
    # Groups nothing uses but each other drop only one level per save, so they're removed here until none are left
    while True:
        leftovers = [group for group in bpy.data.node_groups if group.name.startswith("_DroneCam_")]
        user_map = bpy.data.user_map(subset=leftovers)
        unused = [group for group in leftovers if not user_map[group]]
        if not unused:
            break
        changes += [f"unused node group {group.name.split('.')[0]} removed" for group in unused]
        bpy.data.batch_remove(unused)
    return changes
