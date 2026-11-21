"""An example edit for batch_update.py: turn off Outline Selected in every 3D view of every screen, so a selected
object (the one the drone clicked) isn't drawn with an orange outline while flying. Run on maps saved with it off, it
changes nothing, which is what the read-back checks."""
import bpy


def edit(addon):
    changes = []
    for screen in bpy.data.screens:
        for area in screen.areas:
            for space in area.spaces:
                if space.type == 'VIEW_3D' and space.overlay.show_outline_selected:
                    space.overlay.show_outline_selected = False
                    changes.append("Outline Selected turned off")
    return changes
