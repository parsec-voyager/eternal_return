import bpy

for obj in bpy.data.objects:
    if obj.modifiers:
        for modifier in list(obj.modifiers):
            if modifier.type == 'SUBSURF':
                obj.modifiers.remove(modifier)
                # bpy.context.view_layer.objects.active = obj
                # obj.modifier_apply(modifier=modifier.name)
                # bpy.context.view_layer.objects.active = None
                print(f"Removed {modifier.type} Surface from: {obj.name}")
