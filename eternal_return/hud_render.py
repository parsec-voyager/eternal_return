"""The HUD in renders (the scene's HUD setting In Renders): the drawn HUD (hud_draw.py) is the viewport's alone, so for
renders through the drone camera, its shapes and text are built as objects under the camera, seen only by renders.

- "Drone HUD", a mesh under the camera: the HUD's fills, and its lines as flat strips as thick as they're drawn, in
  the HUD's space (hud.py) turned into the camera's (it looks down -Z), so the object's scale is the HUD unit: a driver
  keeps it at the camera's clip start over hud.CLIP_START_DEPTH, in the camera's own scale, at any drone scale, with
  no Python
- a text object under it for each line of text, "Drone HUD Text"
- both in two materials glowing the HUD's colors (MATERIALS), so lighting doesn't change them

They're disabled in the viewport, where the drawn HUD shows, and made again from what the HUD shows before every frame
(update_handler(), after hud.replay_handler() has set it from the Drone's recording): a render of a recorded flight
shows the HUD as it was flown, its gimbals and readouts frame by frame. Changing the objects' data from a frame handler
while rendering needs the interface locked, so turning In Renders on locks it (Render > Lock Interface).

Everything sits a little further than the drawn HUD (RENDER_DEPTH), not to touch the camera's clip start, where a
render would cut it.
"""
import bpy
import numpy as np

from . import hud

OBJECT_KEY = "er_hud_render"
INDEX_KEY = "er_hud_render_index"
MESH_NAME = "Drone HUD"
TEXT_NAME = "Drone HUD Text"
# The HUD's color, its marks' and its deadzones' (hud_draw's Solid mode ones), as (name, color)
MATERIALS = (("Drone HUD", (0.72, 0.72, 0.72, 1.0)), ("Drone HUD Marks", (0.25, 0.25, 0.25, 1.0)),
             ("Drone HUD Deadzones", (0.52, 0.52, 0.52, 1.0)))
# HUD units further than the drawn HUD
RENDER_DEPTH = 0.0005


def in_renders_changed(scene):
    """In Renders changed: the objects made or deleted, the interface locked for renders"""
    settings = scene.drone_hud
    if settings.in_renders:
        scene.render.use_lock_interface = True
        update(scene)
    else:
        remove(scene)


def hud_object(camera_obj):
    return next((child for child in camera_obj.children if child.get(OBJECT_KEY) == "mesh"), None)


def material(index):
    name, color = MATERIALS[index]
    found = bpy.data.materials.get(name)
    if found is not None:
        return found
    found = bpy.data.materials.new(name)
    found.diffuse_color = color
    found.use_nodes = True
    nodes = found.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    output = nodes.new("ShaderNodeOutputMaterial")
    found.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return found


def ensure(scene, camera_obj):
    """The HUD's mesh object under the camera, made if missing"""
    obj = hud_object(camera_obj)
    if obj is not None:
        return obj
    mesh = bpy.data.meshes.new(MESH_NAME)
    for index in range(len(MATERIALS)):
        mesh.materials.append(material(index))
    obj = bpy.data.objects.new(MESH_NAME, mesh)
    obj[OBJECT_KEY] = "mesh"
    for collection in camera_obj.users_collection:
        collection.objects.link(obj)
    # Set from Python, the parent inverse stays the identity: the object sits in the camera's own space
    obj.parent = camera_obj
    obj.hide_viewport = True
    obj.hide_select = True
    for index in range(3):
        driver = obj.driver_add("scale", index).driver
        driver.type = 'SCRIPTED'
        clip = driver.variables.new()
        clip.name = "clip"
        clip.type = 'SINGLE_PROP'
        clip.targets[0].id_type = 'CAMERA'
        clip.targets[0].id = camera_obj.data
        clip.targets[0].data_path = "clip_start"
        size = driver.variables.new()
        size.name = "size"
        size.type = 'TRANSFORMS'
        size.targets[0].id = camera_obj
        size.targets[0].transform_type = 'SCALE_Z'
        size.targets[0].transform_space = 'WORLD_SPACE'
        driver.expression = f"clip / {hud.CLIP_START_DEPTH} / size"
    return obj


def strips(polylines):
    """Polylines (points, half their thickness) in the HUD's space as quads: a strip along each segment"""
    quads = []
    for points, half in polylines:
        points = np.asarray(points, dtype=np.float64)
        for start, end in zip(points[:-1], points[1:]):
            along = end[:2] - start[:2]
            length = np.hypot(*along)
            if length == 0.0:
                continue
            side = np.array([-along[1], along[0], 0.0]) / length * half
            quads.append([start - side, end - side, end + side, start + side])
    return quads


def to_camera(points):
    """HUD space points in the camera's (it looks down -Z), RENDER_DEPTH further"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return np.stack([points[:, 0], points[:, 1], -(points[:, 2] + RENDER_DEPTH)], axis=1)


def write_mesh(obj, shapes):
    """The HUD's fills and lines (its color), marks and deadzones (theirs) as the object's mesh, each at its own
    depth (hud.py)"""
    faces, materials = [], []
    for triangle in shapes["fills"]:
        faces.append(list(triangle))
        materials.append(0)
    for triangle in shapes["bands"]:
        faces.append(list(triangle))
        materials.append(2)
    for quad in strips(shapes["lines"]):
        faces.append(quad)
        materials.append(0)
    for quad in strips(shapes["marks"]):
        faces.append(quad)
        materials.append(1)
    vertices, polygons = [], []
    for face in faces:
        polygons.append(list(range(len(vertices), len(vertices) + len(face))))
        vertices.extend(to_camera(face).tolist())
    mesh = obj.data
    mesh.clear_geometry()
    mesh.from_pydata(vertices, [], polygons)
    mesh.polygons.foreach_set("material_index", materials)
    mesh.update()


def write_texts(obj, texts):
    """A text object under obj for each of texts (hud.shapes()), those left over hidden from renders"""
    objects = sorted((child for child in obj.children if child.get(OBJECT_KEY) == "text"),
                     key=lambda child: child[INDEX_KEY])
    for index, (text, anchor, size, align, on) in enumerate(texts):
        if index < len(objects):
            text_obj = objects[index]
        else:
            curve = bpy.data.curves.new(TEXT_NAME, 'FONT')
            curve.materials.append(material(0))
            text_obj = bpy.data.objects.new(TEXT_NAME, curve)
            text_obj[OBJECT_KEY] = "text"
            text_obj[INDEX_KEY] = index
            for collection in obj.users_collection:
                collection.objects.link(text_obj)
            text_obj.parent = obj
            text_obj.hide_viewport = True
            text_obj.hide_select = True
            objects.append(text_obj)
        curve = text_obj.data
        if curve.body != text:
            curve.body = text
        curve.size = size
        curve.align_x = align
        curve.align_y = 'CENTER' if on == 'MIDDLE' else 'TOP_BASELINE'
        text_obj.location = to_camera(anchor)[0]
        text_obj.hide_render = False
    for text_obj in objects[len(texts):]:
        text_obj.hide_render = True


def update(scene):
    """Make the HUD's objects show what the HUD shows now, with In Renders on"""
    settings = getattr(scene, "drone_hud", None)
    references = getattr(scene, "references", None)
    camera_obj = references.camera_obj if references else None
    if settings is None or not settings.in_renders or camera_obj is None:
        return
    obj = ensure(scene, camera_obj)
    shapes = hud.shapes(settings)
    write_mesh(obj, shapes)
    write_texts(obj, shapes["texts"])


def remove(scene):
    """Delete the scene's HUD objects, with their data"""
    for obj in [obj for obj in scene.objects if obj.get(OBJECT_KEY)]:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            (bpy.data.meshes if isinstance(data, bpy.types.Mesh) else bpy.data.curves).remove(data)


@bpy.app.handlers.persistent
def update_handler(scene, depsgraph=None):
    update(scene)


def register():
    # After hud.replay_handler, which sets what the HUD shows from a recording; and before a still render, which comes
    # with no frame change
    for handlers in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.render_pre):
        if update_handler not in handlers:
            handlers.append(update_handler)


def unregister():
    for handlers in (bpy.app.handlers.frame_change_pre, bpy.app.handlers.render_pre):
        if update_handler in handlers:
            handlers.remove(update_handler)
