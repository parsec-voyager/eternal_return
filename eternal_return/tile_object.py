"""Tile Selected Object (phase 7 of the engines' plan, 2026-11-25): any object made an endless surface map from the
sidebar, the map build scripts' setup moved into the add-on. The object's shape is laid out once, now, as the map's
mesh (layout.py): as about so many quads, near square on the shape (Blender's QuadriFlow, squared(), and on a smooth
surface fit()), or its own faces as they are. That mesh is the map's top level, every face splitting factor x factor
forever below it (surface.py), with a floor motif on every level's cells and open frames standing on the lattice
(designs.py's looks), and a scaling config that loops and scales the drone with it. The object itself is hidden, not
changed; Ctrl+Z takes it all back.

make_map() is the whole setup, without the button: what a build script calls for the same map.
"""

import bpy
import numpy as np
from mathutils import Vector

from . import designs
from . import layout
from . import parts
from . import surface
from . import surface_substitution

# The map's settings ("surface") other than its factor and chart, as every surface map's
MAP_SETTINGS = {"rho": 1.0, "window": 1.25, "floor_cells": 4.0}
# The settle that draws the map before the button returns: at most this many seconds of the frames' builds
SETTLE_BUDGET = 10.0


def smooth_chart(sides):
    """The smooth surface (subdivision.py's) a mesh of faces with these numbers of sides has: Catmull-Clark's for
    quads, Loop's for triangles, Stam and Loop's for both"""
    if all(s == 4 for s in sides):
        return "catmull_clark"
    if all(s == 3 for s in sides):
        return "loop"
    return "stam_loop"


def own_faces(obj, depsgraph):
    """obj's own faces (modifiers applied, in world coordinates) as a map's mesh: faces of more than 4 sides cut into
    triangles, and vertices and edges on no face left out (the map's lattice is its faces')"""
    import bmesh
    mesh = layout.world_mesh(obj, depsgraph)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bmesh.ops.delete(bm, geom=[e for e in bm.edges if not e.link_faces], context='EDGES')
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    if not len(mesh.polygons):
        bpy.data.meshes.remove(mesh)
        raise ValueError(f"{obj.name} has no faces to tile")
    if not layout.manifold(mesh):
        bpy.data.meshes.remove(mesh)
        raise ValueError(f"{obj.name}'s faces meet three or more at an edge: lay it out as quads instead")
    return mesh


def fitted(mesh, chart):
    """mesh's points moved so its smooth surface (chart) passes through them (layout.fit()): a subdivision surface
    shrinks inside its mesh"""
    wanted = layout.points_of(mesh)
    top = surface.Topology(None, positions=wanted, faces=[tuple(p.vertices) for p in mesh.polygons], chart=chart)
    points, _miss, _rounds = layout.fit(top.chart, wanted)
    mesh.vertices.foreach_set("co", points.astype(np.float32).ravel())
    mesh.update()


def viewer_point(context):
    """Where the map is seen from: the drone camera, else the 3D view's eye, else None"""
    camera = context.scene.drone_rig.camera
    if camera is not None:
        return Vector(camera.matrix_world.translation)
    region = getattr(context, "region_data", None)
    if region is not None:
        return Vector(region.view_matrix.inverted().translation)
    return None


def forget_builds():
    """Every map's builds, levels and meshes kept in memory let go, as after loading a file: a map made again under a
    name an undone one had (the button's redo) mustn't be drawn from that one's"""
    surface.forget()
    parts.forget()


def make_map(context, obj, quads=True, tiles=layout.LAYOUT_FACES, smooth=True, floor="strapwork", frames=True,
             factor=2, merge=False):
    """obj made an endless surface map (see the module's docstring): its shape laid out as about tiles quads, or its own
    faces (quads False); on its smooth surface (smooth) or its faces' flat ones; with the floor motif (designs.FLOORS)
    and the open frames (frames), each level's cells factor x factor of the one above's. Returns (the map's object,
    what was done: faces, corners of other than 4 faces, chart)."""
    scene = context.scene
    depsgraph = context.evaluated_depsgraph_get()
    if quads:
        mesh, done = layout.laid_out(obj, int(tiles), chart="catmull_clark" if smooth else "bilinear", merge=merge)
    else:
        mesh, done = own_faces(obj, depsgraph), {}
    sides = [len(p.vertices) for p in mesh.polygons]
    chart = smooth_chart(sides) if smooth else "bilinear"
    if chart != "bilinear" and not (quads and chart == "catmull_clark"):
        # (laid_out() fit its quads already)
        fitted(mesh, chart)
    designs.face_tones(mesh)
    forget_builds()

    collection = obj.users_collection[0] if obj.users_collection else scene.collection
    name = obj.name + " Map"
    mesh.name = name + " Source"
    source = bpy.data.objects.new(mesh.name, mesh)
    collection.objects.link(source)
    source.hide_viewport = source.hide_render = source.hide_select = True

    # The anchor, where the map is exact from: the vertex nearest where it's seen from
    points = layout.points_of(mesh)
    eye = viewer_point(context)
    vertex = int(np.argmin(np.linalg.norm(points - np.array(eye), axis=1))) if eye is not None else 0
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = 'PLAIN_AXES'
    # Drawn as nothing: the drone's clicks select the map, and its cross showed in the selection's orange
    empty.empty_display_size = 0.0
    collection.objects.link(empty)
    empty[surface.PROPERTY] = {**MAP_SETTINGS, "factor": int(factor), "chart": chart}
    empty[surface.SOURCE] = source
    world = surface.World(0, (surface.VERTEX, vertex, 0, 0), 1.0, surface.topology_of(empty), int(factor))
    # The map's origin is its anchor's point (on a smooth chart the vertex's limit position)
    empty.location = Vector(surface.d_float(world.anchor_point()))
    empty[surface.STATE] = world.text()
    empty[surface.VERSION] = 1
    if frames:
        empty[parts.PROPERTY] = designs.frames_settings()
    motif = designs.floor_settings(floor)
    if motif is not None:
        empty[surface_substitution.PROPERTY] = motif

    config = bpy.data.texts.new(name + " config")
    config.write(designs.scaling_config(int(factor)))
    empty["scaling_config"] = config
    if "default_config" not in bpy.data.texts:
        # What the drone goes back to when it lets go of the map (flight.py applies it on deselecting)
        import os
        with open(os.path.join(os.path.dirname(__file__), "default_config.json")) as f:
            bpy.data.texts.new("default_config").write(f.read())

    # The original hidden, as it's drawn and as it renders: the map draws its shape now
    obj.hide_set(True)
    obj.hide_render = True
    for other in context.selected_objects:
        other.select_set(False)
    empty.select_set(True)
    context.view_layer.objects.active = empty
    context.view_layer.update()
    if eye is not None and surface.drone_clip_end() > 0.0:
        # Drawn as the drone sees it now (else at the next flight's start)
        surface.settle(empty, eye)
        if frames:
            parts.settle(empty, eye, budget=SETTLE_BUDGET)
    valences = np.bincount([len(f) for f in surface.topology_of(empty).vertex_faces])
    done = {"faces": len(mesh.polygons), "corners": {k: int(c) for k, c in enumerate(valences) if c and k != 4},
            "chart": chart, **{k: v for k, v in done.items() if k == "voxels"}}
    return empty, done


def tileable(obj):
    """Whether obj has a shape to tile, and isn't a map already or a part of one"""
    return (obj is not None and obj.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'}
            and not surface.is_surface(obj) and obj.get(surface.FLOOR_KEY) is None
            and obj.get(parts.OWNER_KEY) is None and obj.name not in {"Drone", "Drone_Camera"})


class ER_OT_TileObject(bpy.types.Operator):
    """Make the selected object an endless map: its surface tiled, every tile splitting forever, with a floor motif and
    3D frames at every level. The object is hidden, not changed"""
    bl_idname = "er.tile_object"
    bl_label = "Tile Selected Object"
    bl_options = {'REGISTER', 'UNDO'}

    layout_kind: bpy.props.EnumProperty(
        name="Tiles", description="What the map's biggest tiles are",
        items=(('QUADS', "Laid Out", "About so many quads laid over the shape, near square everywhere on it"),
               ('FACES', "Own Faces", "The object's own faces as they are (more than 4 sides cut into triangles)")),
        default='QUADS')
    tiles: bpy.props.IntProperty(name="Tile Count", description="About how many quads to lay the shape out in: fewer "
                                 "make bigger tiles, and the surface follows the shape less closely",
                                 default=layout.LAYOUT_FACES, min=50, soft_max=5000)
    smooth: bpy.props.BoolProperty(name="Smooth", description="Fly over the faces' smooth subdivision surface, not "
                                   "their flat ones", default=True)
    merge: bpy.props.BoolProperty(name="Merge Parts", description="Make overlapping parts one surface first (slower; "
                                  "a shape that isn't one closed surface is merged anyway)", default=False)
    floor: bpy.props.EnumProperty(name="Floor", description="The motif drawn on every level's tiles",
                                  items=designs.FLOORS, default="strapwork")
    frames: bpy.props.BoolProperty(name="Frames", description="Open 3D frames (posts, beams, horseshoe arches) "
                                   "standing on every level's tiles", default=True)
    factor: bpy.props.IntProperty(name="Split", description="How many ways each level's tiles split along a side "
                                  "at the next (the map loops by its powers)", default=2, min=2, max=5)

    @classmethod
    def poll(cls, context):
        return (context.mode == 'OBJECT' and tileable(context.active_object)
                and not getattr(context.window_manager, "er_fly_modal_running", False))

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=320)

    def draw(self, context):
        column = self.layout.column()
        column.use_property_split = True
        column.prop(self, "layout_kind")
        if self.layout_kind == 'QUADS':
            column.prop(self, "tiles")
            column.prop(self, "merge")
        column.prop(self, "smooth")
        column.prop(self, "floor")
        column.prop(self, "frames")
        column.prop(self, "factor")

    def execute(self, context):
        obj = context.active_object
        wm = context.window_manager
        wm.progress_begin(0, 1)
        try:
            empty, done = make_map(context, obj, quads=self.layout_kind == 'QUADS', tiles=self.tiles,
                                   smooth=self.smooth, floor=self.floor, frames=self.frames, factor=self.factor,
                                   merge=self.merge)
        except (ValueError, RuntimeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        finally:
            wm.progress_end()
        odd = ", ".join(f"{count} of {k}" for k, count in sorted(done["corners"].items()))
        message = f"{empty.name}: {done['faces']} tiles" + (f", corners {odd}" if odd else "") + f"; {obj.name} hidden"
        space = context.space_data
        if (space is not None and space.type == 'VIEW_3D' and space.shading.type == 'SOLID'
                and space.shading.color_type != 'VERTEX'):
            # The map's colors are its meshes' "Col" attributes, which Solid mode shows colored by Attribute (by
            # Material, the whole map is its one material's color)
            space.shading.color_type = 'VERTEX'
            message += "; Solid view colored by Attribute"
        self.report({'INFO'}, message)
        return {'FINISHED'}


OPERATORS = (ER_OT_TileObject,)
