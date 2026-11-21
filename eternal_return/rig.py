"""The drone's rig, built in code: the Drone, a flat triangle the core moves (physics.py), and the Drone_Camera,
parented to the triangle's three vertices and tilted up by the scene's Camera Tilt (a driver). The HUD is drawn
(hud.py, hud_draw.py).
"""
import math

import bpy

from . import physics

MESH_NAME = "_ER_Drone"
# The rig's triangle, about its center, in the Drone object's space: 20 cm wide, 20 cm deep
TRIANGLE = ((-0.1, -1.0 / 15.0, 0.0), (0.1, -1.0 / 15.0, 0.0), (0.0, 2.0 / 15.0, 0.0))

# The camera's tilt driver: its expression, and what its variable reads
TILT_EXPRESSION = "90*pi/180+var"
TILT_PATH = "drone_physics.camera_tilt"


def build_drone(scene):
    """A new Drone, at rest at the world origin, turned with its triangle's normal (Z) along world -Y"""
    mesh = bpy.data.meshes.new(MESH_NAME)
    mesh.from_pydata(TRIANGLE, [], [(0, 1, 2)])
    physics.keep_rest(mesh)
    drone_obj = bpy.data.objects.new("Drone", mesh)
    drone_obj.rotation_euler = (-math.pi / 2, 0.0, 0.0)
    scene.collection.objects.link(drone_obj)
    if physics.available():
        physics.write(drone_obj, physics.at_rest())
    return drone_obj


def build_camera(scene, drone_obj, clip_start):
    """A new Drone_Camera on drone_obj's triangle, a 10 mm lens with the add-on's clip range from clip_start"""
    camera_data = bpy.data.cameras.new(name="Drone_Camera")
    camera_data.lens = 10
    camera_data.clip_start = clip_start
    camera_data.clip_end = clip_start * 100_000
    camera_obj = bpy.data.objects.new("Drone_Camera", camera_data)
    scene.collection.objects.link(camera_obj)
    camera_obj.rotation_euler[0] = math.pi / 2
    camera_obj.parent = drone_obj
    camera_obj.parent_type = 'VERTEX_3'
    camera_obj.parent_vertices = [0, 1, 2]
    drive_camera_tilt(camera_obj, scene)
    return camera_obj


def drive_camera_tilt(camera_obj, scene):
    """Have camera_obj's X rotation follow the scene's Camera Tilt, in place of any driver it had there"""
    animation = camera_obj.animation_data or camera_obj.animation_data_create()
    old = animation.drivers.find("rotation_euler", index=0)
    if old is not None:
        animation.drivers.remove(old)
    driver = animation.drivers.new("rotation_euler", index=0).driver
    driver.type = 'SCRIPTED'
    var = driver.variables.new()
    var.name = "var"
    var.type = 'SINGLE_PROP'
    var.targets[0].id_type = 'SCENE'
    var.targets[0].id = scene
    var.targets[0].data_path = TILT_PATH
    driver.expression = TILT_EXPRESSION
