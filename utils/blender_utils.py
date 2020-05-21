import bpy
import bpy_extras
import numpy as np

from math import radians
from mathutils import Matrix

def get_calibration_matrix_K_from_blender(camd):
    '''
    get camera intrinsic matrix
    '''
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_width_in_mm
        #s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

def transfrom_object(trans_vec, rot_mat, scale, name='obj'):
    '''
    transform object using rotation and translation
    '''
    for i, obj in enumerate(bpy.context.selected_objects):
        if name is not None:
            if len(bpy.context.selected_objects) == 1:
                obj.name = name
            else:
                obj.name = name + '_' + str(i)

        # Compute world matrix
        trans_4x4 = Matrix.Translation(trans_vec)
        rot_4x4 = Matrix(rot_mat).to_4x4()
        scale_4x4 = Matrix(np.eye(4))
        obj.matrix_world = trans_4x4 * rot_4x4 * scale_4x4
        print(obj.matrix_world)

def clear_scene():
    '''
    clear blender system for scene and object with pose, refer to 3D-R2N2
    '''
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern="RotCenter")
    bpy.ops.object.select_pattern(pattern="Lamp*")
    bpy.ops.object.select_pattern(pattern="Camera")
    bpy.ops.object.select_all(action='INVERT')
    bpy.ops.object.delete()

    # The meshes still present after delete
    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)
    for item in bpy.data.materials:
        bpy.data.materials.remove(item)

def clear_mv():
    '''
    clear blender system for object mv
    '''
    for object in bpy.context.scene.objects:
        if object.name in ['Camera']:
            object.select = False
        else:
            object.select = True
    bpy.ops.object.delete()  

    # The meshes still present after delete
    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)
    for item in bpy.data.materials:
        bpy.data.materials.remove(item)


def add_camera(xyz=(0,0,0), fov=1, name=None, proj_model='PERSP', sensor_fit='HORIZONTAL'):
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object

    cam.rotation_euler[0] = radians(0)
    cam.rotation_euler[1] = radians(0)
    cam.rotation_euler[2] = radians(0)

    if name is not None:
        cam.name = name
    
    cam.location = xyz
    cam.data.type = proj_model
    cam.data.angle = fov
    cam.data.sensor_fit = sensor_fit

    return cam


