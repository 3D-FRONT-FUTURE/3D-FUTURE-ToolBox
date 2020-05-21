'''
Render Scene idmap image
'''
import argparse, sys, os, time
import logging
import numpy as np
import json
import math
import urllib.request
import bpy
import pdb
import random
import colorsys

# add current path to env
sys.path.append(os.getcwd())
from mathutils import Matrix
from math import radians
from utils.blender_utils import *

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors

def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors

def init_blender():
    scene = bpy.context.scene
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1200
    scene.render.resolution_percentage = 100

    # Delete default cube
    for obj in bpy.data.objects:
        obj.select = True
    bpy.ops.object.delete()


def transform_object(trans_vec, rot_mat, index, colors, name='obj'):
    rgb = colors[index]
    blender_rgb = np.array(rgb) / 255
    
    for i, obj in enumerate(bpy.context.selected_objects):
        if name is not None:
            if len(bpy.context.selected_objects) == 1:
                obj.name = name
            else:
                obj.name = name + '_' + str(i)

        # material color
        if len(obj.data.materials) == 0:
            mat = bpy.data.materials.new(obj.name)
            mat.diffuse_color = (blender_rgb[0], blender_rgb[1], blender_rgb[2])
            mat.diffuse_shader = 'FRESNEL'
            mat.diffuse_intensity = 1.0
            mat = bpy.data.materials[obj.name]
            obj.data.materials.append(mat)
        else:
            obj.active_material.diffuse_color = (blender_rgb[0], blender_rgb[1], blender_rgb[2])

        for mtl in range(len(bpy.data.materials)):
            bpy.data.materials[mtl].use_shadeless = True

        color_inferior = obj.active_material.diffuse_color
        
        real_rgb = float(255.999 * pow(color_inferior.r, 1 / 2.2)), float(255.999 * pow(color_inferior.g, 1 / 2.2)), float(255.999 * pow(color_inferior.b, 1 / 2.2))
        
        # transformation
        trans_4x4 = Matrix.Translation(trans_vec)
        rot_4x4 = Matrix(rot_mat).to_4x4()
        scale_4x4 = Matrix(np.eye(4))
        obj.matrix_world = trans_4x4 * rot_4x4 * scale_4x4

    return real_rgb

def render_image(save_file):
    for o in bpy.data.objects:
        if o.type == 'CAMERA':
            bpy.context.scene.camera = o
            break
    
    scene = bpy.context.scene
    scene.render.filepath = save_file
    bpy.ops.render.render(write_still=True)  # render still

def render_function(model_pose_infos, model_dir, image_dir):
    
    # 1. set color value for object 
    instance_color_dict = []
    colors = ncolors(len(model_pose_infos))
    
    for index, model_pose_info in enumerate(model_pose_infos):
        # 2. get object params
        model_id = model_pose_info['shape_id']
        trans_vec = np.array(model_pose_info['translation'])
        rot_mat = np.array(model_pose_info['rotation'])
        fov = math.radians(model_pose_info['fov'])

        # 3. load object
        obj_file = os.path.join(model_dir, model_id + '.obj')
        if not os.path.exists(obj_file): 
            print('obj file not exists: ', obj_file)
        bpy.ops.import_scene.obj(filepath=obj_file)
        
        # 4. transform object and set color
        transform_object(trans_vec, rot_mat, index, colors, name=model_id + '_'+str(index))
    
    # 5. set camera
    camera = add_camera((0, 0, 0), fov, 'camera')
    
    # 6. world lighting
    world = bpy.data.worlds['World']
    world.light_settings.use_ambient_occlusion = True
    world.light_settings.ao_factor = 1.0
    
    # 7. get intrinsic matrix
    K_blender = get_calibration_matrix_K_from_blender(camera.data)
    K = np.array(K_blender)
    np.save('data/scene_data/intrinsic.npy', K)

    # 8. render scene image
    save_file = os.path.join(args.output_folder, 'render_image')
    render_image(save_file)

    # 9. clear sys
    clear_scene()


   
############## main function
model_pose_info_file = 'data/scene_data/scene_pose_info.npy'
model_pose_infos = np.load(model_pose_info_file, allow_pickle=True)

model_dir = 'data/scene_data/models'
img_dir = 'data/scene_data/images'

'''
img_dir = 'data/temp/image/69169_raw.png'
model_dir = 'data/temp/models/69169'
model_pose_info_file = '/home/shunming/choose_sample/pose_info_dict.npy'
all_model_pose_infos = np.load(model_pose_info_file, allow_pickle=True).item()
model_pose_infos = all_model_pose_infos['69169']
'''

# render and projection process
init_blender()
render_function(model_pose_infos,model_dir, img_dir)

