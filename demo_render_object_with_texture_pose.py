'''
render object in scene using provided pose info
This code can be used for retrieval and reconstruction (background images only)

pose_info_dict - obtained from train_set.json for both retrieval and reconstruction dataset

Note: image size must be 1200x1200, if used for retrieval or reconstruction with background
'''
import bpy
import pdb
import random
import argparse, sys, os, time
import logging
import numpy as np
import json
from mathutils import Matrix
import math

# add current path to env
sys.path.append(os.getcwd())
from utils.blender_utils import *
from math import radians

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.0,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)


### setting
# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_environment = True
bpy.context.scene.render.image_settings.file_format = args.format
bpy.context.scene.render.image_settings.color_depth = args.color_depth

# Clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# Create input render layer node.
render_layers = tree.nodes.new('CompositorNodeRLayers')

depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.label = 'Depth Output'
if args.format == 'OPEN_EXR':
    links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
else:
    # Remap as other types can not represent the full range of depth.
    normalize = tree.nodes.new(type="CompositorNodeNormalize")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    links.new(render_layers.outputs['Depth'], normalize.inputs[0])
    links.new(normalize.outputs[0], depth_file_output.inputs[0])

scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
scale_normal.blend_type = 'MULTIPLY'
# scale_normal.use_alpha = True
scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
bias_normal.blend_type = 'ADD'
# bias_normal.use_alpha = True
bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_normal.outputs[0], bias_normal.inputs[1])

normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = 'Normal Output'
links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
albedo_file_output.label = 'Albedo Output'
links.new(render_layers.outputs['Env'], albedo_file_output.inputs[0])

scene = bpy.context.scene
scene.render.resolution_x = 1200    # default 1200
scene.render.resolution_y = 1200    # default 1200
scene.render.resolution_percentage = 100

# Delete default cube
for obj in bpy.data.objects:
    obj.select = True
bpy.ops.object.delete()

def render_pose_function(trans_vec, rot_mat, fov, model_file, texture_file): 
    # loading model
    try: bpy.ops.import_scene.obj(filepath=model_file)
    except: return None
    
    # tranform object by pose
    transfrom_object(trans_vec, rot_mat, scale=1.0, name='obj')
    
    # camera and lighting config
    camera = add_camera((0, 0, 0), fov, 'camera')

    world = bpy.data.worlds['World']
    world.light_settings.use_ambient_occlusion = True
    world.light_settings.ao_factor = 1.0

    for o in bpy.data.objects:
        if o.type == 'CAMERA':
            bpy.context.scene.camera = o
            break
    
    # rendering
    model_id = model_file.split('/')[-1].split('.')[0]
    scene = bpy.context.scene
    scene.render.filepath = os.path.join(args.output_folder, model_id, 'texture_pose')
    bpy.ops.render.render(write_still=True)  # render still

    # delet objects in sys
    clear_scene()

# get from train_set
pose_info_dict = {
        'translation': [-1.903410504246893, -0.8875827469705174, -4.3289407945140415],
        'rotation': [[0.9187694361566392, 0.0, 0.394789780863919], [0.062338450260199324, 0.9874546445251975, -0.14507635600675325], [-0.3898370027251619, 0.15790289739461846, 0.9072431469806705]],
        'fov': 1.0462120316141805}

model_file = 'data/mv_data/models/0000000.obj'
texture_file = 'data/mv_data/texture/0000000.png'

render_pose_function(pose_info_dict['translation'], pose_info_dict['rotation'], pose_info_dict['fov'], model_file, texture_file)

