'''
Project 3D object into 2D image plane for mv images

Used for retrieval and reconstruction dataset.

model_pose_info - same as the provided pose info
1. for FUTURE3D dataset, you can find model_pose_info in train_set.json for scene images and model_info.json for mv images;
2. for you own dataset, pose_info can be obatined from blender;

intrinsic:
1. for FUTURE3D dataset, intrinsic can be computeed using fov data and provided tool(/blender_utils.py);
2. for you own dataset, intrinsic can be computed through blender using our tool;
'''
import os,pdb
import json
import cv2
import math
import open3d as o3d
import numpy as np
import scipy.linalg as linalg

from utils.utils import *
from scipy.spatial.transform import Rotation as R_

def projection(model_pose_infos, K, model_dir, img_dir, save_dir):
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[0][2]
    
    model_file = os.path.join(model_dir, '0000000.obj')
    for view in range(12): # deafule 12 views
        img_file = os.path.join(img_dir, 'mv_0000000_' + str(view) + '.jpg')
        img = cv2.imread(img_file)
        h_size,w_size, channel = img.shape
        
        obj_file = model_file
        mesh_vertices = get_obj_vertex_ali(obj_file)
        
        view_pose_info = model_pose_infos['pose_' + str(view)]
        trans_vec = np.array(view_pose_info['translation'])
        rot_mat = np.array(view_pose_info['rotation'])
        
        # since blender coordinate is different with object coordination, we need to do additional 
        # transformation operations.
        # For mv image rendering, object rotate 90 degrees around x axis in blender. Thus euler angle should be changed as below:
        # blender_coordinate xyz -> object coordinate xzy (x = 90 - x', x: euler in x for object coordinate, x': euler in x for blender coordinate)
        
        # get rotation and translation in object coordinate
        r_blender = R_.from_matrix(rot_mat)
        x_angle, z_angle, y_angle = r_blender.as_euler('xyz', degrees=True) # xyz ->xzy blender_coordinate -> object coordinate
        r_object = R_.from_euler('yxz', [[-y_angle, 90 - x_angle, z_angle]], degrees=True)
        rot_mat_object = r_object.as_matrix()[0]
        dist = np.sqrt(np.square(trans_vec[0]) + np.square(trans_vec[1]) + np.square(trans_vec[2]))
        trans_vec_object = [0, 0, dist]
        
        mesh_vertices_trans = np.transpose(np.dot(rot_mat_object, np.transpose(mesh_vertices)))
        mesh_vertices_trans = mesh_vertices_trans - trans_vec_object
        
        # project 3d->2d
        X, Y, Z = mesh_vertices_trans.T
        h = (-Y) / (-Z) * fy + cy
        w = X / (-Z) * fx + cx
        
        # draw projection points
        h = np.minimum(np.maximum(h, 0), h_size - 1)
        w = np.minimum(np.maximum(w, 0), w_size - 1)
        img[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
        img[np.round(h).astype(int), np.round(w).astype(int), 1] = 255

        save_file = os.path.join(save_dir, 'view_' + str(view) + '.png')
        cv2.imwrite(save_file, img)

def main():
    model_pose_info_file = 'data/mv_data/mv_pose_info.json'
    intrinsic_file = 'data/mv_data/intrinsic.npy'

    with open(model_pose_info_file, 'r') as f:
        model_pose_infos = json.load(f)
    K = np.load(intrinsic_file, allow_pickle=True)

    model_dir = 'data/mv_data/norm_models'
    save_dir = 'demo_results/mv/projection'
    img_dir = 'data/mv_data/mv_images/'
    
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    projection(model_pose_infos, K, model_dir, img_dir, save_dir)

if __name__ == '__main__':
    main()


