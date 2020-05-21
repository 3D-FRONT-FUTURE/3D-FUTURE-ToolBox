'''
Project 3D object into 2D image plane for scene image

Used for segementaion dataset and scene images in retrieval and reconstruction dataset.
'''
import os,pdb
import cv2
import math
import open3d as o3d
import numpy as np
import scipy.linalg as linalg

from utils.utils import *
from scipy.spatial.transform import Rotation as R_


def projection(model_pose_infos, K, model_dir, img_file, save_dir):
    img = cv2.imread(img_file)
    h_size,w_size, channel = img.shape
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[0][2]
    
    for index, model_pose_info in enumerate(model_pose_infos):
        model_id = model_pose_info['shape_id']
        trans_vec = np.array(model_pose_info['translation'])
        rot_mat = np.array(model_pose_info['rotation'])
        fov = math.radians(model_pose_info['fov'])
        
        #pdb.set_trace()
        obj_file = os.path.join(model_dir, model_id + '.obj')
        mesh_vertices = get_obj_vertex_ali(obj_file)
        
        # transformation w->c
        mesh_vertices_trans = np.transpose(np.dot(rot_mat, np.transpose(mesh_vertices)))
        mesh_vertices_trans = mesh_vertices_trans - (-trans_vec)
        
        # project 3d->2d
        X, Y, Z = mesh_vertices_trans.T
        h = (-Y) / (-Z) * fy + cy
        w = X / (-Z) * fx + cx
        
        # draw projection points
        h = np.minimum(np.maximum(h, 0), h_size - 1)
        w = np.minimum(np.maximum(w, 0), w_size - 1)
        img[np.round(h).astype(int), np.round(w).astype(int), 2] = 0
        img[np.round(h).astype(int), np.round(w).astype(int), 1] = 255

    save_file = os.path.join(save_dir, 'projection_scene.png')
    cv2.imwrite(save_file, img)

def main():
    model_pose_info_file = 'data/scene_data/scene_pose_info.npy'
    intrinsic_file = 'data/scene_data/intrinsic.npy' 
    model_pose_infos = np.load(model_pose_info_file, allow_pickle=True)
    K = np.load(intrinsic_file, allow_pickle=True)

    model_dir = 'data/scene_data/models'
    save_dir = 'demo_results/scene'
    img_file = 'data/scene_data/images/raw_scene.png'

    projection(model_pose_infos, K, model_dir, img_file, save_dir)

if __name__ == '__main__':
    main()


