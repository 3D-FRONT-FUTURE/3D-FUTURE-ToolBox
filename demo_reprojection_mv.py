'''
Reproject and transform object giving depth image
'''
import numpy as np
import pdb
import open3d as o3d
from scipy.spatial.transform import Rotation as R_
import scipy.linalg as linalg
import math
import cv2
import os

def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix


def euler2rotmat(angle, axis='x'):
    if axis == 'x': axis_vec = [1, 0, 0]
    elif axis == 'y': axis_vec = [0, 1, 0],
    elif axis == 'z': axis_vec = [0, 0, 1],

    radians = math.radians(angle)
    rot_mat = rotate_mat(axis_vec, radians)

    return rot_mat

def depth_to_point(depth, cam_K, cam_W, save_dir):
        N = 256
        M = 256
        # create pixel location tensor
        xx = np.arange(0, 256, 1)
        yy = np.arange(0, 256, 1)
        py, px = np.meshgrid(xx, yy)
        p = np.stack((px,256-py), axis=0)
        p = p.reshape(2, N*M)

        mask_idx = np.where(depth.reshape(-1) != 1.e10)

        depth[depth==1.e10] = 0
        d = depth.reshape(1, N*M)

        cx = 128.0
        cy = 128.0
        fx = 239.99998474
        fy = 239.99998474
        yy = d*(p[1] - cy) / fx
        xx = d*(p[0] - cx) / fy
        # create terms of mapping equation x = P^-1 * d*(qp - b)
        P = cam_K[:2, :2] 
        q = cam_K[2:3, 2:3]   
        b = cam_K[:2, 2:3].repeat(N*M, axis=1)
        Inv_P = np.linalg.inv(P)
        rightside = (p * q - b) * d
        x_xy = np.matmul(Inv_P, rightside)

        #x_xy = np.stack((xx,yy), axis=0)[:,0,:]
        x_xy = x_xy[:,mask_idx[0]]
        zz = d[0, mask_idx[0]]
        pcd_world = np.stack((-x_xy[0],x_xy[1],-zz),axis=1)

        pcd_world = pcd_world + cam_W[:,3]
        rot_z = np.array([[0,1,0],[-1,0,0],[0,0,1]])
        pcd_world = np.matmul(pcd_world, rot_z)

        cam_W[:,3] = [0,0,0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_world)
        o3d.io.write_point_cloud(os.path.join(save_dir, "reproject_pcd_world.ply"), pcd)

depth = cv2.imread('data/mv_data/reprojection/depth.exr', -1)
depth = np.array(depth[:,:,0])

# cam_k and cam_W both obationed from blender
cam_K = np.array([[239.99998474, 0, 128.0], [0, 239.99998474, 128.0], [0.0,0.0,1.0]])
cam_W = np.array([[-0.86602539, -0.12126785,  0.48507142,  1.60000062],
       [ 0.50000024, -0.21004197,  0.84016794,  2.771281  ],
       [ 0.        ,  0.97014248,  0.24253562,  0.80000001],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

Inv_W = np.linalg.inv(cam_W)

R = cam_W[0:3,0:3]
t = cam_W[0:3,3]
r = R_.from_matrix(R)

# compute transformation matrix (R, t)
x_angle, z_angle, y_angle = r.as_euler('xyz', degrees=True)  # xyz ->xzy
rot_mat_x = euler2rotmat(90-x_angle, axis='x')
rot_mat_z = euler2rotmat(z_angle, axis='z')
rot_mat_y = euler2rotmat(-y_angle, axis='y')
dist = np.sqrt(np.square(t[0]) + np.square(t[1]) + np.square(t[2]))
R = np.dot(rot_mat_x, rot_mat_y)
t = np.array([0,0,dist])
t = t.reshape(3, 1)
cam_W = np.concatenate((R, t), axis=1)

save_dir = 'demo_results/mv/reprojection'
if not os.path.exists(save_dir): os.makedirs(save_dir)

depth_to_point(depth,cam_K, cam_W, save_dir)



