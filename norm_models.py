"""
normalize obj and remove same instance
"""
import os
import numpy as np
import copy
import shutil
import pdb
import argparse

from utils.utils import *

def main():
    model_dir = 'data/scene_data/models/'
    model_names = os.listdir(model_dir)
    for model_name in model_names:
        model_file = os.path.join(model_dir, model_name)
        save_file = os.path.join(model_dir, model_name.split('.')[0] + '_norm.obj')
        try:
            # get obj vertex
            mesh_vertics = get_obj_vertex_ali(model_file) 
            norm_mesh_vertics = normalize_vertex(mesh_vertics)
        except:
            print(model_file)
            continue
        replace_and_save_obj(norm_mesh_vertics.tolist(), model_file, save_file)


if __name__ == '__main__':
    main()

